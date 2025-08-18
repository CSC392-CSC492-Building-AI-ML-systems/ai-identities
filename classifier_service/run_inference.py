import os
import sys
import pickle
import json
from typing import Callable
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import re


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCES_DIR = os.path.join(SCRIPT_DIR, 'resources')
VECTORIZER_PATH = os.path.join(RESOURCES_DIR, 'vectorizer.pkl')
LIBRARY_AVERAGES_PATH = os.path.join(RESOURCES_DIR, 'library_averages.pkl')
LLM_SET_PATH = os.path.join(RESOURCES_DIR, 'llm_set.json')
CONFIDENCE_THRESHOLD = 0.65


def load_inference_assets():
    """
    Loads the vectorizer, pre-computed library averages, and model metadata.
    """
    if not os.path.exists(VECTORIZER_PATH) or not os.path.exists(LIBRARY_AVERAGES_PATH):
        raise FileNotFoundError(
            "Required resources not found. Please run 'precompute_library.py' first and ensure "
            f"'vectorizer.pkl' is in the '{RESOURCES_DIR}' directory."
        )

    with open(VECTORIZER_PATH, 'rb') as f:
        try:
            vectorizer = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load vectorizer: {e}")

    with open(LIBRARY_AVERAGES_PATH, 'rb') as f:
        try:
            library_averages = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load library averages: {e}")

    try:
        with open(LLM_SET_PATH, 'r') as f:
            llm_set = json.load(f)

        llm_family_map = dict()
        for item in llm_set:
            llm_name = item['model_name']
            llm_name = llm_name.replace('/', '_', 1)
            llm_family_map[llm_name] = item['model_family']
    except Exception as e:
        raise RuntimeError(f"Failed to load the set of LLMs: {e}")

    return vectorizer, library_averages, llm_family_map


def _as_ndarray(x) -> np.ndarray:
    """Convert numpy.matrix or sparse mean result to (1, dim) ndarray."""
    x = np.asarray(x)
    if x.ndim == 1:            # flatten → row-vector
        x = x.reshape(1, -1)
    return x


def predict_with_scores(unknown_responses: list[str],
                        library_prompt_averages: dict[tuple[str, str], np.ndarray],
                        vectorizer: object, metric: Callable, top_k: int = 3,
                        do_normalize: bool = False) -> tuple[list[tuple[str, float]], dict[str, float]]:
    """
    Predicts top-k models for a list of unknown responses against pre-computed library averages for a single prompt.

    :param unknown_responses: A list of response strings from an unknown LLM.
    :param library_prompt_averages: A dict mapping (model, temp_bin) to its average vector for the specific prompt.
    :param vectorizer: The pre-fitted TF-IDF vectorizer.
    :param metric: A callable similarity/distance function.
    :param top_k: The number of top predictions to return.
    :param do_normalize: Whether to L2 normalize vectors.
    :return: A tuple containing:
             - A list of top-k (model, score) tuples.
             - A dictionary of all (model, score) pairs.
    """
    is_similarity = 'similarity' in metric.__name__

    # Vectorize the unknown responses and compute the average
    unknown_vectors = vectorizer.transform(unknown_responses)
    avg_unknown_vec = np.asarray(unknown_vectors.mean(axis=0))
    if do_normalize:
        avg_unknown_vec = normalize(avg_unknown_vec, norm='l2')

    # Compare against the library averages for the prompt
    known_llm_scores = {}
    known_llms = sorted(
        list(set(model for model, bin_name in library_prompt_averages.keys()))
    )

    for known_llm in known_llms:
        best_score = -np.inf if is_similarity else np.inf
        for bin_name in ['low', 'medium', 'high', 'overall']:
            key = (known_llm, bin_name)
            if key in library_prompt_averages:
                lib_vec = _as_ndarray(library_prompt_averages[key])
                if do_normalize:
                    lib_vec = normalize(lib_vec, norm='l2')

                score = metric(avg_unknown_vec, lib_vec)[0][0]

                if is_similarity:
                    best_score = max(best_score, score)
                else:
                    best_score = min(best_score, score)

        known_llm_scores[known_llm] = best_score

    # Sort models by the best score found across temp bins
    sorted_scores = sorted(known_llm_scores.items(), key=lambda item: item[1],
                           reverse=is_similarity)

    top_k_predictions = sorted_scores[:top_k]
    all_predictions = dict(sorted_scores)

    return top_k_predictions, all_predictions


def remove_cot_block(response: str, is_special_model: bool = False) -> str:
    """
    Robustly remove any <think>...</think> blocks from a response string, handling
    all positions (start/mid/end), nesting, and malformed/unpaired cases.
    Preserves content in incomplete/malformed cases to avoid data loss.

    :param response: Raw response string.
    :param is_special_model: If True, treat as model that skips <think> but uses </think>.
    :return: Cleaned response without CoT (never empty unless input was).
    """
    if not response:
        return response

    original = response

    # Iterative removal to handle nesting and multiples
    while True:
        # Find and remove complete paired blocks (non-greedy, anywhere)
        new_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        if new_response == response:
            break  # No more changes
        response = new_response

    # Handle unpaired/malformed after paired removal
    # Unpaired <think>: Remove tag, keep content after
    response = re.sub(r'<think>', '', response, flags=re.DOTALL)

    # Unpaired </think>: Remove tag, keep content before/after
    response = re.sub(r'</think>', '', response, flags=re.DOTALL)

    # Special model handling (assumes no <think>, but </think> closes initial CoT)
    if is_special_model:
        # If no </think>, keep all (incomplete)
        if '</think>' not in original:
            response = original  # Override to preserve

    response = response.strip()

    # Fallback to original if somehow emptied (rare)
    if not response:
        return original

    return response


def preprocess_responses(responses: list[str]) -> list[str]:
    """
    Preprocess a list of responses by detecting special model behavior and optionally removing CoT blocks.
    Returns cleaned responses and stats for debugging.

    :param responses: List of raw response strings.
    :return: List of cleaned response strings.
    """
    if not responses:
        return responses

    # Detect if this is a special model (never uses <think>, but uses </think> at least once)
    uses_think = any('<think>' in resp for resp in responses)
    uses_close = any('</think>' in resp for resp in responses)
    is_special = not uses_think and uses_close

    cleaned_responses = []
    for original_response in responses:
        cleaned_response = remove_cot_block(original_response, is_special)
        cleaned_responses.append(cleaned_response)

    return cleaned_responses


def identify_llm(responses: list[str]) -> dict:
    """
    Identifies an LLM based on a list of its responses.

    :param responses: A list of strings, where each string is a response from the unknown LLM.
    :return: A dictionary containing the prediction results.
    """
    cleaned_responses = preprocess_responses(responses)
    vectorizer, library_averages, llm_family_map = load_inference_assets()
    top_k_preds, all_scores = predict_with_scores(
        unknown_responses=cleaned_responses,
        library_prompt_averages=library_averages,
        vectorizer=vectorizer,
        metric=cosine_similarity,
        top_k=3
    )

    top_prediction, top_score = top_k_preds[0]
    if top_score >= CONFIDENCE_THRESHOLD:
        prediction_result = {
            "prediction": [(llm, round(score, 2)) for llm, score in top_k_preds],
            "all_scores": all_scores
        }
    else:
        top3_llm_families = []
        for llm, _ in top_k_preds:
            llm_family = llm_family_map[llm]
            if llm_family not in top3_llm_families and len(top3_llm_families) < 3:
                top3_llm_families.append(llm_family)

        prediction_result = {
            "prediction": ["unknown"],
            "top3_llm_families": top3_llm_families,
            "all_scores": all_scores
        }

    return prediction_result


if __name__ == '__main__':
    sample_responses_from_unknown_llm = [
        """
        **The Discovery of "Umbra": A Novel Gustatory Sensation and Its Implications for Gastronomy and Health**  

        The human palate is traditionally understood to perceive five primary tastes: sweet, salty, sour, bitter, and umami. However, recent theoretical advancements in gustatory neuroscience suggest the existence of a sixth, previously uncharacterized taste modality—**"umbra."** Derived from the Latin *umbra* (shadow), this taste evokes a paradoxical sensation of **cool depth**, reminiscent of the interplay between darkness and faint luminescence. It is neither bitter nor umami but occupies an intermediary space, eliciting a **lingering, velvety resonance** on the tongue, akin to the sensation of inhaling crisp night air while tasting something faintly metallic yet soothing.  
        
        ### **Sensory Profile and Physiological Perception**  
        Umbra is detected by a newly hypothesized subset of **TRPM8-adjacent taste receptors**, which respond to a unique class of **bioactive flavonoids** found in certain subterranean fungi, deep-sea algae, and select root vegetables. Unlike menthol, which merely stimulates cold-sensitive neurons, umbra engages both **thermosensory and chemosensory pathways**, producing a **synesthetic experience**—cool yet dense, faintly mineral but not astringent. Its aftertaste is prolonged, subtly altering the perception of subsequent flavors by muting excessive sweetness while enhancing savory undertones.  
        
        ### **Culinary Applications and Gastronomic Innovation**  
        Foods rich in umbra include:  
        1. **Nocturnal truffles** (*Tuber umbraticus*), a newly discovered fungal species thriving in low-oxygen forest substrates.  
        2. **Abyssal kelp** (*Thalassumbra profunda*), a bioluminescent algae harvested from mesopelagic zones.  
        3. **Shadow-root tubers** (*Scorzonera umbratica*), a cultivar selectively bred for high umbra-inducing glycosides.  
        
        In haute cuisine, umbra could serve as a **"flavor harmonizer,"** bridging disparate taste profiles. For instance, its cooling depth might balance the fiery intensity of capsaicin-laden dishes or temper the cloying richness of caramelized desserts. Experimental chefs might employ umbra-rich reductions as a **gastronomic "tonal base,"** much as bass frequencies anchor musical compositions.  
        
        ### **Sociocultural and Health Implications**  
        The introduction of umbra could revolutionize dietary practices by:  
        1. **Reducing sugar dependence**—its inherent complexity may satisfy cravings without excessive sweetness.  
        2. **Enhancing nutrient absorption**—early studies suggest umbra compounds upregulate **enteric serotonin receptors**, potentially improving gut-brain axis signaling.  
        3. **Shifting culinary paradigms**—its novelty may spur a **"neo-terroir" movement**, emphasizing rare, umbra-laden ingredients as luxury commodities.  
        
        However, ethical concerns arise regarding **sustainable harvesting** of deep-sea and fungal sources, necessitating biotechnological synthesis to prevent ecological disruption.  
        
        ### **Conclusion**  
        Umbra represents a frontier in **gustatory science**, challenging extant taxonomies of taste while offering profound opportunities for culinary arts and metabolic health. Further research must elucidate its receptor mechanisms and long-term dietary effects, but its potential to redefine human flavor perception is indisputable. The palate, it seems, has shadows yet to be illuminated.  
        
        (Word count: 798)
        """
    ]

    print("--- Running Inference ---")
    try:
        result = identify_llm(sample_responses_from_unknown_llm)
        print(json.dumps(result, indent=4))
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
