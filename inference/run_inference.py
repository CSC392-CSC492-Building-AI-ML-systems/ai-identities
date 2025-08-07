import os
import sys
import pickle
import json
# Add the project root directory to the Python path to allow imports from response_classifier
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from response_classifier.classifier_model_scripts.classifier_model import \
    predict_with_scores, get_metric_func
from response_classifier.classifier_model_scripts.llm_meta_data import load_llm_meta_data, \
    get_llm_family_and_branch


# Paths are relative to the project root, where the script is executed from.
INFERENCE_DIR = 'inference'
ASSETS_DIR = os.path.join(INFERENCE_DIR, 'assets')
VECTORIZER_PATH = os.path.join(ASSETS_DIR, 'vectorizer.pkl')
LIBRARY_AVERAGES_PATH = os.path.join(ASSETS_DIR, 'library_averages.pkl')
LLM_META_PATH = os.path.join('response_classifier', 'configs', 'llm_set.json')
CONFIDENCE_THRESHOLD = 0.40  # Placeholder until the optimal threshold is found


def load_inference_assets():
    """Loads the vectorizer, pre-computed library averages, and model metadata."""
    if not os.path.exists(VECTORIZER_PATH) or not os.path.exists(LIBRARY_AVERAGES_PATH):
        raise FileNotFoundError(
            "Required assets not found. Please run 'precompute_library.py' first and ensure "
            f"'vectorizer.pkl' is in the '{ASSETS_DIR}' directory."
        )
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(LIBRARY_AVERAGES_PATH, 'rb') as f:
        library_averages = pickle.load(f)

    llm_meta_map = load_llm_meta_data(LLM_META_PATH)

    return vectorizer, library_averages, llm_meta_map


def identify_llm(responses: list[str]) -> dict:
    """
    Identifies an LLM based on a list of its responses.

    :param responses: A list of strings, where each string is a response from the unknown LLM.
    :param threshold: The cosine similarity threshold for making a confident prediction.
    :return: A dictionary containing the prediction results.
    """
    vectorizer, library_averages, llm_meta_map = load_inference_assets()
    metric_func = get_metric_func('cosine_similarity')

    top_k_preds, all_scores = predict_with_scores(
        unknown_responses=responses,
        library_prompt_averages=library_averages,
        vectorizer=vectorizer,
        metric=metric_func,
        top_k=3
    )

    if not top_k_preds:
        return {"prediction": "unknown",
                "reason": "No models in library to compare against."}

    # Decision logic
    top_prediction, top_score = top_k_preds[0]
    if top_score >= CONFIDENCE_THRESHOLD:
        prediction_result = {
            "prediction": [(llm, round(score, 2)) for llm, score in top_k_preds],
            "all_scores": all_scores
        }
    else:
        prediction_result = {
            "prediction": ["unknown"],
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
