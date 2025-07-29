import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from typing import Callable
from sklearn.preprocessing import normalize


def compute_library_averages(library: dict[str, pd.DataFrame],
                             vector_col: str = 'response_vector') -> dict:
    """
    Compute average vector per (model, prompt, temp_bin) and per (model, prompt, 'overall') in library.

    :param library: A dictionary that maps LLM names  the values to the
        corresponding Pandas DataFrame containing the response data.
    :param vector_col: String name of the vector column.
    :return: A dictionary where keys are the tuples (model, prompt, bin_name)
        and values are NumPy arrays that are the average vector for that group.
        bin_name can be 'low', 'medium', 'high', or 'overall'.
    """
    averages = {}
    for model, df in library.items():
        # Per-bin averages
        for (prompt, bin_name), group in df.groupby(['prompt', 'temp_bin']):
            avg_vec = np.mean(np.array(group[vector_col].tolist()), axis=0)
            averages[(model, prompt, bin_name)] = avg_vec

        # Overall average per (model, prompt) across all temps
        for prompt, group in df.groupby('prompt'):
            avg_vec = np.mean(np.array(group[vector_col].tolist()), axis=0)
            averages[(model, prompt, 'overall')] = avg_vec

    return averages


def predict_unknown(unknown_data: dict[str, pd.DataFrame], library_avgs: dict,
                    metric: Callable, vector_col: str = 'response_vector', top_k: int = 3,
                    do_normalize: bool = False) -> dict[str, dict[str, list[str]]]:
    """
    Predict top-k models for each (unknown_model, prompt) in unknown_data by averaging unknown vectors per prompt
    and comparing to library averages (per bin and overall for same prompt).

    :param unknown_data: Dict that maps unknown_llm to DataFrame with 'prompt' and 'response_vector'.
    :param library_avgs: The averages dict from `compute_library_averages()`.
    :param metric: A callable function (e.g., `cosine_similarity` or `euclidean_distances`).
    :param vector_col: String name of the vector column.
    :param top_k: Number of top predictions to return per (unknown_model, prompt).
    :param do_normalize: If True, apply L2 normalization to vectors before comparison.
    :return: A nested dict structured `{unknown_llm: {prompt: [top_k predicted LLMs]}}`.
    """
    predictions = {}
    is_similarity = 'similarity' in metric.__name__

    # For each (unknown_llm, user prompt) pair
    for unknown_llm, df in unknown_data.items():
        predictions[unknown_llm] = {}
        # Average unknown vectors per prompt
        for prompt, group in df.groupby('prompt'):
            if len(group) == 0:
                continue
            avg_vec = np.mean(np.array(group[vector_col].tolist()), axis=0)
            if do_normalize:
                avg_vec = normalize(avg_vec.reshape(1, -1), norm='l2')[0]

            # Collect best score per known llm (across its bins + overall for this prompt)
            known_llm_scores = {}
            known_llms = set(m for (m, p, b) in library_avgs if p == prompt)
            for known_llm in known_llms:
                best_score = -np.inf if is_similarity else np.inf
                for bin_name in ['low', 'medium', 'high', 'overall']:
                    key = (known_llm, prompt, bin_name)
                    if key in library_avgs:
                        lib_vec = library_avgs[key]
                        if do_normalize:
                            lib_vec = normalize(lib_vec.reshape(1, -1), norm='l2')[0]

                        score = metric(avg_vec.reshape(1, -1), lib_vec.reshape(1, -1))[0][0]
                        if is_similarity:
                            best_score = max(best_score, score)
                        else:
                            best_score = min(best_score, score)

                if known_llm not in known_llm_scores:
                    known_llm_scores[known_llm] = best_score

            # Sort models by best score (desc for sim, asc for dist)
            if known_llm_scores:
                sorted_llms = sorted(known_llm_scores, key=known_llm_scores.get,
                                       reverse=is_similarity)
                predictions[unknown_llm][prompt] = sorted_llms[:top_k]
            else:
                predictions[unknown_llm][prompt] = []

    return predictions


def get_metric_func(metric_name: str) -> Callable:
    if metric_name == 'cosine_similarity':
        return cosine_similarity
    elif metric_name == 'euclidean_distances':
        return euclidean_distances
    else:
        raise ValueError(f"Unknown metric: {metric_name}")