import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from typing import Callable
from sklearn.preprocessing import normalize
from scipy.sparse import vstack as sparse_vstack
import scipy.sparse
from tqdm import tqdm


def _as_ndarray(x) -> np.ndarray:
    """Convert numpy.matrix or sparse mean result to (1, dim) ndarray."""
    x = np.asarray(x)
    if x.ndim == 1:            # flatten → row-vector
        x = x.reshape(1, -1)
    return x


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
            if len(group) == 0:
                continue
            avg_vec = get_avg_vec(group, vector_col)
            averages[(model, prompt, bin_name)] = avg_vec

        # Overall average per (model, prompt) across all temps
        for prompt, group in df.groupby('prompt'):
            if len(group) == 0:
                continue
            avg_vec = get_avg_vec(group, vector_col)
            averages[(model, prompt, 'overall')] = avg_vec

    return averages


def get_avg_vec(group: pd.DataFrame, vector_col: str) -> np.ndarray:
    """
    Compute average vector for a group, handling both sparse and dense vectors.

    :param group: Pandas DataFrame group.
    :param vector_col: String name of the vector column.
    :return: Average vector as 2D NumPy array (1, dim).
    """
    vectors = group[vector_col].tolist()
    if vectors and isinstance(vectors[0], scipy.sparse.csr_matrix):
        stacked = sparse_vstack(vectors)
        mean_arr = np.asarray(stacked.mean(axis=0))  # ndarray
        avg_vec = mean_arr.reshape(1, -1)
    else:
        if vectors:
            stacked = np.array(vectors)
            avg_vec = np.mean(stacked, axis=0, keepdims=True)
        else:
            avg_vec = np.array([]).reshape(1, 0)  # Handle empty
    return avg_vec


def predict_unknown(unknown_data: dict[str, pd.DataFrame], library_avgs: dict,
                    metric: Callable, vector_col: str = 'response_vector', top_k: int = 3,
                    do_normalize: bool = False, return_scores: bool = False,
                    single_prompt_mode: bool = False) -> dict[str, dict]:
    """
    Predict top-k models for each (unknown_model, prompt) in unknown_data by averaging unknown vectors per prompt
    and comparing to library averages (per bin and overall for same prompt).

    :param unknown_data: Dict that maps unknown_llm to DataFrame with 'prompt' and 'response_vector'.
    :param library_avgs: The averages dict from `compute_library_averages()`.
    :param metric: A callable function (e.g., `cosine_similarity` or `euclidean_distances`).
    :param vector_col: String name of the vector column.
    :param top_k: Number of top predictions to return per (unknown_model, prompt).
    :param do_normalize: If True, apply L2 normalization to vectors before comparison.
    :param return_scores: If True, returns
        {unknown_llm: {combo: [(pred_llm, score), ...]}} instead of just lists of LLMs.
    :param single_prompt_mode: If True, assumes library_avgs keys are (model, bin_name)
        (no prompt), skips prompt filtering, and unpacks accordingly.
    :return: A nested dict structured `{unknown_llm: {group_key: [top_k predicted LLMs]}}`,
        where group_key is a tuple (prompt, system_prompt, technique) if those columns
        exist, else just the prompt str.
    """
    predictions = {}
    is_similarity = 'similarity' in metric.__name__

    # For each (unknown_llm, user prompt) pair
    for unknown_llm, df in tqdm(unknown_data.items(), desc="Making predictions for unknown LLMs", leave=False):
        predictions[unknown_llm] = {}
        group_cols = ['prompt']
        if 'system_prompt' in df.columns:
            group_cols.append('system_prompt')
        if 'neutralization_technique' in df.columns:
            group_cols.append('neutralization_technique')

        # Average unknown vectors per prompt
        for combo, group in df.groupby(group_cols):
            if len(group) == 0:
                continue

            # Use the existing helper function that handles both sparse and dense vectors
            avg_vec = get_avg_vec(group, vector_col)
            if do_normalize:
                avg_vec = normalize(avg_vec, norm='l2')

            # Collect best score per known llm (across its bins + overall for this prompt)
            known_llm_scores = {}
            # Extract prompt. If `combo` is tuple (full group), it's `combo[0]`.
            # Otherwise, it's the str itself
            prompt = combo[0] if isinstance(combo, tuple) else combo
            if single_prompt_mode:
                known_llms = set(m for (m, b) in library_avgs.keys())
            else:
                known_llms = set(m for (m, p, b) in library_avgs if p == prompt)

            for known_llm in known_llms:
                best_score = -np.inf if is_similarity else np.inf
                for bin_name in ['low', 'medium', 'high', 'overall']:
                    if single_prompt_mode:
                        key = (known_llm, bin_name)
                    else:
                        key = (known_llm, prompt, bin_name)

                    if key in library_avgs:
                        lib_vec = library_avgs[key]  # Sparse (1, dim)
                        lib_vec = _as_ndarray(lib_vec)
                        if do_normalize:
                            lib_vec = normalize(lib_vec, norm='l2')

                        score = metric(avg_vec, lib_vec)[0][0]
                        if not isinstance(score, (int, float)):
                            raise ValueError(f"Non-numeric score detected: {score} "
                                             f"(type: {type(score)})")

                        if is_similarity:
                            best_score = max(best_score, score)
                        else:
                            best_score = min(best_score, score)

                known_llm_scores[known_llm] = best_score

            # Sort models by best score (desc for sim, asc for dist)
            if known_llm_scores:
                sorted_pairs = sorted(known_llm_scores.items(), key=lambda item: item[1],
                                      reverse=is_similarity)
                if return_scores:
                    predictions[unknown_llm][combo] = sorted_pairs[:top_k]  # List of (llm, score)
                else:
                    predictions[unknown_llm][combo] = [pair[0] for pair in sorted_pairs[:top_k]]  # Just llms
            else:
                predictions[unknown_llm][combo] = []

    return predictions


def get_metric_func(metric_name: str) -> Callable:
    if metric_name == 'cosine_similarity':
        return cosine_similarity
    elif metric_name == 'euclidean_distances':
        return euclidean_distances
    else:
        raise ValueError(f"Unknown metric: {metric_name}")