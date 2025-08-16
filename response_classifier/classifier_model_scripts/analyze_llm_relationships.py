import os
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.preprocessing import normalize
from tabulate import tabulate
from data_processor import load_processed
from llm_meta_data import load_llm_meta_data, get_llm_family_and_branch
from classifier_model import get_avg_vec


BEST_PROMPT_PATH = '../configs/best_user_prompt.json'
METHOD_NAME = 'tfidf_trigram'
SPLIT_NAME = 'train'
PROCESSED_PATH = f'../data/processed/{METHOD_NAME}/{SPLIT_NAME}/'
LIBRARY_AVERAGES_PATH = f'../data/processed/{METHOD_NAME}/{SPLIT_NAME}/library_averages.pkl'
OUTPUT_DIR = '../results/library_data_analysis/'
os.makedirs(OUTPUT_DIR, exist_ok=True)
META_MAP = load_llm_meta_data('../configs/llm_set.json')
DO_NORMALIZE = True


def get_vector(library_averages: dict, key: tuple, do_normalize: bool,
               default_shape: tuple = None) -> np.ndarray:
    """
    Get a vector from library_averages for a key, optionally normalize, and reshape.
    If key not found and default_shape provided, return zeros of that shape.
    """
    if key in library_averages:
        vec = library_averages[key].reshape(1, -1)
    elif default_shape:
        vec = np.zeros((1, default_shape[1])).reshape(1, -1)
    else:
        raise ValueError(f"Key {key} not found and no default shape provided.")

    if do_normalize and np.linalg.norm(vec) > 0:
        vec = normalize(vec, norm='l2')

    return vec


def load_library_data() -> dict[str, pd.DataFrame]:
    """
    Load processed library data (train split) with response vectors.
    """
    return load_processed(METHOD_NAME, SPLIT_NAME)


def load_library_averages() -> dict[tuple[str, str], np.ndarray]:
    """
    Load precomputed library averages for the best prompt.
    Assumes keys are (model, bin_name).
    """
    if not os.path.exists(LIBRARY_AVERAGES_PATH):
        raise FileNotFoundError(f"Library averages not found at {LIBRARY_AVERAGES_PATH}. Run precompute_library.py first.")
    with open(LIBRARY_AVERAGES_PATH, 'rb') as f:
        return pickle.load(f)


def filter_for_best_prompt(data: dict[str, pd.DataFrame], best_prompt: str) -> dict[str, pd.DataFrame]:
    """
    Filter each LLM's DataFrame to only include the best prompt.
    """
    filtered = {}
    for llm, df in data.items():
        prompt_df = df[df['prompt'] == best_prompt].copy()
        if not prompt_df.empty:
            filtered[llm] = prompt_df
    return filtered


def save_df_to_text(df: pd.DataFrame, file_path: str, title: str = "") -> None:
    """
    Save a DataFrame to a nicely formatted text file using tabulate.
    """
    with open(file_path, 'w') as f:
        if title:
            f.write(f"{title}\n\n")
        f.write(tabulate(df, headers='keys', tablefmt='simple'))
        f.write("\n")


def save_matrix_to_text(matrix_df: pd.DataFrame, file_path: str, title: str = "") -> None:
    """
    Save a matrix DataFrame to a nicely formatted text file using tabulate.
    Includes index as the first column.
    """
    # Reset index to include it as a column
    matrix_df = matrix_df.reset_index().rename(columns={'index': ''})
    with open(file_path, 'w') as f:
        if title:
            f.write(f"{title}\n\n")
        f.write(tabulate(matrix_df, headers='keys', tablefmt='simple', showindex=False))
        f.write("\n")


def save_list_to_text(items: list, file_path: str, title: str = "", headers: list[str] = None) -> None:
    """
    Save a list of items (e.g., tuples) to a nicely formatted text file using tabulate.
    Handles both flat lists and lists of lists/tuples.
    """
    with open(file_path, 'w') as f:
        if title:
            f.write(f"{title}\n\n")
        if headers:
            f.write(tabulate(items, headers=headers, tablefmt='simple'))
        else:
            f.write(tabulate(items, tablefmt='simple'))
        f.write("\n")


def save_sorted_similarities(sim_df: pd.DataFrame, file_path: str, title: str = "") -> None:
    """
    Save sorted similarities for each LLM as sections in a text file.
    Each section lists other LLMs sorted by descending similarity.
    """
    with open(file_path, 'w') as f:
        if title:
            f.write(f"{title}\n\n")
        for llm in sim_df.index:
            # Get row, exclude self (similarity=1.0), sort descending
            row = sim_df.loc[llm].drop(llm)  # Drop self
            sorted_row = row.sort_values(ascending=False)
            sorted_pairs = list(zip(sorted_row.index, sorted_row.values))
            f.write(f"Sorted Similarities for {llm}:\n")
            f.write(tabulate(sorted_pairs, headers=['LLM', 'Similarity Score'], tablefmt='simple'))
            f.write("\n\n")


def analyze_inter_llm_similarity(library_averages: dict[tuple[str, str], np.ndarray]) -> None:
    """
    Computes M×M cosine similarity, saves text file, heatmap, and dendrogram.
    """
    llms = sorted(set(m for m, _ in library_averages.keys()))
    vec_list = [get_vector(library_averages, (llm, 'overall'), DO_NORMALIZE) for llm in llms]
    overall_vectors = np.vstack(vec_list)
    sim_matrix = cosine_similarity(overall_vectors)
    sim_df = pd.DataFrame(sim_matrix, index=llms, columns=llms)
    path_str = os.path.join(OUTPUT_DIR, 'inter_llm_similarity_matrix_overall.txt')
    title_str = "Inter-LLM Cosine Similarity (Overall, Best Prompt) - Sorted per LLM"
    save_sorted_similarities(sim_df, path_str, title=title_str)

    # Heatmap
    plt.figure(figsize=(22, 18))
    ax = sns.heatmap(sim_df, cmap='viridis', annot=False,
                     cbar_kws={'label': 'Cosine Similarity'})
    plt.title('Inter-LLM Cosine Similarity Heatmap (Overall, Best Prompt)', fontsize=24)
    plt.xticks(rotation=45, ha='right', fontsize=18)
    plt.yticks(rotation=0, fontsize=18)
    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel('Cosine Similarity', fontsize=20)  # Colorbar label size
    cbar.ax.tick_params(labelsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'inter_llm_heatmap_overall.png'))
    plt.close()

    # Dendrogram
    dist_matrix = 1 - sim_matrix
    np.fill_diagonal(dist_matrix, 0)
    condensed_dist = squareform(dist_matrix)
    Z = linkage(condensed_dist, method='average')
    fig, ax = plt.subplots(figsize=(22, 16))
    dendrogram(Z, labels=llms, leaf_rotation=45, leaf_font_size=18, orientation='top', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), ha='right')
    ax.set_title('Dendrogram of LLM Clustering (Overall, Best Prompt)', fontsize=26)
    ax.set_ylabel('Average Linkage Distance', fontsize=22)
    ax.tick_params(axis='y', labelsize=18)
    # Increase branch thickness
    linecollections = ax.collections  # Get all line collections (branches)
    for lc in linecollections:
        lc.set_linewidth(3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'inter_llm_dendrogram_overall.png'))
    plt.close()


def analyze_llm_response_consistency(library_data: dict[str, pd.DataFrame]) -> None:
    """
    Computes per-model metrics: mean/std cosine to centroid (overall and per
    temp bin), and cosine similarities between bin centroids and overall centroids.
    """
    dispersion_data = []
    for llm, df in library_data.items():
        metrics = {'llm': llm}

        # Compute centroids
        centroids = {}
        for bin_name, group in df.groupby('temp_bin'):
            centroids[bin_name] = get_avg_vec(group, 'response_vector')
        centroids['overall'] = get_avg_vec(df, 'response_vector')

        # Inter-centroid similarities
        for bin_name in ['low', 'medium', 'high']:
            if bin_name in centroids:
                cos_sim = cosine_similarity(normalize(centroids[bin_name], norm='l2'),
                                            normalize(centroids['overall'], norm='l2'))[0][0]
                metrics[f'cosine_{bin_name}_overall'] = cos_sim

        # Mean/std cosine to centroid (requires individual vectors)
        for bin_name in ['low', 'medium', 'high', 'overall']:
            if bin_name == 'overall':
                group = df
            else:
                group = df[df['temp_bin'] == bin_name]
            if not group.empty:
                centroid = centroids[bin_name]
                vectors = np.vstack([v.toarray() if hasattr(v, 'toarray') else v for v in group['response_vector']])
                vectors_norm = normalize(vectors, norm='l2')
                centroid_norm = normalize(centroid, norm='l2')
                cosines = cosine_similarity(vectors_norm, centroid_norm).flatten()
                metrics[f'mean_cosine_to_centroid_{bin_name}'] = np.mean(cosines)
                metrics[f'std_cosine_to_centroid_{bin_name}'] = np.std(cosines)

        dispersion_data.append(metrics)

    dispersion_df = pd.DataFrame(dispersion_data)
    save_df_to_text(dispersion_df, os.path.join(OUTPUT_DIR, 'intra_llm_dispersion_metrics.txt'), title="Intra-LLM Dispersion Metrics")


def analyze_family_separation(library_averages: dict[tuple[str, str], np.ndarray], meta_map: dict) -> None:
    """
    Analyze the difference of between-family and within-family cosine similarity.
    Uses overall vectors to compute mean similarity to same/other families.
    """
    llms = sorted(set(m for m, _ in library_averages.keys()))
    vec_list = [get_vector(library_averages, (llm, 'overall'), DO_NORMALIZE) for llm in llms]
    overall_vectors = np.vstack(vec_list)
    sim_matrix = cosine_similarity(overall_vectors)

    family_sim_data = []
    for i, llm in enumerate(llms):
        family, _ = get_llm_family_and_branch(llm, meta_map)
        same_family_indices = [j for j in range(len(llms)) if j != i and get_llm_family_and_branch(llms[j], meta_map)[0] == family]
        other_family_indices = [j for j in range(len(llms)) if j != i and j not in same_family_indices]

        same_mean = np.mean(sim_matrix[i, same_family_indices]) if same_family_indices else np.nan
        other_mean = np.mean(sim_matrix[i, other_family_indices]) if other_family_indices else np.nan
        mean_score_difference = same_mean - other_mean if not (np.isnan(same_mean) or np.isnan(other_mean)) else np.nan

        family_sim_data.append({
            'llm': llm,
            'family': family,
            'mean_sim_same_family': same_mean,
            'mean_sim_other_family': other_mean,
            'score_difference': mean_score_difference
        })

    family_sim_df = pd.DataFrame(family_sim_data)
    save_df_to_text(family_sim_df, os.path.join(OUTPUT_DIR, 'family_separation_metrics.txt'),
                    title="Family Separation Metrics")


def analyze_inter_llm_similarity_temp(library_averages: dict[tuple[str, str], np.ndarray]) -> None:
    """
    Analyze the impact of LLM temperature on inter-LLM Similarity.
    Computes similarity matrices per temp bin and differences from overall.
    """
    llms = sorted(set(m for m, _ in library_averages.keys()))
    temp_bins = ['low', 'medium', 'high']
    vec_list = [get_vector(library_averages, (llm, 'overall'), DO_NORMALIZE) for llm in llms]
    overall_vectors = np.vstack(vec_list)
    overall_sim = cosine_similarity(overall_vectors)

    matrix_diffs = []
    for bin_name in temp_bins:
        vec_list = [get_vector(library_averages, (llm, bin_name), DO_NORMALIZE, default_shape=overall_vectors.shape)
                    for llm in llms]
        bin_vectors = np.vstack(vec_list)
        bin_sim = cosine_similarity(bin_vectors)
        bin_df = pd.DataFrame(bin_sim, index=llms, columns=llms)

        # Save sorted similarities instead of full matrix
        path_str = os.path.join(OUTPUT_DIR, f'inter_llm_similarity_matrix_{bin_name}.txt')
        title_str = f"Inter-LLM Cosine Similarity ({bin_name.capitalize()}, Best Prompt) - Sorted per LLM"
        save_sorted_similarities(bin_df, path_str, title=title_str)

        # Heatmap
        plt.figure(figsize=(22, 18))
        ax = sns.heatmap(bin_df, cmap='viridis', annot=False, cbar_kws={'label': 'Cosine Similarity'})
        plt.title(f'Inter-LLM Cosine Similarity Heatmap ({bin_name.capitalize()}, Best Prompt)', fontsize=24)
        plt.xticks(rotation=45, ha='right', fontsize=18)
        plt.yticks(rotation=0, fontsize=18)
        cbar = ax.collections[0].colorbar
        cbar.ax.set_ylabel('Cosine Similarity', fontsize=20)
        cbar.ax.tick_params(labelsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'inter_llm_heatmap_{bin_name}.png'))
        plt.close()

        # Mean difference from overall
        diff = np.abs(bin_sim - overall_sim).mean()
        matrix_diffs.append({'temp_bin': bin_name, 'mean_diff_from_overall': diff})

    diff_df = pd.DataFrame(matrix_diffs)
    save_df_to_text(diff_df, os.path.join(OUTPUT_DIR, 'temp_impact_metrics.txt'),
                    title="Temperature Impact Metrics")


def main():
    # Load best prompt (file is a list, take the first element)
    with open(BEST_PROMPT_PATH, 'r') as f:
        best_prompt_data = json.load(f)
        best_prompt = best_prompt_data[0]  # It's a list with one string

    # Load data and averages
    print("Loading data and library averages")
    library_data = load_library_data()
    library_data = filter_for_best_prompt(library_data, best_prompt)
    library_averages = load_library_averages()

    # Run analyses
    print("Analysis 1/4: analyzing inter-LLM similarity")
    analyze_inter_llm_similarity(library_averages)
    print("Analysis 2/4: analyzing LLM response consistency")
    analyze_llm_response_consistency(library_data)
    print("Analysis 3/4: analyzing LLM family separation")
    analyze_family_separation(library_averages, META_MAP)
    print("Analysis 4/4: analyzing inter-LLM similarity for temperature bins")
    analyze_inter_llm_similarity_temp(library_averages)

    print(f"All analyses complete. Results saved to {OUTPUT_DIR}")

if __name__ == '__main__':
    main()