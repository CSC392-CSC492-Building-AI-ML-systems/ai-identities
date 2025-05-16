import json
import sys
import os
import numpy as np
from scipy.spatial.distance import cosine, cityblock

# Ensure the script can import from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_identity import EnhancedModelManager

def compute_jaccard_similarity(set1, set2):
    """
    Compute Jaccard similarity between two sets of words
    
    Args:
        set1 (set): First set of words
        set2 (set): Second set of words
    
    Returns:
        float: Jaccard similarity score
    """
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

def compute_model_drift(manager, word_freq1, word_freq2):
    """
    Compute drift between two word frequency distributions using 
    the advanced fingerprinting method from EnhancedModelManager
    
    Args:
        manager (EnhancedModelManager): Initialized model manager
        word_freq1 (dict): First word frequency distribution
        word_freq2 (dict): Second word frequency distribution
    
    Returns:
        dict: Detailed drift metrics
    """
    # Create advanced fingerprints for both distributions
    fingerprint1 = manager._create_advanced_fingerprint(word_freq1)
    fingerprint2 = manager._create_advanced_fingerprint(word_freq2)
    
    # Cosine similarity on normalized frequencies
    normalized_freq1 = list(fingerprint1['normalized_freq'].values())[:15]
    normalized_freq2 = list(fingerprint2['normalized_freq'].values())[:15]
    cosine_dist = 1 - cosine(normalized_freq1, normalized_freq2)
    
    # Manhattan distance on rarity scores
    rarity_scores1 = list(fingerprint1['rarity_scores'].values())[:15]
    rarity_scores2 = list(fingerprint2['rarity_scores'].values())[:15]
    manhattan_dist = 1 - cityblock(rarity_scores1, rarity_scores2)
    
    # Jaccard similarity on distinctive words
    base_distinct_words = set(fingerprint1['distinctive_words'].keys())
    current_distinct_words = set(fingerprint2['distinctive_words'].keys())
    jaccard_dist = compute_jaccard_similarity(base_distinct_words, current_distinct_words)
    
    # Zipf's law fit comparison
    zipf_drift = abs(
        fingerprint1['zipf_fit']['slope'] - 
        fingerprint2['zipf_fit']['slope']
    ) if fingerprint1['zipf_fit'] and fingerprint2['zipf_fit'] else 0
    
    # Weighted drift score
    drift_score = (
        0.4 * cosine_dist + 
        0.3 * manhattan_dist + 
        0.2 * jaccard_dist + 
        0.1 * (1 - zipf_drift)
    )
    
    return {
        'cosine_similarity': cosine_dist,
        'manhattan_distance': manhattan_dist,
        'jaccard_similarity': jaccard_dist,
        'zipf_drift': zipf_drift,
        'combined_drift_score': drift_score
    }

def main():
    # Initialize the EnhancedModelManager
    manager = EnhancedModelManager()
    
    # Paths to the JSON files
    test_files = [
        'classifiers/testing_IDS/Llama-3.2-90B-Vision-Instruct_results_1.0.json',
        'classifiers/testing_IDS/Llama-3.2-90B-Vision-Instruct_results_0.5.json',
        'classifiers/testing_IDS/Meta-Llama-3.1-70B-Instruct-Turbo_results_0.5.json',
        'classifiers/testing_IDS/Meta-Llama-3.1-70B-Instruct-Turbo_results_1.0.json'
    ]
    
    # Load data from files
    word_freqs = []
    for file_path in test_files:
        with open(file_path, 'r') as f:
            word_freqs.append(json.load(f))
    
    # Compute drift scores between different model runs
    print("Drift Analysis Results:")
    print("-" * 50)
    
    # Comparison matrix (comparing all pairs)
    for i in range(len(word_freqs)):
        for j in range(i+1, len(word_freqs)):
            drift_result = compute_model_drift(manager, word_freqs[i], word_freqs[j])
            
            print(f"Drift between file {test_files[i]} and {test_files[j]}:")
            print(f"Cosine Similarity: {drift_result['cosine_similarity']:.4f}")
            print(f"Manhattan Distance: {drift_result['manhattan_distance']:.4f}")
            print(f"Jaccard Similarity: {drift_result['jaccard_similarity']:.4f}")
            print(f"Zipf Drift: {drift_result['zipf_drift']:.4f}")
            print(f"Combined Drift Score: {drift_result['combined_drift_score']:.4f}")
            print("-" * 50)

if __name__ == "__main__":
    main()