import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, \
    confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Callable
from classifier_model import compute_library_averages, predict_unknown, get_metric_func
from llm_meta_data import get_llm_family_and_branch
from tqdm import tqdm
from analyze_final_clf_dataset import detect_refusal


def compute_metrics(predictions: dict[str, dict]) -> dict:
    """
    Compute metrics: top1/top3 acc, f1, etc. + confusion matrix.
    Assumes predictions are nested: {llm: {group_key: top-k list}}, where
    group_key is str or tuple. Flattens to per unique group for metrics.
    """
    # For each (llm, group_key) pairs (group_key could be str or tuple)
    true_labels = []
    top1_pred_labels = []
    top3_pred_lists = []
    for llm, group_preds in predictions.items():
        for group_key, pred_list in group_preds.items():
            true_labels.append(llm)
            if pred_list:
                top1_pred_labels.append(pred_list[0][0])
                top3_pred_lists.append([p for p, _ in pred_list])
            else:
                top1_pred_labels.append('unknown')
                top3_pred_lists.append([])

    top1_acc = accuracy_score(true_labels, top1_pred_labels)
    top3_acc = np.mean([true in top3 for true, top3 in zip(true_labels, top3_pred_lists)])
    f1 = f1_score(true_labels, top1_pred_labels, average='macro', zero_division=0)
    precision = precision_score(true_labels, top1_pred_labels, average='macro', zero_division=0)
    recall = recall_score(true_labels, top1_pred_labels, average='macro', zero_division=0)

    all_labels = sorted(set(true_labels) | set(top1_pred_labels))
    cm = confusion_matrix(true_labels, top1_pred_labels, labels=all_labels)
    return {
        'top1_accuracy': top1_acc,
        'top3_accuracy': top3_acc,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm.tolist()
    }


def compute_held_out_llm_metrics(predictions: dict, meta_map: dict, threshold: float) -> dict:
    """
    Computes top-1 and top-3 accuracy at the model family and branch levels.
    """
    if not meta_map:
        return {"error": "Model metadata not available."}

    family_matches_top1, branch_matches_top1 = 0, 0
    family_matches_top3, branch_matches_top3 = 0, 0
    total_preds = 0
    ood_count = 0

    for true_llm, prompt_preds in predictions.items():
        for pred_list in prompt_preds.values():
            if not pred_list:
                continue

            total_preds += 1
            true_family, true_branch = get_llm_family_and_branch(true_llm, meta_map)

            # Top-1 LLM family and branch check
            pred_family_top1, pred_branch_top1 = get_llm_family_and_branch(pred_list[0][0], meta_map)
            if true_family == pred_family_top1:
                family_matches_top1 += 1
            if true_branch == pred_branch_top1:
                branch_matches_top1 += 1

            # Top-3 LLM family and branch check
            top3_families = {get_llm_family_and_branch(p, meta_map)[0] for p, _ in pred_list}
            top3_branches = {get_llm_family_and_branch(p, meta_map)[1] for p, _ in pred_list}
            if true_family in top3_families:
                family_matches_top3 += 1
            if true_branch in top3_branches:
                branch_matches_top3 += 1

            max_score = max(s for _, s in pred_list) if pred_list else 0
            if max_score < threshold:
                ood_count += 1

    if total_preds == 0: return {}

    ood_accuracy = ood_count / total_preds

    return {
        'top1_family_accuracy': family_matches_top1 / total_preds,
        'top3_family_accuracy': family_matches_top3 / total_preds,
        'top1_branch_accuracy': branch_matches_top1 / total_preds,
        'top3_branch_accuracy': branch_matches_top3 / total_preds,
        'total_predictions': total_preds,
        'ood_detection_accuracy': ood_accuracy,
        'ood_false_positive_rate': 1 - ood_accuracy
    }


def save_confusion_matrix(cm: np.ndarray, labels: list[str], output_path: str) -> None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, xticklabels=labels, yticklabels=labels)
    plt.savefig(output_path)
    plt.close()


def apply_threshold(predictions_with_scores: dict[str, dict], threshold: float,
                    top_k: int = 3) -> dict[str, dict]:
    """
    Apply confidence threshold to predictions with scores. Returns thresholded
    predictions (lists of LLMs or "unknown").
    """
    thresholded_preds = {}
    for unknown_llm, combo_preds in predictions_with_scores.items():
        thresholded_preds[unknown_llm] = {}
        for combo, score_pairs in combo_preds.items():
            if not score_pairs:
                thresholded_preds[unknown_llm][combo] = []
                continue

            max_score = score_pairs[0][1]  # Top-1 score (assuming sorted)
            if max_score < threshold:
                thresholded_preds[unknown_llm][combo] = ["unknown"]
            else:
                thresholded_preds[unknown_llm][combo] = [pair[0] for pair in score_pairs[:top_k]]

    return thresholded_preds


def compute_metrics_with_refusals(predictions: dict[str, dict],
                                  unknown_data: dict[str, pd.DataFrame], meta_map: dict,
                                  is_held_out: bool = False, threshold: float = None) -> dict:
    """
    Compute metrics overall, for refusals, and for non-refusals.

    :param predictions: Thresholded predictions.
    :param unknown_data: Original data dict for refusal detection.
    :param meta_map: For family/branch metrics if held-out.
    :param is_held_out: If True, compute OOD metrics; else in-distribution metrics.
    :param threshold: Confidence threshold for OOD detection (required if is_held_out).
    """
    if is_held_out and threshold is None:
        raise ValueError("Threshold is required for held-out metrics.")

    # Flatten predictions and add refusal tags
    flat_rows = []
    for llm, combo_preds in predictions.items():
        for combo, pred_list in combo_preds.items():
            if isinstance(combo, tuple):
                if len(combo) == 2:
                    prompt, sys_prompt = combo
                    technique = None
                elif len(combo) == 3:
                    prompt, sys_prompt, technique = combo
                else:
                    raise ValueError(f"Unexpected combo length: {len(combo)} for {combo}")
            else:
                prompt = combo
                sys_prompt = None
                technique = None

            filter_cond = ((unknown_data[llm]['prompt'] == prompt) &
                           (unknown_data[llm]['system_prompt'] == sys_prompt))
            if technique is not None:
                filter_cond &= (unknown_data[llm]['neutralization_technique'] == technique)
                
            group = unknown_data[llm][filter_cond]
            responses = group['response'].tolist()
            is_refusal_list = [detect_refusal(r) for r in responses]
            # Average per combo (majority vote for refusal)
            is_refusal = (sum(is_refusal_list) / len(is_refusal_list)) > 0.5
            flat_rows.append({
                'true_llm': llm,
                'combo': combo,
                'pred_list': pred_list,  # list of (llm, score)
                'is_refusal': is_refusal
            })

    flat_df = pd.DataFrame(flat_rows)

    # Helper to compute subset metrics
    def compute_subset(df_subset, threshold):
        subset_preds = {row['true_llm']: {row['combo']: row['pred_list']} for _, row in df_subset.iterrows()}
        if is_held_out:
            metrics = compute_held_out_llm_metrics(subset_preds, meta_map, threshold)
        else:
            metrics = compute_metrics(subset_preds)
        return metrics

    overall_metrics = compute_subset(flat_df, threshold)
    refusal_metrics = compute_subset(flat_df[flat_df['is_refusal']], threshold)
    non_refusal_metrics = compute_subset(flat_df[~flat_df['is_refusal']], threshold)

    return {
        'overall': overall_metrics,
        'refusal': refusal_metrics,
        'non_refusal': non_refusal_metrics
    }


def tune_threshold(non_held_out_tuning_set: dict, held_out_tuning_set: dict,
                   library_avgs: dict, metric_func: Callable, meta_map: dict,
                   thresholds: list[float]) -> dict:
    """
    Evaluate thresholds on tuning sets and return the metric values.
    """
    results = {}
    for thresh in tqdm(thresholds, desc="Evaluating thresholds on tuning sets"):
        # Predict with scores
        non_held_preds = predict_unknown(non_held_out_tuning_set, library_avgs,
                                         metric_func, return_scores=True, single_prompt_mode=True)
        held_preds = predict_unknown(held_out_tuning_set, library_avgs, metric_func,
                                     return_scores=True, single_prompt_mode=True)

        # Compute metrics for all, refusal, and non-refusal data points
        non_held_metrics = compute_metrics_with_refusals(non_held_preds,
                                                         non_held_out_tuning_set,
                                                         meta_map, is_held_out=False,
                                                         threshold=thresh)
        held_metrics = compute_metrics_with_refusals(held_preds, held_out_tuning_set,
                                                     meta_map, is_held_out=True,
                                                     threshold=thresh)

        results[thresh] = {
            'non_held_out': non_held_metrics,
            'held_out': held_metrics
        }

    return results