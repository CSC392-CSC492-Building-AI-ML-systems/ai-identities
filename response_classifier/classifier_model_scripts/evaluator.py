import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, \
    confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Callable
from classifier_model import predict_unknown
from llm_meta_data import get_llm_family_and_branch
from tqdm import tqdm
from analyze_final_clf_dataset_refusal import detect_refusal


def save_confusion_matrix_xy(cm: np.ndarray, ylabels: list[str], xlabels: list[str],
                             output_path: str, title: str | None = None,
                             xtick_rotation: int = 0) -> None:
    plt.figure(figsize=(max(10, 0.5*len(xlabels)), max(8, 0.45*len(ylabels))))
    ax = sns.heatmap(cm, annot=True, fmt='d', xticklabels=xlabels, yticklabels=ylabels,
                     cmap='Blues')
    if title:
        plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.setp(ax.get_xticklabels(), rotation=xtick_rotation, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def compute_metrics(predictions_with_scores: dict[str, dict], threshold: float) -> dict:
    """
    Compute metrics: top1/top3 acc, f1, etc. + confusion matrix.
    Assumes predictions are nested: {llm: {group_key: top-k list}}, where
    group_key is str or tuple. Flattens to per unique group for metrics.

    Decision-level (ALL samples):
      - top1_accuracy: accuracy over all examples, counting 'unknown' as incorrect.
      - top3_accuracy: over all examples; if decision is 'unknown', no top-3 is credited.

    Identified-only:
      - top1_accuracy_identified, top3_accuracy_identified, f1_score, precision, recall
        computed only on examples where score >= threshold.

    Rates:
      - unknown_rate: fraction of examples predicted as 'unknown' (abstentions).
      - misidentification_rate: fraction of ALL examples where a wrong non-'unknown' label was predicted.
      - identified_error_rate: fraction of identified examples that were wrong (conditional error).
    """
    # For each (llm, group_key) pairs (group_key could be str or tuple)
    true_labels, top1_pred_labels, top3_pred_lists = [], [], []
    unknown_count = 0
    total_count = 0
    for llm, group_preds in predictions_with_scores.items():
        for pred_list_with_scores in group_preds.values():
            total_count += 1
            true_labels.append(llm)
            if pred_list_with_scores and pred_list_with_scores[0][1] >= threshold:
                # Confident prediction
                top1_pred_labels.append(pred_list_with_scores[0][0])
                top3_pred_lists.append([p for p, _ in pred_list_with_scores])
            else:
                # Not confident, predict 'unknown'
                top1_pred_labels.append('unknown')
                top3_pred_lists.append([])  # No top-3 prediction
                unknown_count += 1

    if total_count == 0:
        return {
            # ALL-sample decision-level accuracies:
            'top1_accuracy_all': 0,
            'top3_accuracy_all': 0,

            # Identified-only quality metrics:
            'top1_accuracy_identified': 0,
            'top3_accuracy_identified': 0,
            'f1_score_identified': 0,
            'precision_identified': 0,
            'recall_identified': 0,

            # Rates:
            'unknown_rate': 0,
            'misidentification_rate': 0,
            'identified_error_rate': 0,

            # Counts and CM:
            'total_predictions': 0,
            'total_identified': 0,
            'confusion_matrix_identified': [],  # identified-only CM
            'confusion_matrix_all': [],  # ALL-sample CM (includes 'unknown')
            'confusion_matrix_all_labels': []
        }

    # ALL-sample accuracies (decision-level)
    top1_acc_all = accuracy_score(true_labels, top1_pred_labels)
    top3_acc_all = float(np.mean(
        [int(true_llm in top3) for true_llm, top3 in zip(true_labels, top3_pred_lists)]))

    labels_all = sorted(set(true_labels) | set(top1_pred_labels))  # includes 'unknown'
    cm_all = confusion_matrix(true_labels, top1_pred_labels, labels=labels_all).tolist()

    # Identified-only metrics
    known_indices = [i for i, label in enumerate(top1_pred_labels) if label != 'unknown']
    total_identified = len(known_indices)

    # Misidentification counts
    misidentified_count = sum(
        1 for i in range(total_count)
        if top1_pred_labels[i] != 'unknown' and top1_pred_labels[i] != true_labels[i]
    )
    misidentification_rate = misidentified_count / total_count
    identified_error_rate = (misidentified_count / total_identified) if total_identified > 0 else 0.0

    if total_identified > 0:
        true_labels_known = [true_labels[i] for i in known_indices]
        top1_preds_known = [top1_pred_labels[i] for i in known_indices]
        top3_preds_known = [top3_pred_lists[i] for i in known_indices]

        top1_acc_id = accuracy_score(true_labels_known, top1_preds_known)
        top3_acc_id = float(np.mean(
            [int(t in t3) for t, t3 in zip(true_labels_known, top3_preds_known)]))
        f1 = f1_score(true_labels_known, top1_preds_known, average='macro',
                      zero_division=0)
        precision = precision_score(true_labels_known, top1_preds_known,
                                    average='macro', zero_division=0)
        recall = recall_score(true_labels_known, top1_preds_known, average='macro',
                              zero_division=0)

        all_labels = sorted(set(true_labels_known) | set(top1_preds_known))
        cm_identified = confusion_matrix(true_labels_known, top1_preds_known,
                              labels=all_labels).tolist()
    else:
        top1_acc_id = 0.0
        top3_acc_id = 0.0
        f1 = 0.0
        precision = 0.0
        recall = 0.0
        cm_identified = []

    return {
        # ALL-sample decision-level accuracies:
        'top1_accuracy_all': top1_acc_all,
        'top3_accuracy_all': top3_acc_all,

        # Identified-only quality metrics:
        'top1_accuracy_identified': top1_acc_id,
        'top3_accuracy_identified': top3_acc_id,
        'f1_score_identified': f1,
        'precision_identified': precision,
        'recall_identified': recall,

        # Rates:
        'unknown_rate': unknown_count / total_count,
        'misidentification_rate': misidentification_rate,
        'identified_error_rate': identified_error_rate,

        # Counts and CM:
        'total_predictions': total_count,
        'total_identified': total_identified,
        'confusion_matrix_identified': cm_identified,  # identified-only
        'confusion_matrix_all': cm_all,  # ALL-sample
        'confusion_matrix_all_labels': labels_all
    }


def compute_per_model_coverage_accuracy(predictions_with_scores: dict[str, dict],
                                        threshold: float) -> dict[str, dict]:
    """
    Returns {llm: {
        'n': int,
        'coverage': float,
        'top1_accuracy_all': float,                # all-sample
        'top1_accuracy_identified': float,     # identified-only
        'top3_accuracy_all': float,            # all-sample
        'top3_accuracy_identified': float      # identified-only
    }}
    coverage = fraction of examples with score >= threshold
    """
    out = {}
    for true_llm, combo_preds in predictions_with_scores.items():
        n = len(combo_preds)
        identified = 0
        correct_all = 0
        correct_id = 0
        top3_hits_all = 0
        top3_hits_id = 0

        for score_pairs in combo_preds.values():
            if score_pairs and score_pairs[0][1] >= threshold:
                identified += 1
                pred1 = score_pairs[0][0]
                if pred1 == true_llm:
                    correct_all += 1
                    correct_id += 1
                top3_models = [p for p, _ in score_pairs[:3]]
                if true_llm in top3_models:
                    top3_hits_all += 1
                    top3_hits_id += 1
            else:
                # un-identified: contributes 0 to all-sample accuracies
                pass

        out[true_llm] = {
            'n': n,
            'coverage': (identified / n) if n else 0.0,
            'top1_accuracy_all': (correct_all / n) if n else 0.0,
            'top1_accuracy_identified': (correct_id / identified) if identified else 0.0,
            'top3_accuracy_all': (top3_hits_all / n) if n else 0.0,
            'top3_accuracy_identified': (top3_hits_id / identified) if identified else 0.0
        }

    return out


def build_held_out_confusion_matrices(predictions_with_scores: dict[str, dict],
                                      threshold: float, meta_map: dict,
                                      library_llms: list[str] | None = None) -> dict:
    """
    Returns a dict with three confusion matrices. Rows are true LLMs; columns vary:
        - 'model': predicted known LLMs + 'unknown'
        - 'family': predicted families + 'unknown'
        - 'branch': predicted branches + 'unknown'
    Each entry has: {'cm': np.ndarray, 'ylabels': [...], 'xlabels': [...]}
    """
    y_true = []
    y_pred_model = []
    y_pred_family = []
    y_pred_branch = []

    for true_llm, combo_preds in predictions_with_scores.items():
        for score_pairs in combo_preds.values():
            y_true.append(true_llm)
            if score_pairs and score_pairs[0][1] >= threshold:
                pm = score_pairs[0][0]
            else:
                pm = 'unknown'

            y_pred_model.append(pm)
            family, branch = get_llm_family_and_branch(score_pairs[0][0], meta_map)
            y_pred_family.append(family)
            y_pred_branch.append(branch)

    row_labels = sorted(set(y_true))

    # Column label sets
    if library_llms:
        model_cols = sorted(set(library_llms)) + ['unknown']
        fam_cols = sorted({get_llm_family_and_branch(m, meta_map)[0] for m in library_llms} - {None, 'unknown'}) + ['unknown']
        br_cols = sorted({get_llm_family_and_branch(m, meta_map)[1] for m in library_llms} - {None, 'unknown'}) + ['unknown']
    else:
        model_cols = sorted(set(y_pred_model)) + (['unknown'] if 'unknown' not in set(y_pred_model) else [])
        fam_cols = sorted(set(y_pred_family)) + (['unknown'] if 'unknown' not in set(y_pred_family) else [])
        br_cols = sorted(set(y_pred_branch)) + (['unknown'] if 'unknown' not in set(y_pred_branch) else [])

    cm_model = create_rectangular_cm(row_labels, model_cols, y_true, y_pred_model)
    cm_family = create_rectangular_cm(row_labels, fam_cols, y_true, y_pred_family)
    cm_branch = create_rectangular_cm(row_labels, br_cols, y_true, y_pred_branch)

    return {
        'model': {'cm': cm_model, 'ylabels': row_labels, 'xlabels': model_cols},
        'family': {'cm': cm_family, 'ylabels': row_labels, 'xlabels': fam_cols},
        'branch': {'cm': cm_branch, 'ylabels': row_labels, 'xlabels': br_cols}
    }


def build_non_held_out_confusion_matrices(predictions_with_scores: dict[str, dict],
                                          threshold: float, meta_map: dict,
                                          library_llms: list[str] | None = None) -> dict:
    """
    Returns a dict with three confusion matrices. Rows are true LLMs; columns vary:
        - 'model': predicted known LLMs (doesn't include "unknown")
        - 'family': predicted families (doesn't include "unknown")
        - 'branch': predicted branches (doesn't include "unknown")
    Each entry has: {'cm': np.ndarray, 'ylabels': [...], 'xlabels': [...]}
    """
    y_true = []
    y_pred_model = []
    y_pred_family = []
    y_pred_branch = []

    for true_llm, combos in predictions_with_scores.items():
        for score_pairs in combos.values():
            if score_pairs and score_pairs[0][1] >= threshold:
                y_true.append(true_llm)
                pm = score_pairs[0][0]
                y_pred_model.append(pm)
                fam, br = get_llm_family_and_branch(pm, meta_map)
                y_pred_family.append(fam if fam else 'unknown')
                y_pred_branch.append(br if br else 'unknown')
            else:
                # skip un-identified rows entirely
                continue

    row_labels = sorted(set(y_true))

    # Column labels
    if library_llms:
        model_cols = sorted(set(library_llms))
        fam_cols = sorted({get_llm_family_and_branch(m, meta_map)[0] for m in library_llms} - {None, 'unknown'})
        br_cols = sorted({get_llm_family_and_branch(m, meta_map)[1] for m in library_llms} - {None, 'unknown'})
    else:
        model_cols = sorted(set(y_pred_model))
        fam_cols = sorted(set(y_pred_family) - {'unknown'})
        br_cols = sorted(set(y_pred_branch) - {'unknown'})

    cm_model = create_rectangular_cm(row_labels, model_cols, y_true, y_pred_model)
    cm_family = create_rectangular_cm(row_labels, fam_cols, y_true, y_pred_family)
    cm_branch = create_rectangular_cm(row_labels, br_cols, y_true, y_pred_branch)

    return {
        'model': {'cm': cm_model, 'ylabels': row_labels, 'xlabels': model_cols},
        'family': {'cm': cm_family, 'ylabels': row_labels, 'xlabels': fam_cols},
        'branch': {'cm': cm_branch, 'ylabels': row_labels, 'xlabels': br_cols}
    }


def create_rectangular_cm(rows, cols, y_t, y_p):
    r_index = {r: i for i, r in enumerate(rows)}
    c_index = {c: j for j, c in enumerate(cols)}
    cm = np.zeros((len(rows), len(cols)), dtype=int)
    for t, p in zip(y_t, y_p):
        if t in r_index and p in c_index:
            cm[r_index[t], c_index[p]] += 1
    return cm


def compute_held_out_llm_metrics(predictions_with_scores: dict[str, dict], meta_map: dict,
                                 threshold: float) -> dict:
    """
    Computes metrics for held-out data.
    - OOD Detection Accuracy: Correctly identifying a held-out model as 'unknown'.
    - Family/Branch Accuracy (identified-only): Computed ONLY on predictions that were NOT classified as 'unknown'.
    - Family/Branch Accuracy (all): Computed on ALL examples, even when the prediction score < threshold.
    Note: false_positive_rate is not applicable for held-out (all examples are OOD w.r.t. the library).
    """
    if not meta_map:
        return {"error": "Model metadata not available."}

    identified_preds, unknown_preds = 0, 0

    # Identified-only counters
    family_matches_top1_id, branch_matches_top1_id = 0, 0
    family_matches_top3_id, branch_matches_top3_id = 0, 0

    # All-sample counters (includes those below threshold)
    family_matches_top1_all, branch_matches_top1_all = 0, 0
    family_matches_top3_all, branch_matches_top3_all = 0, 0

    for true_llm, prompt_preds in predictions_with_scores.items():
        for pred_list_with_scores in prompt_preds.values():
            if not pred_list_with_scores:
                unknown_preds += 1  # No prediction is treated as 'unknown'
                continue

            # Compute family/branch matches regardless of threshold (ALL)
            true_family, true_branch = get_llm_family_and_branch(true_llm, meta_map)
            pred_llm_top1 = pred_list_with_scores[0][0]
            pred_family_top1, pred_branch_top1 = get_llm_family_and_branch(pred_llm_top1, meta_map)
            if true_family != 'unknown' and pred_family_top1 != 'unknown' and true_family == pred_family_top1:
                family_matches_top1_all += 1
            if true_branch != 'unknown' and pred_branch_top1 != 'unknown' and true_branch == pred_branch_top1:
                branch_matches_top1_all += 1

            top3_families = {get_llm_family_and_branch(p, meta_map)[0] for p, _ in pred_list_with_scores}
            top3_branches = {get_llm_family_and_branch(p, meta_map)[1] for p, _ in pred_list_with_scores}
            if true_family != 'unknown' and 'unknown' not in top3_families and true_family in top3_families:
                family_matches_top3_all += 1
            if true_branch != 'unknown' and 'unknown' not in top3_branches and true_branch in top3_branches:
                branch_matches_top3_all += 1

            # Threshold-based OOD detection and identified-only metrics
            max_score = pred_list_with_scores[0][1]
            if max_score < threshold:
                unknown_preds += 1  # Correctly flagged as OOD if truly held-out
            else:
                identified_preds += 1
                if (true_family != 'unknown' and pred_family_top1 != 'unknown' and
                        true_family == pred_family_top1):
                    family_matches_top1_id += 1
                if (true_branch != 'unknown' and pred_branch_top1 != 'unknown' and
                        true_branch == pred_branch_top1):
                    branch_matches_top1_id += 1
                if (true_family != 'unknown' and 'unknown' not in top3_families and
                        true_family in top3_families):
                    family_matches_top3_id += 1
                if (true_branch != 'unknown' and 'unknown' not in top3_branches and
                        true_branch in top3_branches):
                    branch_matches_top3_id += 1

    total_predictions = identified_preds + unknown_preds
    if total_predictions == 0:
        return {
            # Identified-only
            'top1_family_accuracy_identified': 0,
            'top3_family_accuracy_identified': 0,
            'top1_branch_accuracy_identified': 0,
            'top3_branch_accuracy_identified': 0,

            # All-sample metrics (requested behavior even when OOD)
            'top1_family_accuracy_all': 0,
            'top3_family_accuracy_all': 0,
            'top1_branch_accuracy_all': 0,
            'top3_branch_accuracy_all': 0,

            # Counts and OOD stats
            'total_predictions': 0,
            'total_identified': 0,
            'ood_detection_accuracy': 0,
            'misidentification_rate': 0
        }

    # Identified-only ratios
    top1_family_id_acc = family_matches_top1_id / identified_preds if identified_preds > 0 else 0
    top3_family_id_acc = family_matches_top3_id / identified_preds if identified_preds > 0 else 0
    top1_branch_id_acc = branch_matches_top1_id / identified_preds if identified_preds > 0 else 0
    top3_branch_id_acc = branch_matches_top3_id / identified_preds if identified_preds > 0 else 0

    # All-sample ratios
    top1_family_all_acc = family_matches_top1_all / total_predictions
    top3_family_all_acc = family_matches_top3_all / total_predictions
    top1_branch_all_acc = branch_matches_top1_all / total_predictions
    top3_branch_all_acc = branch_matches_top3_all / total_predictions

    return {
        # Identified-only
        'top1_family_accuracy_identified': top1_family_id_acc,
        'top3_family_accuracy_identified': top3_family_id_acc,
        'top1_branch_accuracy_identified': top1_branch_id_acc,
        'top3_branch_accuracy_identified': top3_branch_id_acc,

        # All-sample metrics (requested behavior even when OOD)
        'top1_family_accuracy_all': top1_family_all_acc,
        'top3_family_accuracy_all': top3_family_all_acc,
        'top1_branch_accuracy_all': top1_branch_all_acc,
        'top3_branch_accuracy_all': top3_branch_all_acc,

        # Counts and OOD stats
        'total_predictions': total_predictions,
        'total_identified': identified_preds,
        'ood_detection_accuracy': unknown_preds / total_predictions,
        'misidentification_rate': identified_preds / total_predictions
    }


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
            if not group.empty:
                # detect_refusal expects raw text; ensure 'response' is present in the df
                responses = group['response'].tolist() if 'response' in group.columns else []
                is_refusal_list = [detect_refusal(r) for r in responses] if responses else []
                is_refusal = (sum(is_refusal_list) / len(is_refusal_list)) > 0.5 if is_refusal_list else False
                flat_rows.append({
                    'true_llm': llm,
                    'combo': combo,
                    'pred_list': pred_list,
                    'is_refusal': is_refusal
                })

    flat_df = pd.DataFrame(flat_rows)
    if flat_df.empty:  # Handle case with no data
        return {'overall': {}, 'refusal': {}, 'non_refusal': {}}

    # Helper to compute subset metrics
    def compute_subset(df_subset, threshold):
        subset_preds = {}
        for _, row in df_subset.iterrows():
            if row['true_llm'] not in subset_preds:
                subset_preds[row['true_llm']] = {}
            subset_preds[row['true_llm']][row['combo']] = row['pred_list']

        if not subset_preds: return {}  # Return empty dict if no predictions in subset

        if is_held_out:
            metrics = compute_held_out_llm_metrics(subset_preds, meta_map, threshold)
        else:
            metrics = compute_metrics(subset_preds, threshold)
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