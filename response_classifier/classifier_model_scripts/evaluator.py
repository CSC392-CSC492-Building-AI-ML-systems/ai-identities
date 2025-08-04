import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, \
    confusion_matrix
from sklearn.preprocessing import normalize
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os
from classifier_model import compute_library_averages, predict_unknown, get_metric_func
from data_processor import load_processed, load_fitted_vectorizer, process_word_freq, process_embeddings
from llm_meta_data import load_llm_meta_data, get_llm_family_and_branch


def compute_metrics(predictions: dict[str, dict[str, list[str]]]) -> dict:
    """
    Compute metrics: top1/top3 acc, f1, etc. + confusion matrix.
    Assumes predictions are nested: {llm: {prompt: top-k list}}.
    Flattens to per (llm, prompt) predictions.
    """
    # For each (llm, user prompt) pairs
    true_labels = []
    top1_pred_labels = []
    top3_pred_lists = []
    for llm, prompt_preds in predictions.items():
        for prompt, pred_list in prompt_preds.items():
            true_labels.append(llm)
            top1_pred_labels.append(pred_list[0])
            top3_pred_lists.append(pred_list)

    top1_acc = accuracy_score(true_labels, top1_pred_labels)
    top3_acc = np.mean([true in top3 for true, top3 in zip(true_labels, top3_pred_lists)])
    f1 = f1_score(true_labels, top1_pred_labels, average='macro')
    precision = precision_score(true_labels, top1_pred_labels, average='macro')
    recall = recall_score(true_labels, top1_pred_labels, average='macro')

    # Modified: Include all unique labels from true and pred to avoid ValueError
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


def compute_held_out_llm_metrics(predictions: dict, meta_map: dict) -> dict:
    """
    Computes top-1 and top-3 accuracy at the model family and branch levels.
    """
    if not meta_map:
        return {"error": "Model metadata not available."}

    family_matches_top1, branch_matches_top1 = 0, 0
    family_matches_top3, branch_matches_top3 = 0, 0
    total_preds = 0

    for true_llm, prompt_preds in predictions.items():
        for pred_list in prompt_preds.values():
            if not pred_list:
                continue

            total_preds += 1
            true_family, true_branch = get_llm_family_and_branch(true_llm, meta_map)
            if true_family == 'unknown': continue  # Skip if true model has no defined family

            # Top-1 Taxonomy Check
            pred_family_top1, pred_branch_top1 = get_llm_family_and_branch(pred_list[0], meta_map)
            if true_family == pred_family_top1:
                family_matches_top1 += 1
            if true_branch == pred_branch_top1:
                branch_matches_top1 += 1

            # Top-3 Taxonomy Check
            top3_families = {get_llm_family_and_branch(p, meta_map)[0] for p in pred_list}
            top3_branches = {get_llm_family_and_branch(p, meta_map)[1] for p in pred_list}
            if true_family in top3_families:
                family_matches_top3 += 1
            if true_branch in top3_branches:
                branch_matches_top3 += 1

    if total_preds == 0: return {}

    return {
        'top1_family_identification_accuracy': family_matches_top1 / total_preds,
        'top3_family_identification_accuracy': family_matches_top3 / total_preds,
        'top1_branch_identification_accuracy': branch_matches_top1 / total_preds,
        'top3_branch_identification_accuracy': branch_matches_top3 / total_preds,
        'total_predictions': total_preds
    }


def evaluate_discriminative_prompts(unknown_data: dict,
                                    predictions: dict[str, dict[str, str]]) -> dict:
    """
    Per-prompt accuracy to find discriminative prompts.
    Flattens to per (model, prompt) for accuracy calculation.
    """
    per_prompt_acc = {}
    # Flatten to DF for grouping
    rows = []
    for model, df in unknown_data.items():
        for prompt in df['prompt'].unique():
            pred = predictions.get(model, {}).get(prompt, 'unknown')
            rows.append({'model': model, 'prompt': prompt, 'prediction': pred})
    all_df = pd.DataFrame(rows)

    for prompt, group in all_df.groupby('prompt'):
        per_prompt_acc[prompt] = accuracy_score(group['model'], group['prediction'])
    # Sort by accuracy descending
    sorted_prompts = dict(
        sorted(per_prompt_acc.items(), key=lambda x: x[1], reverse=True))

    return sorted_prompts


def save_confusion_matrix(cm: np.ndarray, labels: list[str], output_path: str) -> None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, xticklabels=labels, yticklabels=labels)
    plt.savefig(output_path)
    plt.close()


def analyze_ood_performance(predictions_with_scores: dict, threshold: float,
                            meta_map: dict) -> dict:
    """
    Analyzes OOD performance using a confidence threshold.
    `predictions_with_scores` format: {true_model: {prompt: [(pred_model, score), ...]}}
    """
    known_count = 0
    unknown_count = 0
    correctly_identified_as_unknown = 0

    family_suggestions_when_unknown = []

    for true_model, prompt_preds in predictions_with_scores.items():
        for pred_list in prompt_preds.values():
            if not pred_list: continue

            top1_pred, top1_score = pred_list[0]

            if top1_score >= threshold:
                # The model is confident, but for a held-out set, this is an error.
                known_count += 1
            else:
                # The model correctly identifies the prediction as low-confidence (Unknown).
                unknown_count += 1
                correctly_identified_as_unknown += 1

                # Check if the suggested family was correct
                true_family, _ = get_llm_family_and_branch(true_model, meta_map)
                pred_family, _ = get_llm_family_and_branch(top1_pred, meta_map)
                family_suggestions_when_unknown.append(true_family == pred_family)

    ood_accuracy = correctly_identified_as_unknown / len(
        predictions_with_scores) if predictions_with_scores else 0
    family_suggestion_accuracy = sum(family_suggestions_when_unknown) / len(
        family_suggestions_when_unknown) if family_suggestions_when_unknown else 0

    return {
        "ood_detection_accuracy": ood_accuracy,
        # How often it correctly flagged an OOD sample as "unknown"
        "family_suggestion_accuracy_when_unknown": family_suggestion_accuracy,
        # When it says "unknown", how often is its family guess right?
        "confidence_threshold": threshold
    }


def evaluate_final_model(selected_clf_method: dict, train_data: dict, test_data: dict,
                         held_out_data: dict, llm_meta_data_path: str = '../configs/llm_set.json',
                         output_file: str = '../results/final_eval.txt') -> dict:
    """
    Evaluate selected method on test and held-out, save to text file.
    """
    clf_method_name = selected_clf_method['name']
    llm_meta_map = load_llm_meta_data(llm_meta_data_path)

    # Load or process train (with fitting for word freq)
    train_output_path = f"../data/processed/{clf_method_name}/train/"
    try:
        train_processed = load_processed(clf_method_name, 'train')
    except FileNotFoundError:
        if 'vectorizer' in selected_clf_method:
            train_processed, _ = process_word_freq(train_data, selected_clf_method, train_output_path)
        else:
            train_processed = process_embeddings(train_data, selected_clf_method, train_output_path)

    # For test and held-out: Use train's fitted vectorizer if word freq
    for split_name, split_data in [('test', test_data), ('held_out', held_out_data)]:
        split_output_path = f"../data/processed/{clf_method_name}/{split_name}/"
        try:
            processed = load_processed(clf_method_name, split_name)
        except FileNotFoundError:
            if 'vectorizer' in selected_clf_method:
                fitted_vectorizer = load_fitted_vectorizer(clf_method_name, 'train')
                if fitted_vectorizer is None:
                    raise FileNotFoundError("Train must be processed first for word freq methods")
                processed, _ = process_word_freq(split_data, selected_clf_method, split_output_path, fitted_vectorizer)
            else:
                processed = process_embeddings(split_data, selected_clf_method, split_output_path)
        if split_name == 'test':
            test_processed = processed
        else:
            held_out_processed = processed

    # Normalization logic: Only for word freq + Euclidean; skip for cosine or embeddings
    do_normalize = (('vectorizer' in selected_clf_method) and
                    (selected_clf_method['metric'] == 'euclidean_distances') and
                    selected_clf_method.get('normalize_vectors', False))

    library_avgs = compute_library_averages(train_processed)  # Use full train as library
    if do_normalize:
        for key, avg_vec in library_avgs.items():
            library_avgs[key] = normalize(avg_vec.reshape(1, -1), norm='l2')[0]

    metric_func = get_metric_func(selected_clf_method['metric'])

    # Test eval
    test_preds = predict_unknown(test_processed, library_avgs, metric_func,
                                 top_k=3, do_normalize=do_normalize)
    test_metrics = compute_metrics(test_preds)
    test_top1_preds = {m: {p: pr[0] if pr else 'unknown' for p, pr in prompts.items()}
                       for m, prompts in test_preds.items()}
    test_metrics['discriminative_prompts'] = evaluate_discriminative_prompts(
        test_processed, test_top1_preds)

    # Held-out eval (treat as unknown) - no discriminative prompts here
    held_out_preds = predict_unknown(held_out_processed, library_avgs, metric_func,
                                     top_k=3, do_normalize=do_normalize)
    held_out_metrics = compute_held_out_llm_metrics(held_out_preds, llm_meta_map)

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    results = {'test': test_metrics, 'held_out': held_out_metrics}
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

    # Collect test labels: all true models + all top-1 predicted models
    test_true_labels = set(test_preds.keys())
    test_pred_labels = set()
    for prompts in test_preds.values():
        for pred_list in prompts.values():
            if pred_list:  # Skip empty
                test_pred_labels.add(pred_list[0])  # Top-1 pred
    test_labels = sorted(test_true_labels | test_pred_labels)

    # Collect held-out labels: all true models + all top-1 predicted models
    held_out_true_labels = set(held_out_preds.keys())
    held_out_pred_labels = set()
    for prompts in held_out_preds.values():
        for pred_list in prompts.values():
            if pred_list:  # Skip empty
                held_out_pred_labels.add(pred_list[0])  # Top-1 pred
    held_out_labels = sorted(held_out_true_labels | held_out_pred_labels)

    save_confusion_matrix(np.array(test_metrics['confusion_matrix']), test_labels,
                          '../results/test_cm.png')
    save_confusion_matrix(np.array(held_out_metrics['confusion_matrix']), held_out_labels,
                          '../results/held_out_cm.png')

    return results