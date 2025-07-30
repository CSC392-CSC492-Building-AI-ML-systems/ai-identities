import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, \
    confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import json
from classifier_model import compute_library_averages, predict_unknown, get_metric_func
from data_processor import process_and_save, load_processed, process_word_freq, process_embeddings, load_fitted_vectorizer
from sklearn.preprocessing import normalize
import os


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


def evaluate_final_model(selected_method: dict, train_data: dict, test_data: dict,
                         held_out_data: dict,
                         output_file: str = '../results/final_eval.txt') -> dict:
    """
    Evaluate selected method on test and held-out, save to text file.
    """
    method_name = selected_method['name']

    # Load or process train (with fitting for word freq)
    train_output_path = f"../data/processed/{method_name}/train/"
    try:
        train_processed = load_processed(method_name, 'train')
    except FileNotFoundError:
        if 'vectorizer' in selected_method:
            train_processed, _ = process_word_freq(train_data, selected_method, train_output_path)
        else:
            train_processed = process_embeddings(train_data, selected_method, train_output_path)

    # For test and held-out: Use train's fitted vectorizer if word freq
    for split_name, split_data in [('test', test_data), ('held_out', held_out_data)]:
        split_output_path = f"../data/processed/{method_name}/{split_name}/"
        try:
            processed = load_processed(method_name, split_name)
        except FileNotFoundError:
            if 'vectorizer' in selected_method:
                fitted_vectorizer = load_fitted_vectorizer(method_name, 'train')
                if fitted_vectorizer is None:
                    raise FileNotFoundError("Train must be processed first for word freq methods")
                processed, _ = process_word_freq(split_data, selected_method, split_output_path, fitted_vectorizer)
            else:
                processed = process_embeddings(split_data, selected_method, split_output_path)
        if split_name == 'test':
            test_processed = processed
        else:
            held_out_processed = processed

    # Normalization logic: Only for word freq + Euclidean; skip for cosine or embeddings
    do_normalize = (('vectorizer' in selected_method) and
                    (selected_method['metric'] == 'euclidean_distances') and
                    selected_method.get('normalize_vectors', False))

    library_avgs = compute_library_averages(train_processed)  # Use full train as library
    if do_normalize:
        for key, avg_vec in library_avgs.items():
            library_avgs[key] = normalize(avg_vec.reshape(1, -1), norm='l2')[0]

    metric_func = get_metric_func(selected_method['metric'])

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
    held_out_metrics = compute_metrics(held_out_preds)

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