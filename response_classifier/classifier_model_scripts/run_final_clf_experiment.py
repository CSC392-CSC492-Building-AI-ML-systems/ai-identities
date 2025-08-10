import yaml
import os
import pickle
import pandas as pd
import argparse
from data_loader import load_raw_data
from data_splitter import hold_out_models, split_tuning_test
from data_processor import process_word_freq
from classifier_model import get_metric_func, predict_unknown
from evaluator import compute_metrics_with_refusals, tune_threshold, apply_threshold
from llm_meta_data import load_llm_meta_data
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def save_dict_as_pickle(data_dict: dict, path: str):
    """Helper to save a dict of DataFrames as pickle."""
    with open(path, 'wb') as f:
        pickle.dump(data_dict, f)


def validate_final_clf_splits(non_held_out_tuning: dict, non_held_out_test: dict,
                              held_out_tuning: dict, held_out_test: dict,
                              bins: dict) -> bool:
    """
    Validate final_clf splits: check sizes, bin presence, and 50/50 ratios per (model, prompt, system_prompt, temp_bin).
    Adapted from validate_splits.py for final_clf (with system_prompt).
    """
    all_models = set(non_held_out_tuning.keys()) | set(non_held_out_test.keys()) | \
                 set(held_out_tuning.keys()) | set(held_out_test.keys())

    # Assuming 1 prompt (best_user_prompt), but multiple system_prompts; adjust totals accordingly
    # You'd need to know expected num_system_prompts from config/data; here assuming it's consistent
    expected_total_per_model_prompt_bin = {bin_name: bin_info['num_points']
                                           for bin_name, bin_info in bins.items()}

    for model in all_models:
        if model in held_out_tuning or model in held_out_test:  # Held-out check
            tuning_df = held_out_tuning.get(model, pd.DataFrame())
            test_df = held_out_test.get(model, pd.DataFrame())
        else:
            tuning_df = non_held_out_tuning.get(model, pd.DataFrame())
            test_df = non_held_out_test.get(model, pd.DataFrame())

        for (prompt, sys_prompt, bin_name), group_tuning in tuning_df.groupby(
                ['prompt', 'system_prompt', 'temp_bin']):
            group_test = test_df[(test_df['prompt'] == prompt) &
                                 (test_df['system_prompt'] == sys_prompt) &
                                 (test_df['temp_bin'] == bin_name)]

            expected_total = expected_total_per_model_prompt_bin.get(bin_name, 0)
            expected_tuning = expected_total // 2  # 50/50 split
            expected_test = expected_total - expected_tuning

            if len(group_tuning) != expected_tuning or len(group_test) != expected_test:
                print(
                    f"Size mismatch for {model}, prompt: {prompt}, system_prompt: {sys_prompt}, bin: {bin_name}")
                print(f"Expected tuning: {expected_tuning}, actual: {len(group_tuning)}")
                print(f"Expected test: {expected_test}, actual: {len(group_test)}")
                return False

            if len(group_tuning) == 0 or len(group_test) == 0:
                print(
                    f"Missing data for {model}, prompt: {prompt}, system_prompt: {sys_prompt}, bin: {bin_name}")
                return False

    print("All final_clf splits validated successfully.")
    return True


def generate_final_report(output_path: str, metrics: dict or dict[float, dict],
                          phase: str, threshold: float = None):
    """
    Generate detailed report for tuning (multi-threshold) or test (single-threshold) phase.
    Uses tabulate to format metrics into tables for better readability.
    """
    def create_big_table(thresh_metrics: dict, metric_keys: list[str]) -> str:
        """Helper to create a single tabulate table with thresholds as rows and metrics as columns."""
        table_data = []
        for thresh in sorted(thresh_metrics.keys()):
            row = [thresh]
            for key in metric_keys:
                value = thresh_metrics[thresh].get(key, 'N/A')
                row.append(f"{value:.4f}" if isinstance(value, float) else value)
            table_data.append(row)
        headers = ['Threshold'] + metric_keys
        return tabulate(table_data, headers=headers, tablefmt='simple')

    with open(output_path, 'w') as f:
        f.write(f"========= Final CLF {phase.capitalize()} Report =========\n\n")

        if isinstance(metrics, dict) and all(isinstance(k, float) for k in metrics):  # Multi-threshold (tuning)
            f.write("Results for Multiple Thresholds:\n\n")

            # Collect metrics by split and category into nested dicts
            collected = {
                'non_held_out': {'overall': {}, 'refusal': {}, 'non_refusal': {}},
                'held_out': {'overall': {}, 'refusal': {}, 'non_refusal': {}}
            }
            for thresh, thresh_metrics in metrics.items():
                for split_name in ['non_held_out', 'held_out']:
                    split_data = thresh_metrics[split_name]
                    for category in ['overall', 'refusal', 'non_refusal']:
                        collected[split_name][category][thresh] = split_data[category]

            # Generate 6 big tables
            for split_name in ['non_held_out', 'held_out']:
                f.write(f"--- {split_name.replace('_', ' ').title()} Metrics ---\n\n")
                for category in ['overall', 'refusal', 'non_refusal']:
                    f.write(f"  {category.capitalize()} Table:\n")
                    if split_name == 'non_held_out':
                        # Common metrics for non-held-out
                        metric_keys = ['top1_accuracy', 'top3_accuracy', 'f1_score', 'precision', 'recall']
                    else:
                        # Common metrics for held-out
                        metric_keys = [
                            'top1_family_identification_accuracy', 'top3_family_identification_accuracy',
                            'top1_branch_identification_accuracy', 'top3_branch_identification_accuracy',
                            'total_predictions', 'ood_detection_accuracy', 'false_positive_rate'
                        ]
                    table_str = create_big_table(collected[split_name][category], metric_keys)
                    f.write(table_str + "\n\n")
        else:  # Single threshold (test) - keep original behavior
            f.write(f"Threshold: {threshold}\n\n")
            for split_name in ['non_held_out', 'held_out']:
                f.write(f"--- {split_name.replace('_', ' ').title()} Metrics ---\n\n")
                split_metrics = metrics[split_name]
                for category in ['overall', 'refusal', 'non_refusal']:
                    f.write(f"  {category.capitalize()}:\n")
                    table_data = [[k, f"{v:.4f}" if isinstance(v, float) else v] for k, v in split_metrics[category].items() if k != 'confusion_matrix']
                    table_str = tabulate(table_data, headers=['Metric', 'Value'], tablefmt='simple')
                    f.write(table_str + "\n\n")

        f.write("\n========= End of Report =========\n")


def save_confusion_matrices(non_held_metrics: dict, held_metrics: dict, output_dir: str):
    """
    Compute and save confusion matrix plots for non_held_out and held_out test sets.
    """
    def plot_cm(cm, labels, title, file_path):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()

    # For non_held_out: Use existing confusion_matrix from overall metrics
    if 'confusion_matrix' in non_held_metrics['overall']:
        cm_non_held = np.array(non_held_metrics['overall']['confusion_matrix'])
        # Assume labels are all unique true and pred labels; for simplicity, we can extract from metrics or hardcode if needed
        # MODIFIED: To make it work, we'll assume labels are sorted unique models; in practice, pass labels if available
        labels_non_held = sorted(set(range(cm_non_held.shape[0])))  # Placeholder; replace with actual labels if computed
        plot_cm(cm_non_held, labels_non_held, 'Non-Held-Out Test Confusion Matrix', os.path.join(output_dir, 'non_held_out_test_cm.png'))

    # For held_out: Compute a custom CM, treating "unknown" as a label, and map to families/branches
    # ADDED: Custom logic for held_out CM since original metrics don't include it
    # Note: This requires predictions to include "unknown"; we can compute CM based on top-1 pred vs true, with "unknown" as extra class
    # For now, placeholder: Implement actual CM computation if needed, or skip if not applicable
    # Example placeholder:
    # cm_held = confusion_matrix(true_labels_held, pred_labels_held, labels=labels_held)
    # plot_cm(cm_held, labels_held, 'Held-Out Test Confusion Matrix', os.path.join(output_dir, 'held_out_test_cm.png'))
    print("Confusion matrix plotting for held_out is placeholder; implement if needed.")


def run_final_clf_experiment(args):
    # Load config and raw data
    data_config = yaml.safe_load(open('../configs/data_config.yaml'))
    methods_cfg = yaml.safe_load(open('../configs/classification_methods_config.yaml'))
    raw_path = '../data/final_clf_dataset_raw_data/'
    raw_data, _, _, _ = load_raw_data(raw_path)

    # Define paths
    splits_path = '../data/splits/final_clf/'
    os.makedirs(splits_path, exist_ok=True)
    non_held_out_tuning_path = os.path.join(splits_path, 'non_held_out_tuning.pkl')
    non_held_out_test_path = os.path.join(splits_path, 'non_held_out_test.pkl')
    held_out_tuning_path = os.path.join(splits_path, 'held_out_tuning.pkl')
    held_out_test_path = os.path.join(splits_path, 'held_out_test.pkl')

    # Common setup: Classifier details
    clf_method_name = 'tfidf_trigram'
    for method in methods_cfg['word_freq']:
        if method['name'] == clf_method_name:
            clf_method = method
    vectorizer_path = '../data/processed/tfidf_trigram/train/vectorizer.pkl'
    library_avgs_path = '../data/processed/tfidf_trigram/train/library_averages.pkl'

    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(library_avgs_path, 'rb') as f:
        library_avgs = pickle.load(f)

    metric_func = get_metric_func('cosine_similarity')
    meta_map = load_llm_meta_data('../configs/llm_set.json')

    if args.action == 'split':
        print("Performing data splitting...")
        non_held_out, held_out = hold_out_models(raw_data, data_config)
        bins = data_config['final_clf_dataset_temp_bins']
        seed = data_config['random_seed']

        non_held_out_tuning, non_held_out_test = split_tuning_test(non_held_out, bins,
                                                                   seed)
        held_out_tuning, held_out_test = split_tuning_test(held_out, bins, seed)

        # Directly save using helper
        save_dict_as_pickle(non_held_out_tuning, non_held_out_tuning_path)
        save_dict_as_pickle(non_held_out_test, non_held_out_test_path)
        save_dict_as_pickle(held_out_tuning, held_out_tuning_path)
        save_dict_as_pickle(held_out_test, held_out_test_path)
        print(f"Splits saved to {splits_path}")

    elif args.action == 'validate_splits':
        print("Validating splits...")
        # Load splits
        with open(non_held_out_tuning_path, 'rb') as f:
            non_held_out_tuning = pickle.load(f)
        with open(non_held_out_test_path, 'rb') as f:
            non_held_out_test = pickle.load(f)
        with open(held_out_tuning_path, 'rb') as f:
            held_out_tuning = pickle.load(f)
        with open(held_out_test_path, 'rb') as f:
            held_out_test = pickle.load(f)

        bins = data_config['final_clf_dataset_temp_bins']
        validate_final_clf_splits(non_held_out_tuning, non_held_out_test, held_out_tuning,
                                  held_out_test, bins)

    elif args.action == 'tune':
        # Perform splitting if not already done
        if not all(os.path.exists(p) for p in [non_held_out_tuning_path, non_held_out_test_path, held_out_tuning_path, held_out_test_path]):
            print("Performing data splitting...")
            non_held_out, held_out = hold_out_models(raw_data, data_config)
            bins = data_config['final_clf_dataset_temp_bins']
            seed = data_config['random_seed']

            non_held_out_tuning, non_held_out_test = split_tuning_test(non_held_out, bins, seed)
            held_out_tuning, held_out_test = split_tuning_test(held_out, bins, seed)

            # Directly save using helper
            save_dict_as_pickle(non_held_out_tuning, non_held_out_tuning_path)
            save_dict_as_pickle(non_held_out_test, non_held_out_test_path)
            save_dict_as_pickle(held_out_tuning, held_out_tuning_path)
            save_dict_as_pickle(held_out_test, held_out_test_path)
            print(f"Splits saved to {splits_path}")
        else:
            print("Splits already exist; skipping splitting.")

        # Load raw tuning splits
        with open(non_held_out_tuning_path, 'rb') as f:
            non_held_out_tuning_raw = pickle.load(f)
        with open(held_out_tuning_path, 'rb') as f:
            held_out_tuning_raw = pickle.load(f)

        # Vectorize tuning splits
        print("Vectorizing non-held out tuning set")
        non_held_out_tuning_vec, _ = process_word_freq(non_held_out_tuning_raw, clf_method,
                                                       output_path='',
                                                       fitted_vectorizer=vectorizer,
                                                       drop_response=False)
        print("Vectorizing held out tuning set")
        held_out_tuning_vec, _ = process_word_freq(held_out_tuning_raw, clf_method,
                                                   output_path='',
                                                   fitted_vectorizer=vectorizer,
                                                   drop_response=False)

        # Evaluate on tuning sets across thresholds
        print("Evaluating on tuning sets...")
        thresholds = [i / 100 for i in range(5, 100, 5)]  # 0.05 to 0.95
        tuning_results = tune_threshold(non_held_out_tuning_vec, held_out_tuning_vec,
                                        library_avgs, metric_func, meta_map, thresholds)

        # Generate report with all threshold results
        generate_final_report('../results/final_clf_tuning_report.txt',
                              tuning_results, 'tuning')

    elif args.action == 'evaluate':
        if args.threshold is None:
            raise ValueError("For 'evaluate' action, --threshold is required (e.g., 0.8)")

        # Load raw test splits (assume they exist from 'tune' action)
        if not os.path.exists(non_held_out_test_path) or not os.path.exists(held_out_test_path):
            raise FileNotFoundError("Test splits not found; run '--action tune' first.")

        with open(non_held_out_test_path, 'rb') as f:
            non_held_out_test_raw = pickle.load(f)
        with open(held_out_test_path, 'rb') as f:
            held_out_test_raw = pickle.load(f)

        # Vectorize test splits
        print("Vectorizing non-held out test set")
        non_held_out_test_vec, _ = process_word_freq(non_held_out_test_raw, clf_method,
                                                     output_path='', fitted_vectorizer=vectorizer)
        print("Vectorizing held out test set")
        held_out_test_vec, _ = process_word_freq(held_out_test_raw, clf_method,
                                                 output_path='', fitted_vectorizer=vectorizer)

        # Predict with scores
        print("Evaluating on non-held out test set")
        non_held_preds = predict_unknown(non_held_out_test_vec, library_avgs, metric_func,
                                         return_scores=True, single_prompt_mode=True)
        print("Evaluating on held out test set")
        held_preds = predict_unknown(held_out_test_vec, library_avgs, metric_func,
                                     return_scores=True, single_prompt_mode=True)

        # Compute metrics
        test_metrics = {
            'non_held_out': compute_metrics_with_refusals(non_held_preds,
                                                          non_held_out_test_raw,
                                                          meta_map, is_held_out=False,
                                                          threshold=args.threshold),
            'held_out': compute_metrics_with_refusals(held_preds, held_out_test_raw,
                                                      meta_map, is_held_out=True,
                                                      threshold=args.threshold)
        }

        save_confusion_matrices(test_metrics['non_held_out'], test_metrics['held_out'],
                                '../results/')

        # Generate report
        generate_final_report('../results/final_clf_test_report.txt',
                              test_metrics, 'test', args.threshold)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Final CLF experiment script.")
    parser.add_argument('--action', type=str, required=True,
                        choices=['split', 'validate_splits', 'tune', 'evaluate'],
                        help="Action: 'split' to split data; 'validate_splits' to "
                             "validate splits; 'tune' to evaluate thresholds on tuning "
                             "sets; 'evaluate' to evaluate on test sets.")
    parser.add_argument('--threshold', type=float, default=None,
                        help="Threshold value for 'evaluate' action (e.g., 0.8).")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    run_final_clf_experiment(args)