from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
from evaluator import compute_metrics
from classifier_model import compute_library_averages, predict_unknown, get_metric_func
from data_processor import process_word_freq, process_and_save, load_processed
import json
from sklearn.preprocessing import normalize
import time
from tqdm import tqdm
import os


def perform_5fold_cv_for_method(train_data: dict[str, pd.DataFrame], clf_method: dict,
                                output_file: str = '../results/cv_metrics.txt',
                                eval_both_metrics: bool = False) -> dict:
    """
    5-fold CV for a single method, appends avg metrics to text file.
    Optionally evaluates with both cosine and Euclidean metrics in one go.
    Records processing/inference times per fold and averages.
    """
    clf_method_name = clf_method['name']

    start_total = time.time()  # Total CV time

    # Flatten train_data into single DF for splitting (use raw, process per fold for word freq, once for embeddings)
    # Sort keys for consistent concat order
    sorted_keys = sorted(train_data.keys())
    all_train_raw = pd.concat([train_data[key].assign(model=key) for key in sorted_keys])
    all_train_raw['stratum'] = (
            all_train_raw['model'] + '|' +
            all_train_raw['prompt'] + '|' +
            all_train_raw['temp_bin']
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Pre-process the full train once for embeddings to avoid recomputing per fold (efficient)
    # For word freq, we'll process per fold later to fit only on train_fold
    full_processed = None
    embedding_preprocessing_time = 0.0
    if 'vectorizer' not in clf_method:  # Embeddings only
        process_start = time.time()
        try:
            full_processed = load_processed(clf_method_name, 'train')
        except FileNotFoundError:
            full_processed = process_and_save(train_data, clf_method, clf_method_name, 'train')
        embedding_preprocessing_time = time.time() - process_start

    fold_metrics_dict = {}  # Keyed by metric name (e.g., 'cosine', 'euclidean')
    fold_times = []  # Per-fold times

    for fold_num, (train_idx, val_idx) in enumerate(
            tqdm(skf.split(all_train_raw, all_train_raw['stratum']),
                 total=skf.get_n_splits(), desc=f"CV: {clf_method_name}", unit="fold")):
        fold_start = time.time()

        train_fold_raw = all_train_raw.iloc[train_idx]
        val_fold_raw = all_train_raw.iloc[val_idx]

        # Group back into dicts (raw)
        train_lib_raw = {model: group.drop(columns=['model']) for model, group in train_fold_raw.groupby('model')}
        val_unknown_raw = {model: group.drop(columns=['model']) for model, group in val_fold_raw.groupby('model')}

        # Process per fold (for word freq) or subset pre-processed (for embeddings)
        process_start = time.time()
        if 'vectorizer' in clf_method:
            # Word freq: Fit on train_fold only, transform val_fold (no save, in-memory)
            train_lib, fitted_vectorizer = process_word_freq(train_lib_raw, clf_method, output_path='', fitted_vectorizer=None)
            val_unknown, _ = process_word_freq(val_unknown_raw, clf_method, output_path='', fitted_vectorizer=fitted_vectorizer)
        else:
            # Embeddings: Subset the pre-processed full data using indices
            # Sort keys for consistent concat order
            all_train_processed = pd.concat([full_processed[key].assign(model=key) for key in sorted(full_processed.keys())])
            train_fold_processed = all_train_processed.iloc[train_idx]
            val_fold_processed = all_train_processed.iloc[val_idx]
            train_lib = {model: group.drop(columns=['model']) for model, group in train_fold_processed.groupby('model')}
            val_unknown = {model: group.drop(columns=['model']) for model, group in val_fold_processed.groupby('model')}
        processing_time = time.time() - process_start

        # Compute library averages (with optional normalization)
        avg_start = time.time()
        library_avgs = compute_library_averages(train_lib)
        avg_time = time.time() - avg_start

        # Define metrics to evaluate
        primary_metric_name = clf_method['metric']
        metrics_to_eval = [primary_metric_name]
        if eval_both_metrics:
            other_metric = 'euclidean_distances' if primary_metric_name == 'cosine_similarity' else 'cosine_similarity'
            metrics_to_eval.append(other_metric)

        pred_time_total = 0
        metrics_time_total = 0
        for metric_name in metrics_to_eval:
            # Normalization logic: Only for word freq + Euclidean; skip for cosine or embeddings
            do_normalize = (('vectorizer' in clf_method) and
                            (metric_name == 'euclidean_distances') and
                            clf_method.get('normalize_vectors', False))

            lib_avgs_norm = library_avgs.copy()
            if do_normalize:
                for key, avg_vec in lib_avgs_norm.items():
                    lib_avgs_norm[key] = normalize(avg_vec.reshape(1, -1), norm='l2')[0]

            metric_func = get_metric_func(metric_name)
            pred_start = time.time()
            # Predict: Now returns {unknown_model: {prompt: [top_k preds]}}
            preds = predict_unknown(val_unknown, lib_avgs_norm, metric_func,
                                    top_k=3, do_normalize=do_normalize)
            pred_time_total += time.time() - pred_start

            metrics_start = time.time()
            metrics = compute_metrics(preds)  # Updated to handle nested preds
            metrics_time_total += time.time() - metrics_start

            fold_metrics_dict.setdefault(metric_name, []).append(metrics)

        fold_time = time.time() - fold_start
        fold_times.append({
            'fold': fold_num + 1,
            'processing_time': [embedding_preprocessing_time, processing_time],
            'avg_computation_time': avg_time,
            'prediction_time': pred_time_total,
            'metrics_time': metrics_time_total,
            'total_fold_time': fold_time
        })

    # Average metrics across folds for each evaluated metric
    avg_results = {}
    for metric_name in metrics_to_eval:
        fold_metrics = fold_metrics_dict[metric_name]
        avg_metrics = {k: np.mean([fold.get(k, 0) for fold in fold_metrics]) for k in
                       fold_metrics[0]}

        avg_results[
            f"{clf_method['name']}_{metric_name.split('_')[0]}"] = avg_metrics  # e.g., name_cosine, name_euclidean

    # Compute average times
    avg_times = {
        'avg_processing_time': np.mean([ft['processing_time'] for ft in fold_times]),
        'avg_avg_computation_time': np.mean([ft['avg_computation_time'] for ft in fold_times]),
        'avg_prediction_time': np.mean([ft['prediction_time'] for ft in fold_times]),
        'avg_metrics_time': np.mean([ft['metrics_time'] for ft in fold_times]),
        'avg_fold_time': np.mean([ft['total_fold_time'] for ft in fold_times]),
        'total_cv_time': time.time() - start_total
    }
    avg_results['times'] = avg_times

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Write all results
    with open(output_file, 'a') as f:
        f.write(json.dumps(avg_results) + '\n')

    return avg_results