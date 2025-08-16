import yaml
import pickle
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import argparse
from data_processor import process_word_freq, process_embeddings, load_processed, load_fitted_vectorizer
from classifier_model import compute_library_averages, predict_unknown, get_metric_func
from evaluator import compute_metrics


def load_system_prompt_data(data_path: str) -> pd.DataFrame:
    """
    Loads and combines all JSON files from the system prompt dataset directory
    into a single DataFrame.
    """
    all_data = []
    if not os.path.exists(data_path) or not os.listdir(data_path):
        raise FileNotFoundError(f"Dataset directory not found or is empty: {data_path}")

    for file_name in os.listdir(data_path):
        if file_name.endswith('.json'):
            model_name = file_name.replace('.json', '')
            file_path = os.path.join(data_path, file_name)
            with open(file_path, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            df['model'] = model_name
            all_data.append(df)

    if not all_data:
        raise ValueError(f"No JSON files found in {data_path}")

    # The 'prompt' in the raw (JSON) files is the base user prompt
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df.rename(columns={'prompt': 'base_user_prompt'})


def analyze_neutralization_technique_effectiveness(results_df: pd.DataFrame) -> dict:
    """
    Analyzes the effectiveness of neutralization techniques against the system
    prompts. Performs the analysis for each classifier method as well as
    on average (across all the candidate classifier methods).

    :return: A dictionary containing two DataFrames: one for per-classifier
        analysis and one for an across-all-classifiers average.
    """
    # Per-classifier analysis
    grouped_per_clf = results_df.groupby(['classifier', 'neutralization_technique'])
    agg_per_clf = grouped_per_clf['is_correct_top1'].agg(['mean', 'var', 'count'])
    per_classifier_analysis = agg_per_clf.rename(columns={'mean': 'accuracy', 'var': 'variance'})

    # Analysis across all classifier methods
    grouped_across_clf = results_df.groupby('neutralization_technique')
    agg_across_clf = grouped_across_clf['is_correct_top1'].agg(['mean', 'var', 'count'])
    across_classifiers_analysis = agg_across_clf.rename(
        columns={'mean': 'accuracy', 'var': 'variance'}
    ).sort_values(by='accuracy', ascending=False)

    return {
        "Effectiveness per Classifier": per_classifier_analysis,
        "Average Effectiveness Across All Classifiers": across_classifiers_analysis
    }


def analyze_user_prompt_discriminativeness(results_df: pd.DataFrame) -> dict:
    """
    Analyzes how discriminative/effective base user prompts are against the
    system prompts. Performs the analysis for each classifier method as well as
    on average (across all the candidate classifier methods).

    :return: A dictionary containing two DataFrames: one for the best-performing
        classifier and one for an across-all-classifiers average.
    """
    # Per-classifier analysis
    grouped_per_clf = results_df.groupby(['classifier', 'base_user_prompt'])
    agg_per_clf = grouped_per_clf['is_correct_top1'].agg(['mean', 'var', 'count'])
    per_classifier_analysis = agg_per_clf.rename(columns={'mean': 'accuracy', 'var': 'variance'})

    # Analysis across all classifier methods
    grouped_across_clf = results_df.groupby('base_user_prompt')
    agg_across_clf = grouped_across_clf['is_correct_top1'].agg(['mean', 'var', 'count'])
    across_classifiers_analysis = agg_across_clf.rename(
        columns={'mean': 'accuracy', 'var': 'variance'}
    ).sort_values(by='accuracy', ascending=False)

    # 3. Per-technique analysis (prompt effectiveness for each neutralization technique)
    grouped_per_tech = results_df.groupby(['neutralization_technique', 'base_user_prompt'])
    agg_per_tech = grouped_per_tech['is_correct_top1'].agg(['mean', 'var', 'count'])
    per_technique_analysis = agg_per_tech.rename(columns={'mean': 'accuracy', 'var': 'variance'})

    return {
        "Prompt Discriminativeness per Classifier": per_classifier_analysis,
        "Average Prompt Discriminativeness Across All Classifiers": across_classifiers_analysis,
        "Prompt Discriminativeness per Neutralization Technique": per_technique_analysis
    }


def _format_analysis_section(title: str, analysis_dict: dict) -> str:
    """
    Helper to format a section of the text report.
    """
    report_parts = [f"--- {title} ---\n"]
    for subtitle, df in analysis_dict.items():
        report_parts.append(f"\n>> {subtitle}\n")
        report_parts.append(df.to_string())
        report_parts.append("\n")
    return "".join(report_parts)


def generate_detailed_report(output_path: str, results_df: pd.DataFrame,
                             overall_metrics: dict):
    """
    Generates and saves a detailed report to a text file to the
    '/response_classifier/results/' directory.
    """
    with open(output_path, 'w') as f:
        f.write("========= System Prompt Robustness Detailed Report =========\n\n")

        # Section 1: Overall Classifier Performance
        f.write("--- Overall Classifier Performance Summary ---\n")
        overall_df = pd.DataFrame(overall_metrics).T.sort_values(by='top1_accuracy', ascending=False)
        cols = ['top1_accuracy', 'top3_accuracy', 'f1_score', 'precision', 'recall']
        f.write(overall_df[cols].to_string(float_format='{:.4f}'.format))
        f.write("\n\n")

        # Section 2: Neutralization Technique Effectiveness
        tech_analysis = analyze_neutralization_technique_effectiveness(results_df)
        f.write(
            _format_analysis_section("Neutralization Technique Effectiveness Analysis", tech_analysis)
        )
        f.write("\n")

        # Section 3: Base User Prompt Discriminativeness
        prompt_analysis = analyze_user_prompt_discriminativeness(results_df)
        f.write(_format_analysis_section("Base User Prompt Discriminativeness Analysis", prompt_analysis)
        )
        f.write("\n")

        f.write("\n========= End of Report =========\n")


def save_plots(output_dir: str, overall_metrics: dict):
    """
    Generates and saves summary plots.
    """
    # Overall Accuracy per Classifier
    overall_df = pd.DataFrame(overall_metrics).T.sort_values(by='top1_accuracy', ascending=False)
    plt.figure(figsize=(14, 8))
    sns.barplot(x=overall_df.index, y=overall_df['top1_accuracy'])
    plt.title('Overall Top-1 Accuracy per Classifier on System Prompt Dataset')
    plt.ylabel('Top-1 Accuracy')
    plt.xlabel('Classifier Method')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'classifier_accuracy_comparison.png'))
    plt.close()


def run_evaluation(args):
    """
    Uses the train set (from the base dataset) as a library of known LLMs and
    evaluates the robustness of the candidate classifier methods on the system
    prompt dataset. Lastly, it generates a detailed report and plots.
    """
    # Load Configurations and Data
    print("Loading configurations and data...")
    methods_cfg = yaml.safe_load(open('../configs/classification_methods_config.yaml'))
    all_methods = methods_cfg['word_freq'] + methods_cfg['embeddings']

    with open('../data/splits/train.pkl', 'rb') as f:
        train_data = pickle.load(f)

    raw_dataset_sys_prompt = load_system_prompt_data('../data/system_prompt_dataset_raw_data/')
    raw_dataset_sys_prompt['prompt'] = raw_dataset_sys_prompt['base_user_prompt']
    # Convert the system prompt DataFrame into the required dictionary format
    sys_prompt_dataset_grouped = {model: group for model, group in raw_dataset_sys_prompt.groupby('model')}

    # Create Output Directory
    output_dir = '../results/'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved to {output_dir}")

    all_results = []
    overall_metrics = {}
    for clf_method in tqdm(all_methods, desc="Evaluating Classifiers on System Prompts", position=0):
        clf_method_name = clf_method['name']

        # Build Library from Training Data (base dataset)
        vectorizer = None
        try:
            # Try to load pre-processed training data
            train_processed = load_processed(clf_method_name, 'train')
            print(f"Loaded cached training data for '{clf_method_name}'.")
            if 'vectorizer' in clf_method:
                vectorizer = load_fitted_vectorizer(clf_method_name, 'train')
        except FileNotFoundError:
            # If not found, process and save it for next time
            print(f"No cached data found for '{clf_method_name}'. Processing and saving...")
            train_output_path = f"../data/processed/{clf_method_name}/train/"
            if 'vectorizer' in clf_method:
                train_processed, vectorizer = process_word_freq(
                    train_data, clf_method, output_path=train_output_path
                )
            else:
                train_processed = process_embeddings(
                    train_data, clf_method, output_path=train_output_path
                )

        if 'vectorizer' in clf_method:
            test_processed, _ = process_word_freq(sys_prompt_dataset_grouped, clf_method,
                                                  output_path='', fitted_vectorizer=vectorizer)
        else:
            test_processed = process_embeddings(sys_prompt_dataset_grouped, clf_method, output_path='')

        library_avgs = compute_library_averages(train_processed)

        metrics_to_eval = [clf_method['metric']]
        if args.eval_both_metrics:
            metrics_to_eval = ['cosine_similarity', 'euclidean_distances']

        for metric_name in tqdm(metrics_to_eval, desc="Metrics", leave=False, position=1):
            classifier_id = f"{clf_method_name}_{metric_name.split('_')[0]}"
            metric_func = get_metric_func(metric_name)
            predictions = predict_unknown(test_processed, library_avgs, metric_func, top_k=3)
            overall_metrics[classifier_id] = compute_metrics(predictions)
            total_examples = sum(len(combo_preds) for combo_preds in predictions.values())
            with tqdm(total=total_examples, desc=f"Results", leave=False, position=2,
                      mininterval=0.5, smoothing=0.05) as ex_pbar:
                for true_model, combo_preds in predictions.items():
                    for combo, pred_list in combo_preds.items():
                        prompt_text, sys_prompt, technique = combo
                        original_rows = raw_dataset_sys_prompt[
                            (raw_dataset_sys_prompt['model'] == true_model) &
                            (raw_dataset_sys_prompt['base_user_prompt'] == prompt_text) &
                            (raw_dataset_sys_prompt['system_prompt'] == sys_prompt) &
                            (raw_dataset_sys_prompt['neutralization_technique'] == technique)
                        ]
                        for _, original_row in original_rows.iterrows():
                            all_results.append({
                                'classifier': classifier_id,
                                'true_model': true_model,
                                'base_user_prompt': original_row['base_user_prompt'],
                                'system_prompt': original_row.get('system_prompt', sys_prompt),
                                'neutralization_technique': original_row.get(
                                    'neutralization_technique', technique),
                                'top1_prediction': pred_list[0] if pred_list else None,
                                'top3_predictions': pred_list,
                            })
                        ex_pbar.update(1)

    # Analyze Results and Generate Reports
    print("Analyzing results and generating reports...")
    results_df = pd.DataFrame(all_results)
    if not results_df.empty:
        results_df['is_correct_top1'] = results_df['true_model'] == results_df['top1_prediction']

    generate_detailed_report(
        output_path=os.path.join(output_dir, 'system_prompt_experiment_report.txt'),
        results_df=results_df,
        overall_metrics=overall_metrics
    )

    save_plots(output_dir=output_dir, overall_metrics=overall_metrics)

    print(f"\nEvaluation on the system prompt dataset complete. Detailed report saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Run system prompt robustness evaluation on all classifiers."
    )
    parser.add_argument(
        '--eval-both-metrics',
        action='store_true',
        help="If set, evaluates each method with both cosine similarity and Euclidean distance."
    )
    args = parser.parse_args()
    run_evaluation(args)


if __name__ == '__main__':
    main()