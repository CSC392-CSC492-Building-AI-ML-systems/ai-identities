import yaml
import pickle
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Assuming these files exist and are correctly implemented
from data_processor import process_word_freq, process_embeddings
from classifier_model import compute_library_averages, predict_unknown, get_metric_func


def run_evaluation():
    """
    Trains all candidate classifiers on Dataset A and evaluates their robustness
    on Dataset B, generating a detailed report.
    """
    # --- 1. Load Configurations and Data ---
    methods_cfg = yaml.safe_load(open('configs/classification_methods_config.yaml'))
    all_methods = methods_cfg['word_freq'] + methods_cfg['embeddings']

    # Load training data (Dataset A)
    with open('data/splits/train.pkl', 'rb') as f:
        train_data = pickle.load(f)

    # Load raw system prompt data (Dataset B)
    # This should be a list of dicts: [{'base_user_prompt': ..., 'system_prompt': ..., 'response': ..., 'model': ...}, ...]
    # You would generate this file using the data collection scripts.
    try:
        raw_dataset_b = pd.read_json(
            'data/system_prompt_raw_data/system_prompt_responses.json')
    except FileNotFoundError:
        print(
            "Error: Dataset B not found. Please run the collection script for system prompts first.")
        return

    # --- 2. Create Output Directory ---
    output_dir = 'results/robustness_report'
    os.makedirs(output_dir, exist_ok=True)

    # --- 3. Iterate Through Each Classifier Method ---
    all_results = []
    for method in tqdm(all_methods, desc="Evaluating Classifiers"):
        method_name = method['name']

        # --- A. Build Library from Training Data (Dataset A) ---
        if 'vectorizer' in method:  # Word Frequency
            train_processed, vectorizer = process_word_freq(train_data, method,
                                                            output_path='')
        else:  # Embeddings
            train_processed = process_embeddings(train_data, method, output_path='')

        library_avgs = compute_library_averages(train_processed)

        # --- B. Process and Predict on Test Data (Dataset B) ---
        # Group Dataset B by model to match the expected input format for prediction
        dataset_b_grouped = {model: group for model, group in
                             raw_dataset_b.groupby('model')}

        if 'vectorizer' in method:
            test_processed, _ = process_word_freq(dataset_b_grouped, method,
                                                  output_path='',
                                                  fitted_vectorizer=vectorizer)
        else:
            test_processed = process_embeddings(dataset_b_grouped, method, output_path='')

        predictions = predict_unknown(test_processed, library_avgs,
                                      get_metric_func(method['metric']), top_k=3)

        # --- C. Format and Store Results ---
        for true_model, prompt_preds in predictions.items():
            for prompt_identifier, pred_list in prompt_preds.items():
                # We need to link this prediction back to the original prompt info
                # This assumes 'prompt' in the processed data is a unique identifier or the text itself
                original_row = raw_dataset_b[
                    (raw_dataset_b['model'] == true_model) &
                    (raw_dataset_b['base_user_prompt'] == prompt_identifier)
                    ].iloc[0]

                all_results.append({
                    'classifier': method_name,
                    'true_model': true_model,
                    'base_prompt': original_row['base_user_prompt'],
                    'system_prompt': original_row['system_prompt'],
                    'neutralization_technique': original_row.get(
                        'neutralization_technique', 'none'),
                    'top1_prediction': pred_list[0] if pred_list else None,
                    'top3_predictions': pred_list,
                })

    # --- 4. Analyze and Save Full Results ---
    results_df = pd.DataFrame(all_results)
    results_df['is_correct_top1'] = results_df['true_model'] == results_df[
        'top1_prediction']
    results_df.to_csv(os.path.join(output_dir, 'detailed_robustness_results.csv'),
                      index=False)

    # --- 5. Generate and Save Summary Reports ---
    with open(os.path.join(output_dir, 'summary_report.txt'), 'w') as f:
        # Overall Classifier Performance
        f.write("--- Overall Classifier Performance ---\n")
        overall_acc = results_df.groupby('classifier')[
            'is_correct_top1'].mean().sort_values(ascending=False)
        f.write(overall_acc.to_string() + "\n\n")

        # Neutralization Technique Effectiveness (for the best classifier)
        best_classifier = overall_acc.index[0]
        f.write(
            f"--- Neutralization Technique Effectiveness (for best classifier: {best_classifier}) ---\n")
        tech_acc = results_df[results_df['classifier'] == best_classifier].groupby(
            'neutralization_technique')['is_correct_top1'].mean().sort_values(
            ascending=False)
        f.write(tech_acc.to_string() + "\n\n")

        # Base Prompt Discriminativeness (for the best classifier)
        f.write(
            f"--- Base Prompt Discriminativeness (for best classifier: {best_classifier}) ---\n")
        prompt_acc = \
        results_df[results_df['classifier'] == best_classifier].groupby('base_prompt')[
            'is_correct_top1'].mean().sort_values(ascending=False)
        f.write(prompt_acc.to_string() + "\n")

    # --- 6. Generate and Save Plots ---
    # Plot 1: Overall Accuracy per Classifier
    plt.figure(figsize=(12, 7))
    sns.barplot(x=overall_acc.index, y=overall_acc.values)
    plt.title('Overall Top-1 Accuracy per Classifier on Dataset B')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'classifier_accuracy_comparison.png'))
    plt.close()

    # Plot 2: Effectiveness of Neutralization Techniques
    plt.figure(figsize=(12, 7))
    sns.barplot(x=tech_acc.index, y=tech_acc.values)
    plt.title(
        f'Effectiveness of Neutralization Techniques (Classifier: {best_classifier})')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'neutralization_effectiveness.png'))
    plt.close()

    print(f"Robustness evaluation complete. Report saved to {output_dir}")


if __name__ == '__main__':
    run_evaluation()

