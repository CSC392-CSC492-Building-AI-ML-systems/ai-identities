import json
import os
import pandas as pd
import re


def remove_cot_block(response: str, is_special_model: bool = False) -> str:
    """
    Robustly remove any <think>...</think> blocks from a response string, handling
    all positions (start/mid/end), nesting, and malformed/unpaired cases.
    Preserves content in incomplete/malformed cases to avoid data loss.

    :param response: Raw response string.
    :param is_special_model: If True, treat as model that skips <think> but uses </think>.
    :return: Cleaned response without CoT (never empty unless input was).
    """
    if not response:
        return response

    original = response

    # Iterative removal to handle nesting and multiples
    while True:
        # Find and remove complete paired blocks (non-greedy, anywhere)
        new_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        if new_response == response:
            break  # No more changes
        response = new_response

    # Handle unpaired/malformed after paired removal
    # Unpaired <think>: Remove tag, keep content after
    response = re.sub(r'<think>', '', response, flags=re.DOTALL)

    # Unpaired </think>: Remove tag, keep content before/after
    response = re.sub(r'</think>', '', response, flags=re.DOTALL)

    # Special model handling (assumes no <think>, but </think> closes initial CoT)
    if is_special_model:
        # If no </think>, keep all (incomplete)
        if '</think>' not in original:
            response = original  # Override to preserve

    response = response.strip()

    # Fallback to original if somehow emptied (rare)
    if not response:
        return original

    return response


def load_raw_data(raw_path: str, remove_cot: bool = True) -> tuple:
    """
    Load all JSON files into a dict of Pandas DataFrames, where the model names
    are the keys and DataFrames are the values. Optionally removes CoT blocks.
    """
    data = {}
    model_stats = {}  # Track per model
    special_models = set()  # Auto-detected special models
    total_responses = 0

    # First pass: Detect special models (never use <think>, but use </think> at least once)
    for file in os.listdir(raw_path):
        if file.endswith('.json'):
            model_name = file.replace('.json', '')
            with open(os.path.join(raw_path, file), 'r') as f:
                raw_list = json.load(f)

            uses_think = any('<think>' in entry.get('response', '') for entry in raw_list)
            uses_close = any('</think>' in entry.get('response', '') for entry in raw_list)
            if not uses_think and uses_close:
                special_models.add(model_name)

    # Second pass: Load and process
    for file in os.listdir(raw_path):
        if file.endswith('.json'):
            model_name = file.replace('.json', '')
            with open(os.path.join(raw_path, file), 'r') as f:
                raw_list = json.load(f)

            is_special = model_name in special_models
            incomplete_open_count = 0
            dangling_close_count = 0
            malformed_count = 0
            empty_after_removal = 0

            for entry in raw_list:
                total_responses += 1
                original = entry.get('response', '')
                if remove_cot:
                    has_think = '<think>' in original
                    has_close = '</think>' in original

                    # Detect malformed (e.g., close before open, or repeated)
                    if has_close and not has_think:
                        dangling_close_count += 1
                    if has_think and not has_close:
                        incomplete_open_count += 1
                    if re.search(r'</think>.*?<think>', original) or re.search(
                            r'<think>.*?<think>', original) or re.search(
                            r'</think>.*?</think>', original):
                        malformed_count += 1

                    cleaned = remove_cot_block(original, is_special)
                    entry['response'] = cleaned

                    if not cleaned:
                        empty_after_removal += 1

            model_stats[model_name] = {
                'is_special': is_special,
                'incomplete_open': incomplete_open_count,
                'dangling_close': dangling_close_count,
                'malformed': malformed_count,
                'empty_after_removal': empty_after_removal
            }

            columns_to_keep = ['prompt', 'response', 'temperature']
            if raw_list and 'system_prompt' in raw_list[0]:
                columns_to_keep.extend(['system_prompt', 'neutralization_technique'])

            df = pd.DataFrame(raw_list)[columns_to_keep]
            df['model'] = model_name
            data[model_name] = df

    return data, model_stats, special_models, total_responses


def log_cot_analysis(data, model_stats, special_models, total_responses):
    analysis_file = '../data/cot_analysis.txt'
    with open(analysis_file, 'w') as f:
        f.write(f"Processed {total_responses} responses across {len(data)} models.\n")
        f.write(
            f"Detected special models (no <think>, but use </think>): {list(special_models)}\n")
        f.write("CoT edge cases per model (out of ~1680):\n")
        for model, stats in model_stats.items():
            f.write(f"  - {model} (special: {stats['is_special']}):\n")
            f.write(
                f"    Incomplete opens (has <think> no close): {stats['incomplete_open']}\n")
            f.write(
                f"    Dangling closes (has </think> no open): {stats['dangling_close']}\n")
            f.write(f"    Malformed/nested: {stats['malformed']}\n")
            f.write(f"    Empty after removal: {stats['empty_after_removal']}\n")
        total_incomplete_open = sum(s['incomplete_open'] for s in model_stats.values())
        total_dangling = sum(s['dangling_close'] for s in model_stats.values())
        total_malformed = sum(s['malformed'] for s in model_stats.values())
        total_empty = sum(s['empty_after_removal'] for s in model_stats.values())
        f.write(
            f"Totals: Incomplete opens: {total_incomplete_open}, Dangling closes: {total_dangling}, Malformed: {total_malformed}, Empties: {total_empty} ({total_empty / total_responses * 100:.2f}%)\n")
    print(f"CoT analysis saved to {analysis_file}")


if __name__ == '__main__':
    data, model_stats, special_models, total_responses = load_raw_data(
        '../data/base_dataset_raw_data/')
    log_cot_analysis(data, model_stats, special_models, total_responses)