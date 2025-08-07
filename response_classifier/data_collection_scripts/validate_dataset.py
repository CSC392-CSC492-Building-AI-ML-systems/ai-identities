import os
import json
import argparse
import yaml
from typing import Callable
from collections import defaultdict
from collect_data import generate_llm_temperatures


def generate_temperature_list(bins_config: dict[str, dict]) -> list[float]:
    """
    Generate sorted temperature list from the temperature bin configuration.
    """
    low_temps = generate_llm_temperatures(bins_config["low"]["min"],
                                          bins_config["low"]["max"],
                                          bins_config["low"]["num_points"])
    med_temps = generate_llm_temperatures(bins_config["medium"]["min"],
                                          bins_config["medium"]["max"],
                                          bins_config["medium"]["num_points"])
    high_temps = generate_llm_temperatures(bins_config["high"]["min"],
                                           bins_config["high"]["max"],
                                           bins_config["high"]["num_points"])

    return low_temps + med_temps + high_temps


def load_reference_set(file_path: str) -> set:
    """
    Load reference set (e.g., set of user prompts, set of system prompts, set of
    neutralization techniques) from JSON file.
    """
    with open(file_path, 'r') as f:
        return set(json.load(f))


def get_dataset_config(dataset_type: str) -> tuple[Callable, Callable]:
    """
    Get grouping and description functions based on dataset type.
    """
    if dataset_type == 'base':
        group_key_func = lambda item: item['prompt']
        group_desc_func = lambda key: f"Prompt: {key[:35]}..."
    else:  # 'system_prompt'
        group_key_func = lambda item: (
            item['prompt'],
            item['system_prompt'],
            item['neutralization_technique']
        )
        group_desc_func = lambda key: (
            f"Prompt: {key[0][:20]}... | "
            f"System: {key[1][:10]}... | "
            f"Technique: {key[2]}"
        )
    return group_key_func, group_desc_func


def load_all_reference_sets(config_dir: str, dataset_type: str) -> dict[str, set]:
    """
    Load all required reference sets (for the dataset type) from JSON files.
    """
    sets = {
        'user_prompts': load_reference_set(os.path.join(config_dir, 'user_prompt_set.json'))
    }
    if dataset_type == 'system_prompt':
        sets['system_prompts'] = load_reference_set(os.path.join(config_dir, 'system_prompt_set.json'))
        sets['techniques'] = load_reference_set(os.path.join(config_dir, 'neutralization_techniques.json'))
    return sets


def process_model_data(file_path: str, group_key_func: Callable, dataset_type: str) -> dict:
    """
    Process a single JSON data file (contains LLM response data for a single LLM),
    collecting stats and values.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)

    model_name = os.path.basename(file_path)[:-5].replace('_', '/', 1)
    group_temperatures = defaultdict(list)
    group_counts = defaultdict(lambda: {'string': 0, 'none': 0, 'empty': 0})
    errors = set()
    found_sets = {
        'user_prompts': set(),
        'system_prompts': set(),
        'techniques': set(),
    }

    for item in data:
        key = group_key_func(item)
        group_temperatures[key].append(item['temperature'])
        found_sets['user_prompts'].add(item['prompt'])

        if item['response'] is None:
            group_counts[key]['none'] += 1
        elif item['response'] == "":
            group_counts[key]['empty'] += 1
        elif isinstance(item['response'], str):
            group_counts[key]['string'] += 1

        if item['error'] is not None:
            errors.add(item['error'])

        if dataset_type == 'system_prompt':
            found_sets['system_prompts'].add(item['system_prompt'])
            found_sets['techniques'].add(item['neutralization_technique'])

    return {
        "model_name": model_name,
        "group_temperatures": group_temperatures,
        "group_counts": group_counts,
        "errors": errors,
        "found_sets": found_sets,
    }


def generate_set_validation_report(found: dict, ref: dict, dataset_type: str) -> str:
    """
    Generate the report section for validating found items against reference sets.
    """
    report = "Reference Set Validation:\n"

    # User Prompts Validation
    match = (found['user_prompts'] == ref['user_prompts'])
    report += f"- User prompts match reference: {match}\n"
    if not match:
        missing = sorted(ref['user_prompts'] - found['user_prompts'])
        extra = sorted(found['user_prompts'] - ref['user_prompts'])
        report += f"  Missing prompts: {len(missing)}\n"
        if missing: report += f"    Examples: {', '.join(p[:20] + '...' for p in missing[:3])}\n"
        report += f"  Extra prompts: {len(extra)}\n"
        if extra: report += f"    Examples: {', '.join(e[:20] + '...' for e in extra[:3])}\n"

    if dataset_type == 'system_prompt':
        # System Prompts Validation
        match = (found['system_prompts'] == ref['system_prompts'])
        report += f"- System prompts match reference: {match}\n"
        if not match:
            missing = sorted(ref['system_prompts'] - found['system_prompts'])
            extra = sorted(found['system_prompts'] - ref['system_prompts'])
            report += f"  Missing system prompts: {len(missing)}\n"
            if missing: report += f"    Examples: {', '.join(p[:20] + '...' for p in missing[:3])}\n"
            report += f"  Extra system prompts: {len(extra)}\n"
            if extra: report += f"    Examples: {', '.join(e[:20] + '...' for e in extra[:3])}\n"

        # Neutralization Techniques Validation
        match = (found['techniques'] == ref['techniques'])
        report += f"- Neutralization techniques match reference: {match}\n"
        if not match:
            missing = sorted(ref['techniques'] - found['techniques'])
            extra = sorted(found['techniques'] - ref['techniques'])
            report += f"  Missing techniques: {len(missing)}\n"
            if missing: report += f"    Examples: {', '.join(missing[:3])}\n"
            report += f"  Extra techniques: {len(extra)}\n"
            if extra: report += f"    Examples: {', '.join(extra[:3])}\n"

    return report


def generate_group_completeness_report(group_counts: dict, ref_sets: dict,
                                       group_desc_func: Callable) -> str:
    """
    Generate the report for group combination completeness (for system_prompt dataset).
    """
    expected_groups = {
        (user_prompt, system_prompt, neutralization_technique)
        for user_prompt in ref_sets['user_prompts']
        for system_prompt in ref_sets['system_prompts']
        for neutralization_technique in ref_sets['techniques']
    }
    found_groups = set(group_counts.keys())
    match = (found_groups == expected_groups)

    report = f"- Group combinations complete: {match}\n"
    if not match:
        missing = sorted(list(expected_groups - found_groups), key=str)
        extra = sorted(list(found_groups - expected_groups), key=str)
        report += f"  Missing groups: {len(missing)}\n"
        if missing:
            report += "    Examples:\n"
            for group in missing[:3]: report += f"      - {group_desc_func(group)}\n"
        report += f"  Extra groups: {len(extra)}\n"
        if extra:
            report += "    Examples:\n"
            for group in extra[:3]: report += f"      - {group_desc_func(group)}\n"

    return report


def generate_group_analysis_report(processed_data: dict, group_desc_func: Callable,
                                   temp_list: list) -> tuple[str, dict]:
    """Generate the report for group-level analysis and calculate response totals."""
    report = "Group Analysis:\n"
    totals = defaultdict(int)

    sorted_keys = sorted(processed_data['group_counts'].keys(), key=str)
    expected_temps_count = len(temp_list)

    for key in sorted_keys:
        counts = processed_data['group_counts'][key]
        string_count, none_count, empty_count = counts['string'], counts['none'], counts['empty']
        collected_temps_count = len(processed_data['group_temperatures'][key])

        if (none_count + empty_count) > 0 or (
                collected_temps_count != expected_temps_count):
            report += f"- {group_desc_func(key)}\n"
            report += f"  String responses: {string_count}\n"
            report += f"  None responses: {none_count}\n"
            report += f"  Empty responses: {empty_count}\n"
            report += f"  Temperatures collected: {collected_temps_count} (expected: {expected_temps_count})\n"

        totals['string'] += string_count
        totals['none'] += none_count
        totals['empty'] += empty_count

    return report, totals


def create_sanity_check_report(directory: str, output_file: str, temp_list: list,
                               dataset_type: str):
    """
    Orchestrates the validation process and generates the final sanity check report.
    This function replaces the original monolithic `validate_dataset` function.
    """
    json_files = sorted([f for f in os.listdir(directory) if f.endswith('.json')])
    full_report_content = []
    sorted_temp_list = sorted(temp_list)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(script_dir, '../configs')

    group_key_func, group_desc_func = get_dataset_config(dataset_type)
    reference_sets = load_all_reference_sets(config_dir, dataset_type)

    for filename in json_files:
        file_path = os.path.join(directory, filename)

        # 1. Process data from a single model file
        processed_data = process_model_data(file_path, group_key_func, dataset_type)

        # 2. Build the report for this model section
        report_section = f"====================\n"
        report_section += f"Model: {processed_data['model_name']}\n"
        report_section += f"====================\n\n"

        # Overall stats
        num_groups = len(processed_data['group_counts'])
        all_temps_complete = all(
            sorted(temps) == sorted_temp_list for temps in
            processed_data['group_temperatures'].values()
        )
        if dataset_type == 'base':
            report_section += f"Number of unique user prompts: {num_groups}\n"
        else:
            report_section += f"Number of unique (user prompt, system prompt, neutralization technique) combinations: {num_groups}\n"
        report_section += f"All groups have complete temperatures: {all_temps_complete}\n\n"

        # Reference set validation report
        report_section += generate_set_validation_report(processed_data['found_sets'],
                                                         reference_sets, dataset_type)

        # Group combination completeness report (system prompt dataset only)
        if dataset_type == 'system_prompt':
            report_section += generate_group_completeness_report(
                processed_data['group_counts'], reference_sets, group_desc_func
            )

        # Detailed analysis of problematic groups
        analysis_report, totals = generate_group_analysis_report(processed_data,
                                                                 group_desc_func,
                                                                 sorted_temp_list)
        report_section += f"\n{analysis_report}\n"

        # Final summary for the model
        report_section += f"Total String Responses: {totals['string']}\n"
        report_section += f"Total None Responses: {totals['none']}\n"
        report_section += f"Total Empty Responses: {totals['empty']}\n"
        report_section += f"Total Errors: {len(processed_data['errors'])}\n"
        report_section += "Unique Errors:\n"
        report_section += "\n".join(
            f"- {error}" for error in processed_data['errors']) or "- None"
        report_section += "\n\n"

        full_report_content.append(report_section)

    with open(output_file, 'w') as f:
        f.write('\n'.join(full_report_content))


def main():
    """
    Main function to parse arguments and run validation.
    """
    parser = argparse.ArgumentParser(description='Validate LLM response datasets')
    parser.add_argument('--dataset-type', choices=['base', 'system_prompt'],
                        required=True, help='Type of dataset to validate (base or system_prompt)')
    args = parser.parse_args()

    # Load configuration
    with open('../configs/data_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Configure dataset parameters
    if args.dataset_type == 'base':
        directory = '../data/base_dataset_raw_data/'
        bins_config = config['temp_bins']
        output_file = '../data/base_dataset_sanity_report.txt'
    else:  # system_prompt
        directory = '../data/system_prompt_dataset_raw_data/'
        bins_config = config['system_prompt_dataset_temp_bins']
        output_file = '../data/system_prompt_dataset_sanity_report.txt'

    # Generate temperature list
    temp_list = generate_temperature_list(bins_config)

    # Run validation
    create_sanity_check_report(directory=directory, output_file=output_file,
                               temp_list=temp_list, dataset_type=args.dataset_type)
    print(f"Validation complete. Report saved to {output_file}")


if __name__ == '__main__':
    main()