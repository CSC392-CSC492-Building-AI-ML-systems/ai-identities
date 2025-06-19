import sys
import os
import json
import re


def load_json(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def get_top_key(hist_dict):
    if not hist_dict:
        return set()
    max_val = max(hist_dict.values())
    return {k for k, v in hist_dict.items() if v == max_val}


def extract_temperature(filename):
    """Extract temperature from filename assuming it ends like *_results_<temp>.json"""
    match = re.search(r'_results_([0-9.]+)\.json$', filename)
    if match:
        return float(match.group(1))
    return None


def make_report_filename(target_fname: str) -> str:
    """
    gemma-3-4b-it_results_0.0.json → gemma-3-4b-it_report_0.0.txt
    """
    stem = os.path.splitext(target_fname)[0]  # remove .json
    stem = stem.replace("results", "report")
    return f"{stem}.txt"


def compare_jsons(target_file, folder_path):
    """
    Compare the `target_file` against every other result json file in `folder_path`
    whose temperature is 0.0 while excluding the `target_file` itself.

    :param target_file: a json file containing the responses of an LLM model for many prompts
    :param folder_path: the path to the folder which contains all the result json files
    :return: multi-line string with the report
    """
    target_data = load_json(target_file)
    target_filename = os.path.basename(target_file)

    # List all JSON files in the folder with temp == 0, excluding the target file
    zero_temp_json_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(".json")
        and f != target_filename
        and extract_temperature(f) == 0.0
    ]

    output_lines = [f"Target: {target_filename}"]

    if not zero_temp_json_files:
        output_lines.append("No JSON file with temperature 0.0 was found.")
        return "\n".join(output_lines)

    zero_temp_files_json_lst = [load_json(f) for f in zero_temp_json_files]
    total_prompts = 0
    overlaps = [0] * len(zero_temp_json_files)

    for prompt, response_histogram_dict in target_data.items():
        target_top_keys = get_top_key(response_histogram_dict)
        total_prompts += 1

        for i, other_data in enumerate(zero_temp_files_json_lst):
            other_counts = other_data.get(prompt, {})
            other_top_keys = get_top_key(other_counts)
            # If the two sets overlap (share at least one string/response),
            # then register a match
            if target_top_keys & other_top_keys:
                overlaps[i] += 1

    output_lines.append(f"Total prompts compared: {total_prompts}")
    for i, fname in enumerate(zero_temp_json_files):
        match_rate = overlaps[i] / total_prompts * 100
        output_lines.append(f"{os.path.basename(fname)}: {overlaps[i]} matches ({match_rate:.2f}%) with target")

    return "\n".join(output_lines)


def main(results_dir, output_dir):
    """
    Generate a full comparison report for every JSON file in `results_dir`
    and write it to `results_dir/report_name`.
    """
    if not os.path.isdir(results_dir):
        sys.exit(f"Directory not found: {results_dir}")

    os.makedirs(output_dir, exist_ok=True)

    json_files = [
        f
        for f in os.listdir(results_dir)
        if f.endswith(".json")
    ]
    if not json_files:
        sys.exit("No JSON files found in the specified directory.")

    for idx, fname in enumerate(sorted(json_files), 1):
        target_path = os.path.join(results_dir, fname)
        report = compare_jsons(target_path, results_dir)

        output_path = os.path.join(output_dir, make_report_filename(fname))
        with open(output_path, "w") as output_file:
            output_file.write(report + "\n\n")

        print(f"[{idx}/{len(json_files)}] processed {os.path.basename(output_path)}")

    print(f"\nCompleted. Reports are saved to {output_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 1:
        print("Usage: python compare_all_models.py")
        sys.exit(1)

    main(results_dir="results", output_dir="model_comparison_results")