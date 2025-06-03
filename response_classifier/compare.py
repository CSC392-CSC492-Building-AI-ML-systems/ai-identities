import sys
import os
import json
import re

def load_json(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

def get_top_key(d):
    if not d:
        return set()
    max_val = max(d.values())
    return {k for k, v in d.items() if v == max_val}

def extract_temperature(filename):
    """Extract temperature from filename assuming it ends like *_results_<temp>.json"""
    match = re.search(r'_results_([0-9.]+)\.json$', filename)
    if match:
        return float(match.group(1))
    return None

def compare_jsons(target_file, folder_path):
    target_data = load_json(target_file)
    target_filename = os.path.basename(target_file)

    # List all JSON files in the folder with temp == 0, excluding the target file
    all_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(".json")
        and f != target_filename
        and extract_temperature(f) == 0.0
    ]

    if not all_files:
        print("No JSON files with temperature 0 found in the folder.")
        return

    other_data_list = [load_json(f) for f in all_files]

    total_prompts = 0
    overlaps = [0] * len(all_files)

    for prompt, target_counts in target_data.items():
        target_top_keys = get_top_key(target_counts)
        total_prompts += 1

        for i, other_data in enumerate(other_data_list):
            other_counts = other_data.get(prompt, {})
            other_top_keys = get_top_key(other_counts)
            if target_top_keys & other_top_keys:
                overlaps[i] += 1

    print(f"\nTotal prompts compared: {total_prompts}")
    for i, fname in enumerate(all_files):
        match_rate = overlaps[i] / total_prompts * 100
        print(f"{os.path.basename(fname)}: {overlaps[i]} matches ({match_rate:.2f}%) with target")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare.py path/to/target.json path/to/folder")
        sys.exit(1)

    target_json = sys.argv[1]
    folder_path = sys.argv[2]

    if not os.path.isfile(target_json):
        print(f"Target JSON file not found: {target_json}")
        sys.exit(1)

    if not os.path.isdir(folder_path):
        print(f"Invalid folder path: {folder_path}")
        sys.exit(1)

    compare_jsons(target_json, folder_path)