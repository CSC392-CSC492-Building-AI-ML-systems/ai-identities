import os
import json
from collections import defaultdict


directory = './model_selection_dataset/raw_data/'
output_file = './model_selection_dataset/raw_data_sanity_check_report.txt'
temp_list = [0.0, 0.03, 0.07, 0.1, 0.13, 0.17, 0.2, 0.23, 0.27, 0.3, 0.33, 0.37, 0.4,
             0.42, 0.43, 0.45, 0.47, 0.48, 0.5, 0.52, 0.53, 0.55, 0.57, 0.58, 0.6, 0.62,
             0.63, 0.65, 0.67, 0.68, 0.7, 0.72, 0.73, 0.75, 0.77, 0.78, 0.8, 0.81, 0.82,
             0.83, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93,
             0.93, 0.94, 0.95, 0.96, 0.97, 0.97, 0.98, 0.99, 1.0, 1.01, 1.02, 1.02, 1.03,
             1.04, 1.05, 1.06, 1.07, 1.07, 1.08, 1.09, 1.1, 1.11, 1.12, 1.12, 1.13, 1.14,
             1.15, 1.16, 1.17, 1.17, 1.18, 1.19]

output_content = []
json_files = sorted([f for f in os.listdir(directory) if f.endswith('.json')])

for filename in json_files:
    file_path = os.path.join(directory, filename)
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Retrieve model name: remove .json and replace the first '-' with '/'
    model_name = filename[:-5].replace('_', '/', 1)
    unique_prompts = set(item['prompt'] for item in data)

    # Group temperatures by prompt
    prompt_temperatures = defaultdict(list)
    for item in data:
        prompt_temperatures[item['prompt']].append(item['temperature'])

    matches_temp_list = all(sorted(temps) == sorted(temp_list) for temps in prompt_temperatures.values())

    # Count per prompt: string responses and None responses
    prompt_counts = defaultdict(lambda: {'string': 0, 'none': 0})
    for item in data:
        prompt = item['prompt']
        response = item['response']
        if isinstance(response, str):
            prompt_counts[prompt]['string'] += 1
        elif response is None:
            prompt_counts[prompt]['none'] += 1

    # Collect instances where 'error' is not `None`
    errors = set(item['error'] for item in data if item['error'] is not None)
    num_errors = sum(1 for item in data if item['error'] is not None)

    # Format the output for this model
    section = f"====================\n"
    section += f"Model: {model_name}\n"
    section += f"====================\n\n"
    section += f"Number of unique prompts: {len(unique_prompts)}\n"
    section += f"Temperatures match predefined temp_list: {matches_temp_list}\n\n"
    section += f"Prompt Analysis:\n"
    for prompt in sorted(prompt_counts.keys()):
        counts = prompt_counts[prompt]
        section += f"- Prompt: {prompt[:35]}\n"
        section += f"  String responses: {counts['string']}\n"
        section += f"  None responses: {counts['none']}\n"
    section += "\n"
    section += f"Total Number of Errors: {num_errors}\n"
    section += f"Unique Errors:\n"
    if errors:
        for error in errors:
            section += f"- {error}\n"
    else:
        section += "- None\n"
    section += "\n"

    output_content.append(section)

# Write all content to the output file
with open(output_file, 'w') as f:
    f.write('\n'.join(output_content))