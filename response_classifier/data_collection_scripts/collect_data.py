import json
import time
import os
import argparse
import concurrent.futures
import uuid
import threading
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from neutralization import apply_neutralization_technique


def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Collect LLM responses from Deepinfra")
    # File path arguments
    parser.add_argument('--llm-file', type=str, required=True,
                        help="Path to the JSON file containing model names and metadata.")
    parser.add_argument('--prompt-file', type=str, required=True,
                        help="Path to the JSON file containing user prompts.")
    parser.add_argument('--output-dir', type=str, required=True,
                        help="Directory to save the output JSON files.")
    parser.add_argument('--system-prompt-file', type=str, default=None,
                        help="Optional: Path to a JSON file with system prompts. If provided, collection will run for each system prompt.")
    parser.add_argument('--neutralization-techniques-file', type=str, default=None,
                        help="Optional: Path to the JSON file containing neutralization techniques.")

    # Data collection control arguments
    parser.add_argument('--llm', type=str,
                        help="Single LLM name to process from the LLM file (e.g., Qwen/Qwen3-235B-A22B).")
    parser.add_argument('--all', action='store_true',
                        help="Process all LLMs from the LLM file where data_collected is false")
    parser.add_argument('--recollect-missing-data', type=str,
                        help="Recollect only missing data points (where response is null) for specified LLM. Provide an LLM name.")
    parser.add_argument('--prevent-cache', action='store_true',
                        help="Add unique ID to prompts to prevent potential caching")
    parser.add_argument('--force', action='store_true',
                        help="Force collection even if data exists or data_collected is true")
    parser.add_argument('--max-workers', type=int, default=75,
                        help="Max concurrent threads (default: 75; adjust for rate limits)")

    # Data collection parameter arguments
    parser.add_argument('--max-tokens', type=int, default=3500,
                        help="Max tokens for responses (default: 3500)")
    parser.add_argument('--temperature', type=float, default=0.7,
                        help="Temperature for generation when data_point_num=1 (default: 0.7)")
    parser.add_argument('--low-bin-num', type=int, default=0,
                        help="Number of data points in low temp bin [0.0, 0.4) (default: 0)")
    parser.add_argument('--med-bin-num', type=int, default=0,
                        help="Number of data points in med temp bin [0.4, 0.8) (default: 0)")
    parser.add_argument('--high-bin-num', type=int, default=0,
                        help="Number of data points in high temp bin [0.8, 1.2) (default: 0)")

    return parser.parse_args()


def load_environment():
    """
    Load environment variables and initialize OpenAI client.
    """
    load_dotenv()
    api_key = os.getenv('DEEPINFRA_API_KEY')
    if not api_key:
        raise ValueError("DEEPINFRA_API_KEY not found in .env file")
    return OpenAI(api_key=api_key, base_url="https://api.deepinfra.com/v1/openai")


def load_json_file(file_path):
    """
    Load JSON from a file, with error handling.
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in file: {file_path}")


def generate_llm_temperatures(low, high, num) -> list[float]:
    """
    Generate evenly spaced temperatures for a bin.
    """
    if num == 0: return []
    if num == 1: return [low]
    step = (high - low) / num
    temps = [round(low + i * step, 2) for i in range(num)]
    return temps


def generate_temps_list(args) -> list[float]:
    """
    Generate temperature list based on data_point_num.
    """
    total_number_of_data_points = args.low_bin_num + args.med_bin_num + args.high_bin_num
    if total_number_of_data_points == 0:
        return [args.temperature]
    else:
        low_temps = generate_llm_temperatures(0.0, 0.4, args.low_bin_num)
        med_temps = generate_llm_temperatures(0.4, 0.8, args.med_bin_num)
        high_temps = generate_llm_temperatures(0.8, 1.2, args.high_bin_num)
        return low_temps + med_temps + high_temps


def process_prompt(llm_name, original_prompt, neutralized_prompt, temperature,
                   system_prompt, neutralization_technique, args, openai_client) -> dict:
    """
    Processes a single prompt at a specific temperature, optionally with a
    system prompt and neutralization technique applied to the user prompt.
    """
    final_prompt_to_send = neutralized_prompt
    if args.prevent_cache:
        unique_id = str(uuid.uuid4())
        final_prompt_to_send = f"{neutralized_prompt} [unique_id: {unique_id}]"

    messages = [{"role": "user", "content": final_prompt_to_send}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = openai_client.chat.completions.create(
                model=llm_name,
                messages=messages,
                max_tokens=args.max_tokens,
                temperature=temperature
            )
            return {
                'prompt': original_prompt,
                'modified_prompt': final_prompt_to_send,
                'response': response.choices[0].message.content.strip(),
                'temperature': temperature,
                'system_prompt': system_prompt,
                'neutralization_technique': neutralization_technique,
                'timestamp': time.time(),
                'error': None,
                'attempt': attempt + 1
            }
        except Exception as e:
            if attempt == (max_retries - 1):
                return {
                    'prompt': original_prompt,
                    'modified_prompt': final_prompt_to_send,
                    'response': None,
                    'temperature': temperature,
                    'system_prompt': system_prompt,
                    'neutralization_technique': neutralization_technique,
                    'timestamp': time.time(),
                    'error': str(e),
                    'attempt': attempt + 1
                }
            time.sleep(2 ** attempt)


def process_model(model_name, model_meta_path, args, prompts, temps_list, system_prompts,
                  neutralization_techniques, openai_client, json_lock):
    """
    Process data collection for a single model.
    """
    # Note that Deepinfra model names contain `/` so we replace it with `_`
    model_filename = model_name.replace('/', '_') + '.json'
    output_file = os.path.join(args.output_dir, model_filename)

    # Skip if data exists and --force is not set
    if os.path.exists(output_file) and not args.force:
        print(f"Skipping {model_name} (data already exists at {output_file}). Use --force to re-collect.")
        return

    # Store responses from `model`
    responses = []
    tasks = []
    active_system_prompts = system_prompts if system_prompts else [None]
    active_neutralization_techniques = neutralization_techniques if neutralization_techniques else ["none"]
    for sys_prompt in active_system_prompts:
        for technique in active_neutralization_techniques:
            for user_prompt in prompts:
                neutralized_prompt = apply_neutralization_technique(technique, user_prompt)
                for temp in temps_list:
                    tasks.append((model_name, user_prompt, neutralized_prompt, temp, sys_prompt, technique))

    # Use ThreadPoolExecutor for parallelism on (prompt, temperature) pairs
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_task = {
            executor.submit(process_prompt, *task, args, openai_client): task
            for task in tasks
        }
        for future in tqdm(concurrent.futures.as_completed(future_to_task),
                           total=len(tasks), desc=f"Collecting response data for {model_name}", leave=False):
            responses.append(future.result())

    # Save LLM response data for `model` to JSON file
    with open(output_file, 'w') as f:
        json.dump(responses, f, indent=4)

    # Update the JSON file storing the set of LLMs. Set data_collected=True for `model`.
    with json_lock:
        updated_model_data = load_json_file(model_meta_path)
        for entry in updated_model_data:
            if entry['model_name'] == model_name:
                entry['data_collected'] = True
                break
        with open(model_meta_path, 'w') as f:
            json.dump(updated_model_data, f, indent=4)


def recollect_missing_for_model(model, args, openai_client):
    """
    Recollect only missing data points (where response is null) for a single model.
    """
    model_filename = model.replace('/', '_') + '.json'
    output_file = os.path.join(args.output_dir, model_filename)

    if not os.path.exists(output_file):
        print(f"Skipping {model}: no existing data file for recollection.")
        return

    existing_responses = load_json_file(output_file)

    tasks = []
    for i, data_point in enumerate(existing_responses):
        if data_point['response'] is None or data_point['response'] == "":
            missing_data_point = (i, data_point['prompt'], data_point['temperature'])
            tasks.append(missing_data_point)

    if not tasks:
        print(f"No missing data for {model}.")
        return

    # Use ThreadPoolExecutor for parallelism
    new_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_task = {executor.submit(process_prompt, p, t, model, args, openai_client): (idx, p, t) for idx, p, t in tasks}
        for future in tqdm(concurrent.futures.as_completed(future_to_task),
                           total=len(tasks), desc=f"Recollecting missing data for {model}", leave=False):
            idx, _, _ = future_to_task[future]
            new_results.append((idx, future.result()))

    # Update response_map with new results
    for idx, result in new_results:
        existing_responses[idx] = result

    # Save updated responses
    with open(output_file, 'w') as f:
        json.dump(existing_responses, f, indent=4)


def main():
    """
    Main entry point for the script.
    """
    args = parse_arguments()
    openai_client = load_environment()

    if args.llm and args.all:
        raise ValueError("Cannot use both --model and --all")
    if not args.llm and not args.all and not args.recollect_missing_data:
        raise ValueError("Specify --model <model-name> or --all or --recollect-missing-data <model-name>")
    if args.force and args.recollect_missing_data:
        raise ValueError("Cannot use --force with --recollect-missing-data")

    sys_prompt_file_missing = not args.system_prompt_file and args.neutralization_techniques_file
    if sys_prompt_file_missing:
        raise ValueError("Cannot use --neutralization-techniques-file without --system-prompt-file")

    # Load user prompts, LLM set, and system prompts
    prompts = load_json_file(args.prompt_file)
    model_data = load_json_file(args.llm_file)
    system_prompts = load_json_file(args.system_prompt_file) if args.system_prompt_file else None
    neutralization_techniques = load_json_file(args.neutralization_techniques_file) if args.neutralization_techniques_file else None

    # Create a list of models to process and collect data from
    if args.all:
        models_to_process = [entry['model_name'] for entry in model_data
                  if (not entry.get('data_collected', False) or args.force)]
    elif args.recollect_missing_data:
        models_to_process = [args.recollect_missing_data]
    elif args.llm:
        models_to_process = [args.llm]

    if not models_to_process:
        print("No models to process (all data already collected or invalid selection).")
        return

    os.makedirs(args.output_dir, exist_ok=True)  # Ensure base dir exists
    model_json_lock = threading.Lock()  # Lock for thread-safe JSON updates
    temps_list = generate_temps_list(args)  # Generate llm temperature list

    # Model executor for parallelizing across models
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as model_executor:
        if args.recollect_missing_data:
            future_to_model = {
                model_executor.submit(
                    recollect_missing_for_model, model_name, args, openai_client
                ): model_name for model_name in models_to_process
            }
        else:
            future_to_model = {
                model_executor.submit(process_model, model_name, args.llm_file,
                                      args, prompts, temps_list, system_prompts,
                                      neutralization_techniques,
                                      openai_client, model_json_lock): model_name
                for model_name in models_to_process
            }
        for future in tqdm(concurrent.futures.as_completed(future_to_model), total=len(models_to_process), desc="Processing models", leave=True):
            future.result()  # Wait for completion

    print(f"Data collection complete. Saved responses to {args.output_dir}")


if __name__ == "__main__":
    main()