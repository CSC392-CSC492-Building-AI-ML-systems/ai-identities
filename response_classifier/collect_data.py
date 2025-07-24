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

# Constants (global, but minimal)
MODEL_JSON = 'llm_set1.json'
PROMPT_JSON = 'user_prompt_set.json'
BASE_DIR = 'model_selection_dataset/raw_data'


def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Collect LLM responses from Deepinfra")
    parser.add_argument('--model', type=str,
                        help="Single model name to process (e.g., Qwen/Qwen3-235B-A22B)")
    parser.add_argument('--all', action='store_true',
                        help="Process all models from llm_set1.json where data_collected is false")
    parser.add_argument('--prevent-cache', action='store_true',
                        help="Add unique ID to prompts to prevent potential caching")
    parser.add_argument('--max-tokens', type=int, default=5000,
                        help="Max tokens for responses (default: 1500)")
    parser.add_argument('--temperature', type=float, default=0.7,
                        help="Temperature for generation when data_point_num=1 (default: 0.7)")
    parser.add_argument('--system-prompt', type=str, default="",
                        help="Optional system prompt (e.g., 'You are a helpful assistant')")
    parser.add_argument('--max-workers', type=int, default=100,
                        help="Max concurrent threads (default: 100; adjust for rate limits)")
    parser.add_argument('--force', action='store_true',
                        help="Force collection even if data exists or data_collected is true")
    parser.add_argument('--data-point-num', type=int, default=1,
                        help="Number of data points per (model, user prompt) pair (default: 1; must be divisible by 3 if >1)")
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


def generate_llm_temperatures(low, high, num, inclusive_high):
    """
    Generate evenly spaced temperatures for a bin.
    """
    if num == 0:
        return []
    if num == 1:
        return [low]  # Start of the bin for single point

    if inclusive_high:
        step = (high - low) / (num - 1)
    else:
        step = (high - low) / num

    temps = [low + i * step for i in range(num)]
    return temps


def generate_temps_list(args):
    """
    Generate temperature list based on data_point_num.
    """
    if args.data_point_num == 1:
        return [args.temperature]
    elif args.data_point_num % 3 != 0:
        raise ValueError("data_point_num must be divisible by 3 when it is greater than 1")
    else:
        num_per_bin = args.data_point_num // 3
        low_temps = generate_llm_temperatures(0.0, 0.35, num_per_bin, inclusive_high=False)
        med_temps = generate_llm_temperatures(0.35, 0.7, num_per_bin, inclusive_high=False)
        high_temps = generate_llm_temperatures(0.7, 1.0, num_per_bin, inclusive_high=True)
        return low_temps + med_temps + high_temps


def process_prompt(prompt, temperature, model, args, openai_client) -> dict:
    """
    Processes a single prompt at a specific temperature.
    """
    if args.prevent_cache:
        unique_id = str(uuid.uuid4())
        modified_prompt = f"{prompt} [unique_id: {unique_id}]"
    else:
        modified_prompt = prompt

    messages = [{"role": "user", "content": modified_prompt}]
    if args.system_prompt:
        messages.insert(0, {"role": "system", "content": args.system_prompt})

    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=args.max_tokens,
            temperature=temperature
        )
        generated_text = response.choices[0].message.content.strip()
        return {
            'prompt': prompt,
            'modified_prompt': modified_prompt if args.prevent_cache else None,
            'response': generated_text,
            'temperature': temperature,
            'timestamp': time.time(),
            'error': None
        }
    except Exception as e:
        return {
            'prompt': prompt,
            'modified_prompt': modified_prompt if args.prevent_cache else None,
            'response': None,
            'temperature': temperature,
            'timestamp': time.time(),
            'error': str(e)
        }


def process_model(model, args, prompts, temps_list, openai_client, json_lock):
    """
    Process data collection for a single model.
    """
    # Note that Deepinfra model names contain `/` so we replace it with `_`
    model_filename = model.replace('/', '_') + '.json'
    output_file = os.path.join(BASE_DIR, model_filename)

    # Skip if data exists and --force not set (additional check beyond JSON flag)
    if os.path.exists(output_file) and not args.force:
        print(f"Skipping {model} (data already exists at {output_file}). Use --force to re-collect.")
        return

    # Store responses from `model`
    responses = []

    # Use ThreadPoolExecutor for parallelism on (prompt, temperature) pairs
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        tasks = [(prompt, temp) for prompt in prompts for temp in temps_list]
        future_to_task = {executor.submit(process_prompt, p, t, model, args, openai_client): (p, t) for p, t in tasks}
        for future in tqdm(concurrent.futures.as_completed(future_to_task),
                           total=len(tasks), desc=f"Collecting response data for {model}", leave=False):
            responses.append(future.result())
            time.sleep(0.5)  # Small delay to avoid rate limits; adjust or remove

    # Save LLM response data for `model` to JSON file
    with open(output_file, 'w') as f:
        json.dump(responses, f, indent=4)

    print(f"Saved responses for {model} to {output_file}")

    # Update llm_set1.json to set data_collected=True for `model`
    with json_lock:
        updated_model_data = load_json_file(MODEL_JSON)
        for entry in updated_model_data:
            if entry['model_name'] == model:
                entry['data_collected'] = True
                break
        with open(MODEL_JSON, 'w') as f:
            json.dump(updated_model_data, f, indent=4)


def main():
    """
    Main entry point for the script.
    """
    args = parse_arguments()
    if args.model and args.all:
        raise ValueError("Cannot use both --model and --all")
    if not args.model and not args.all:
        raise ValueError("Specify --model <name> or --all")

    openai_client = load_environment()

    # Load user prompts and model data
    prompts = load_json_file(PROMPT_JSON)
    model_data = load_json_file(MODEL_JSON)

    # Get models to process
    if args.all:
        models = [entry['model_name'] for entry in model_data
                  if (not entry.get('data_collected', False) or args.force)]
    else:
        models = [args.model]

    if not models:
        print("No models to process (all data already collected or invalid selection).")
        return

    os.makedirs(BASE_DIR, exist_ok=True)  # Ensure base dir exists
    model_json_lock = threading.Lock()  # Lock for thread-safe JSON updates
    temps_list = generate_temps_list(args)  # Generate llm temperature list

    # Outer/model executor for parallelizing across models
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as model_executor:  # Adjust workers for rate limits
        future_to_model = {
            model_executor.submit(
                process_model, model, args, prompts, temps_list, openai_client, model_json_lock
            ): model for model in models
        }
        for future in tqdm(concurrent.futures.as_completed(future_to_model), total=len(models), desc="Processing models", leave=True):
            future.result()  # Wait for completion

    print("Data collection complete.")


if __name__ == "__main__":
    main()