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
    parser.add_argument('--max-tokens', type=int, default=3500,
                        help="Max tokens for responses (default: 3500)")
    parser.add_argument('--temperature', type=float, default=0.7,
                        help="Temperature for generation when data_point_num=1 (default: 0.7)")
    parser.add_argument('--system-prompt', type=str, default="",
                        help="Optional system prompt (e.g., 'You are a helpful assistant')")
    parser.add_argument('--max-workers', type=int, default=125,
                        help="Max concurrent threads (default: 125; adjust for rate limits)")
    parser.add_argument('--force', action='store_true',
                        help="Force collection even if data exists or data_collected is true")
    parser.add_argument('--low-bin-num', type=int, default=0,
                        help="Number of data points in low temp bin [0.0, 0.4) (must be multiple of 6 if >0; default is 0)")
    parser.add_argument('--med-bin-num', type=int, default=0,
                        help="Number of data points in med temp bin [0.4, 0.8) (must be multiple of 6 if >0; default is 0)")
    parser.add_argument('--high-bin-num', type=int, default=0,
                        help="Number of data points in high temp bin [0.8, 1.2) (must be multiple of 6 if >0; default is 0)")
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
    if num == 0:
        return []
    if num == 1:
        return [low]  # Start of the bin for single point

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
        for num, name in [(args.low_bin_num, 'low'), (args.med_bin_num, 'med'), (args.high_bin_num, 'high')]:
            if num > 0 and num % 6 != 0:
                raise ValueError(f"{name}-bin-num must be a multiple of 6 when greater than 0")

        low_temps = generate_llm_temperatures(0.0, 0.4, args.low_bin_num)
        med_temps = generate_llm_temperatures(0.4, 0.8, args.med_bin_num)
        high_temps = generate_llm_temperatures(0.8, 1.2, args.high_bin_num)
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

    max_retries = 3
    for attempt in range(max_retries):
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
                'error': None,
                'attempt': attempt + 1
            }
        except Exception as e:
            if attempt == (max_retries - 1):
                return {
                    'prompt': prompt,
                    'modified_prompt': modified_prompt if args.prevent_cache else None,
                    'response': None,
                    'temperature': temperature,
                    'timestamp': time.time(),
                    'error': str(e),
                    'attempt': attempt + 1
                }
            time.sleep(2 ** attempt)


def process_model(model, args, prompts, temps_list, openai_client, json_lock):
    """
    Process data collection for a single model.
    """
    # Note that Deepinfra model names contain `/` so we replace it with `_`
    model_filename = model.replace('/', '_') + '.json'
    output_file = os.path.join(BASE_DIR, model_filename)

    # Skip if data exists and --force is not set
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

    # Save LLM response data for `model` to JSON file
    with open(output_file, 'w') as f:
        json.dump(responses, f, indent=4)

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

    # Create a list of models to process and collect data from
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

    # Model executor for parallelizing across models
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as model_executor:  # Adjust workers for rate limits
        future_to_model = {
            model_executor.submit(
                process_model, model, args, prompts, temps_list, openai_client, model_json_lock
            ): model for model in models
        }
        for future in tqdm(concurrent.futures.as_completed(future_to_model), total=len(models), desc="Processing models", leave=True):
            future.result()  # Wait for completion

    print(f"Data collection complete. Saved responses to {BASE_DIR}")


if __name__ == "__main__":
    main()