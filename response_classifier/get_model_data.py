import threading
import requests
import json
import argparse
import time
from constants import PROMPTS
from collections import defaultdict, Counter
from queue import Queue

# Clean and prepare prompt list
prompt_set = list(set(PROMPTS.split("\n")))
prompt_set = [p.strip() for p in prompt_set if p.strip()]

# Create a queue with each prompt 10 times
prompt_queue = Queue()
for prompt in prompt_set:
    for _ in range(20):
        prompt_queue.put(prompt)

# Parse CLI arguments
parser = argparse.ArgumentParser(description="Collecting data for APIs")
parser.add_argument("--url", type=str, required=True, help="OpenAI-compatible API endpoint")
parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key")
parser.add_argument("--model", type=str, required=True, help="Model to use for completion")
parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for the model")
args = parser.parse_args()

# Initialize thread-safe data structures
lock = threading.Lock()
test_dict = defaultdict(Counter)
counter = 0

def get_response():
    global counter
    while not prompt_queue.empty():
        try:
            prompt = prompt_queue.get_nowait()
        except:
            return

        headers = {
            "Authorization": f"Bearer {args.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": args.temperature,
            "max_tokens": 1
        }

        try:
            response = requests.post(f"{args.url}/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            response_json = response.json()

            content = response_json["choices"][0]["message"]["content"].strip()

            with lock:
                test_dict[prompt][content] += 1
                counter += 1
                if counter % 50 == 0:
                    print(f"Completed {counter} requests")

        except Exception as e:
            print(f"Error: {e}")
            time.sleep(0.5)
            get_response()

def run_threads(num_threads):
    threads = []

    for _ in range(num_threads):
        thread = threading.Thread(target=get_response)
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

# Main execution
if __name__ == "__main__":
    total_requests = len(prompt_set) * 10
    print(f"Using {len(prompt_set)} unique prompts, totaling {total_requests} requests")
    print("Starting data collection...")

    while not prompt_queue.empty():
        batch_size = min(200, prompt_queue.qsize())
        print(f"Starting batch with {batch_size} threads")
        run_threads(batch_size)

    print("All requests completed!")

    # Convert defaultdict(Counter) to regular dicts for JSON serialization
    result_dict = {prompt: dict(counter) for prompt, counter in test_dict.items()}

    if "/" in args.model:
        filename = f"./results/{args.model.split('/')[1]}_results_{args.temperature}.json"
    else:
        filename = f"./results/{args.model}_results_{args.temperature}.json"

    with open(filename, "w") as f:
        json.dump(result_dict, f, indent=2)

    print(f"Results saved to {filename}")