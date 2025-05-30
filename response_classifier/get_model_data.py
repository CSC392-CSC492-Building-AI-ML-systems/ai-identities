import threading
import requests
import numpy as np
import json
import argparse
import uuid
import re
import time
from constants import PROMPTS
from collections import defaultdict


# Assuming PROMPTS is defined elsewhere
# PROMPTS = "prompt1\nprompt2\nprompt3\n..."

prompt_set = list(set(PROMPTS.split("\n")))
# Remove empty strings if any
prompt_set = [p.strip() for p in prompt_set if p.strip()]

parser = argparse.ArgumentParser(description="Collecting data for APIs")
parser.add_argument("--url", type=str, required=True, help="OpenAI-compatible API endpoint")
parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key")
parser.add_argument("--model", type=str, required=True, help="Model to use for completion")
args = parser.parse_args()

# Initialize thread-safe data structures
lock = threading.Lock()
test_dict = defaultdict(lambda: defaultdict(int))
counter = 0
prompt_index = 0

def get_response():
    global prompt_index, counter
    # Get next prompt from prompt_set (cycling through)
    if not prompt_set:
        print("No prompts available in prompt_set")
        return
    
    with lock:
        prompt = prompt_set[prompt_index % len(prompt_set)]
        prompt_index += 1
    
    headers = {
        "Authorization": f"Bearer {args.api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": args.model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "max_tokens": 1
    }
    
    try:
        response = requests.post(f"{args.url}/chat/completions", headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes
        response_json = response.json()
        
        content = response_json["choices"][0]["message"]["content"]
        
        with lock:
            test_dict[prompt] = content
            
            counter += 1
            if counter % 10 == 0:  # Progress indicator
                print(f"Completed {counter} requests")
                
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(0.5)

def run_threads(num_threads):
    """Helper function to create and manage threads"""
    threads = []
    
    for _ in range(num_threads):
        thread = threading.Thread(target=get_response)
        thread.start()
        threads.append(thread)
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()

# Main execution
if __name__ == "__main__":
    print(f"Using {len(prompt_set)} unique prompts")
    print("Starting data collection...")
    
    # First batch: 4 rounds of 200 threads each
    for i in range(4):
        print(f"Starting batch {i+1}/4 with 200 threads")
        run_threads(200)
        print(f"Completed batch {i+1}/4")
    
    # Second batch: 150 threads
    print("Starting final batch with 150 threads")
    run_threads(150)
    print("All requests completed!")
    
    # Compute statistics
    if("/" in args.model):
        filename = f"./results/{args.model.split("/")[1]}_results.json"
    else:
        filename = f"{args.model}_results.json"
    with open(filename, "w") as f:
        f.write(json.dumps(test_dict, indent="\t"))
    print(f"Results saved to {filename}")