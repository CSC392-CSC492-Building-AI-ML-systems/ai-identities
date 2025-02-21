import os
import threading
import json
import time
import psutil
import requests
import subprocess
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import toml
from openai import OpenAI

parser = argparse.ArgumentParser(
    prog="python3 boolq_eval.py",
    description="Run Boolq test on multiple ollama models with load balancing",
)
parser.add_argument(
    "-c",
    "--config",
    help="Configuration file. Default=config.toml",
    default="config.toml",
)
parser.add_argument("-p", "--parallel", type=int, help="Number of parallel requests")
parser.add_argument("-m", "--model", help="Model name")
args = parser.parse_args()
config = toml.load(open(args.config))

if args.parallel:
    config["test"]["parallel"] = args.parallel
if args.model:
    config["server"]["model"] = args.model

# im writing these comments so that i dont forget about it💀
# FOR TESTING PURPOSES:
# I'll prolly add a test flag as an argument, which would only test 10 examples
#  mainly to check if load balacning is wokring on Niagara (on the debug partition)
#  something like: python run_openai.py --config config.toml --parallel 4 --model llama3.2:3b --test

# by adding these lines:
# parser.add_argument("--test", action="store_true", help="Run with test dataset")
# if args.test:
#     dataset = dataset[:10]
# along with changing the nodes/cores and time in the slurm script

# Define server configurations
SERVERS = [
    {"url": "http://localhost:11434/v1", "port": 11434},
    {"url": "http://localhost:11435/v1", "port": 11435},
    {"url": "http://localhost:11436/v1", "port": 11436}
]

# Create OpenAI clients for each server
clients = [
    OpenAI(
        base_url=server["url"],
        api_key=config["server"]["api_key"],
        timeout=config["server"]["timeout"]
    )
    for server in SERVERS
]

results = []
predictions = []
ground_truth = []
idx_check = set()

print("Loading BoolQ dataset...")

# not sure about this as drop_eval needs it the parquet files loaded explicitly by name
# will update if needed

dataset = load_dataset(os.environ.get('SCRATCH')+'/ai-identities/performance-evals/boolq-data', split="validation")
print(f"Loaded {len(dataset)} examples")
print("Starting evaluation...")

def get_server_for_index(idx):
    """
    Determine which server to use based on the last digit of the index
    """
    last_digit = idx % 10
    if last_digit <= 2:
        return 0  # Server 1
    elif last_digit <= 5:
        return 1  # Server 2
    else:
        return 2  # Server 3

def format_prompt(question, passage):
    """
    Format the prompt for the model
    """
    return f"""Given the following passage and question, answer with only 'yes' or 'no'.

Passage: {passage}

Question: {question}

Answer:"""

def query_ollama(idx):
    """
    Send a query to the appropriate Ollama server based on the index, aka last digit of question id
    """
    example = dataset[idx]

    if idx in idx_check:
        return None, None
    
    server_idx = get_server_for_index(idx)
    client = clients[server_idx]
    
    prompt = format_prompt(example['question'], example['passage'])
    try:
        response = client.chat.completions.create(
            model=config["server"]["model"],
            messages=[{"role":"user", "content":prompt}],
            temperature=config["inference"]["temperature"],
            max_tokens=config["inference"]["max_tokens"],
            top_p=config["inference"]["top_p"],
            frequency_penalty=0,
            presence_penalty=0,
            timeout=config["server"]["timeout"],
        )
        response_str = response.choices[0].message.content.strip()
        cleaned_response = clean_response(response_str)
        idx_check.add(idx)

        return {
            'question': example['question'],
            'passage': example['passage'],
            'predicted': cleaned_response,
            'actual': 'yes' if example['answer'] else 'no',
            'correct': (cleaned_response == 'yes') == example['answer'],
            'server': f"server-{server_idx+1}"  # Track which server handled the request
        }, cleaned_response

    except Exception as e:
        raise RuntimeError(f"Error querying Ollama, specifically server {server_idx+1}: {e}")

def clean_response(response):
    """
    Clean the model's response to get just yes/no
    """
    response = response.lower().strip()
    if 'yes' in response:
        return 'yes'
    elif 'no' in response:
        return 'no'
    else:
        return 'invalid'

with ThreadPoolExecutor(max_workers=config["test"]["parallel"]) as executor:
    futures = {
        executor.submit(query_ollama, idx): idx
        for idx in range(len(dataset))
    }

    for future in tqdm(
        as_completed(futures), total=len(futures), smoothing=0.0, ascii=True
    ):
        idx = futures[future]
        result, cleaned_response = future.result()
        
        if result == None:
            continue
        
        results.append(result)
        predictions.append(cleaned_response == 'yes')
        ground_truth.append(dataset[idx]['answer'])

# Calculate metrics
accuracy = accuracy_score(ground_truth, predictions)
print(f"\nAccuracy: {accuracy:.4f}")

# Create detailed report
results_df = pd.DataFrame(results)
print("\nClassification Report:")
print(classification_report(ground_truth, predictions))

# Found this online, could help debugging if load balancing is not working
# its checking the distribution of requests across servers
server_distribution = results_df['server'].value_counts()
print("\nServer Distribution:")
print(server_distribution)

# Save results
results_df.to_csv(config["server"]["model"]+'_boolq_results.csv', index=False)
print("\nResults saved to "+config["server"]["model"]+'_boolq_results.csv')