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

parser = argparse.ArgumentParser(
	prog="python3 boolq_eval.py",
	description="Run Boolq test on an ollama model",
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


results = []
predictions = []
ground_truth = []

def format_prompt(question, passage):
    """
    Format the prompt for the model
    """
    return f"""Given the following passage and question, answer with only 'yes' or 'no'.

Passage: {passage}

Question: {question}

Answer:"""

def query_ollama(example):
    """
    Send a query to the Ollama API and get the response
    """

    prompt = format_prompt(example['question'], example['passage'])
    try:
        response = requests.post('http://localhost:11434/api/generate',
                               json={
                                   'model': config["server"]["model"],
                                   'prompt': prompt,
                                   'stream': False
                               })
        response.raise_for_status()
        response_str = response.json()['response']
        cleaned_response = clean_response(response_str)

        return {
            'question': example['question'],
            'passage': example['passage'],
            'predicted': cleaned_response,
            'actual': 'yes' if example['answer'] else 'no',
            'correct': (cleaned_response == 'yes') == example['answer']
        }, cleaned_response

    except Exception as e:
        raise RuntimeError(f"Error querying Ollama: {e}")



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
    


print("Loading BoolQ dataset...")
dataset = load_dataset("./boolq-data", split="validation")
print(f"Loaded {len(dataset)} examples")
print("Starting evaluation...")


with ThreadPoolExecutor(max_workers=config["test"]["parallel"]) as executor:
    futures = {
        executor.submit(query_ollama, dataset[idx]): idx
        for idx in range(len(dataset))
    }

    for future in tqdm(
        as_completed(futures), total=len(futures), smoothing=0.0, ascii=True
    ):
        idx = futures[future]
        result, cleaned_response = future.result()
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

# Save results
results_df.to_csv(config["server"]["model"]+'_boolq_results.csv', index=False)
print("\nResults saved to "+config["server"]["model"]+'_boolq_results.csv')
               

