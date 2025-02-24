import os
import json
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
import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('boolq_eval.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

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
parser.add_argument("-t", "--test", action="store_true", help="Run with test dataset")
args = parser.parse_args()

logging.info(f"Starting script with arguments: {args}")

try:
    config = toml.load(open(args.config))
    logging.info("Successfully loaded config file")
except Exception as e:
    logging.error(f"Failed to load config file: {e}")
    sys.exit(1)

if args.parallel:
    config["test"]["parallel"] = args.parallel
if args.model:
    config["server"]["model"] = args.model


# ____      THIS SNIPPET IS CRUCIAL FOR EVERY EVAL     ____________
# ____ IF GOING MULTI-NODE, THIS IS THE PART TO ADD    ____

# We set
# Get server URLs from environment variable
server_list = os.environ.get('OLLAMA_SERVERS', '').split(',')
logging.info(f"OLLAMA_SERVERS environment variable: {os.environ.get('OLLAMA_SERVERS', 'Not set')}")

if not server_list[0]:
    server_list = ['localhost:11434', 'localhost:11435', 'localhost:11436']
    # IT SHOULD NEVER BE HERE, EVEN IF ALL NODES SOMEHOW ACTUALLY HAVE SAME URL
    logging.warning("Using default server list as OLLAMA_SERVERS not set")

SERVERS = [
    {"url": f"http://{server}/v1", "port": int(server.split(':')[1])}
    for server in server_list if server  # Only include non-empty server strings
    #(doesnt do much, just avoids populating ips of the nodes that dont actually run an ollama node)
]
logging.info(f"Configured servers: {SERVERS}")

# Create OpenAI clients for each server
try:
    clients = [
        OpenAI(
            base_url=server["url"],
            api_key=config["server"]["api_key"],
            timeout=config["server"]["timeout"]
            )
        for server in SERVERS
    ]
    logging.info("Successfully created OpenAI clients")
except Exception as e:
    logging.error(f"Failed to create OpenAI clients: {e}")
    sys.exit(1)

results = []
predictions = []
ground_truth = []
idx_check = set()
server_index = 0
lock = threading.Lock()

def get_next_server():
    """
    Get the next ollama server by cycling through every cycle
    Better/Easier to scale than checking last id of question
    Because the number of ollama nodes will be variable (eventually)"""
    global server_index
    with lock:
        server = server_index
        server_index = (server_index + 1) % len(SERVERS)
    return server

def format_prompt(question, passage):
    """Format the prompt for the model."""
    return f"""Given the following passage and question, answer with only 'yes' or 'no'.

Passage: {passage}

Question: {question}

Answer:"""

def check_server_health(client, server_url):
    """Check if the server is responsive."""
    try:
        # Try a simple API call to check server health
        response = client.chat.completions.create(
            model=config["server"]["model"],
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1
        )
        logging.info(f"Health check successful for {server_url}")
        return True
    except Exception as e:
        logging.error(f"Server health check failed for {server_url}: {e}")
        return False

def clean_response(response):
    """Clean the model's response to get just yes/no."""
    response = response.lower().strip()
    if 'yes' in response:
        return 'yes'
    elif 'no' in response:
        return 'no'
    else:
        return 'invalid'

def query_ollama(idx):
    """Send a query to the appropriate Ollama server."""
    try:
        example = dataset_list[idx]
        logging.debug(f"Processing example {idx}")
    except IndexError:
        logging.error(f"Index {idx} is out of range for dataset size {len(dataset_list)}")
        return None, None

    if idx in idx_check:
        logging.warning(f"Index {idx} already processed")
        return None, None

    server_idx = get_next_server()
    client = clients[server_idx]
    server_url = SERVERS[server_idx]["url"]
    
    logging.info(f"Attempting query to server {server_url} for example {idx}")

    prompt = format_prompt(example['question'], example['passage'])
    try:
        logging.debug(f"Sending request to server {server_idx} for example {idx}")
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
        
        with lock:
            idx_check.add(idx)

        result = {
            'question': example['question'],
            'passage': example['passage'],
            'predicted': cleaned_response,
            'actual': 'yes' if example['answer'] else 'no',
            'correct': (cleaned_response == 'yes') == example['answer'],
            'server': f"server-{server_idx+1}",
            'raw_response': response_str
        }
        
        logging.info(f"Successfully processed example {idx} on server {server_idx}")
        logging.debug(f"Result: {result}")
        
        return result, cleaned_response

    except Exception as e:
        logging.error(f"Error querying Ollama server {server_idx+1} for example {idx}: {e}")
        return None, None

# Load BoolQ dataset
try:
    dataset_path = os.environ.get('SCRATCH')+'/ai-identities/performance-evals/boolq-data' if 'SCRATCH' in os.environ else './boolq-data'
    logging.info(f"Loading dataset from: {dataset_path}")
    dataset = load_dataset(dataset_path, split="validation")
    dataset_list = list(dataset)
    if args.test:
        dataset_list = dataset_list[:10]
    logging.info(f"Successfully loaded dataset with {len(dataset_list)} examples")
except Exception as e:
    logging.error(f"Failed to load dataset: {e}")
    sys.exit(1)

# Verify servers are healthy before starting
# ________________________________Snippet 2_____________________________________________________
# OPTIONAL: Check if servers are functionaly every now and then
# useful to add in your script, easy to debug and see whats going wrong
# is populated in server-logs for each node as well as the script-log called boolq_eval.log
healthy_servers = 0
for idx, (server, client) in enumerate(zip(SERVERS, clients)):
    if check_server_health(client, server["url"]):
        healthy_servers += 1
    else:
        logging.error(f"Server {idx+1} is not healthy")

if healthy_servers == 0:
    logging.error("No healthy servers available. Exiting.")
    sys.exit(1)

logging.info(f"Found {healthy_servers} healthy servers")
# ________________________________Snippet 2 end_____________________________________________________
# Main evaluation loop
try:
    with ThreadPoolExecutor(max_workers=min(config["test"]["parallel"], len(dataset_list))) as executor:
        futures = {
            executor.submit(query_ollama, idx): idx
            for idx in range(len(dataset_list))
        }

        for future in tqdm(
            as_completed(futures), 
            total=len(futures), 
            smoothing=0.0, 
            ascii=True,
            desc="Processing examples"
        ):
            try:
                idx = futures[future]
                result, cleaned_response = future.result()

                if result is None:
                    logging.warning(f"No result for example {idx}")
                    continue

                results.append(result)
                predictions.append(cleaned_response == 'yes')
                ground_truth.append(dataset_list[idx]['answer'])
                logging.debug(f"Successfully processed and stored result for example {idx}")
            except Exception as e:
                logging.error(f"Error processing result for example {idx}: {e}")

except Exception as e:
    logging.error(f"Error in main evaluation loop: {e}")
    sys.exit(1)

# Calculate and save metrics
if predictions and ground_truth:
    try:
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Calculate accuracy
        accuracy = accuracy_score(ground_truth, predictions)
        logging.info(f"\nAccuracy: {accuracy:.4f}")
        
        # Generate classification report
        report = classification_report(ground_truth, predictions)
        logging.info(f"\nClassification Report:\n{report}")
        
        # Calculate server distribution (this should be almost equal for all servers
        # if load balancing is working properly / all clients were functional the whole time
        server_distribution = results_df['server'].value_counts()
        logging.info(f"\nServer Distribution:\n{server_distribution}")
        
        # Save results
        output_file = f"{config['server']['model']}_boolq_results.csv"
        results_df.to_csv(output_file, index=False)
        logging.info(f"\nResults saved to {output_file}")
        
    except Exception as e:
        logging.error(f"Error calculating/saving metrics: {e}")
else:
    logging.error("No predictions or ground truth available to calculate metrics")

logging.info(f"\nProcessed {len(idx_check)} out of {len(dataset_list)} examples")
