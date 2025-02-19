import gzip
import json
import random
import re

def load_test_cases(test_jsonl_path, subset=1.0):
    with gzip.open(test_jsonl_path, "rt") as f:
        test_samples = [json.loads(line) for line in f]
    if subset < 1.0:
        test_samples = random.sample(test_samples, int(len(test_samples) * subset))
    return test_samples

def extract_answer(response):
    match = re.search(r"Answer: (.*)", response)
    return match.group(1).strip() if match else response.strip()

def normalize(text):
    # Normalize text for comparison
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return " ".join(text.split())