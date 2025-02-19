import os 
import json
import random
import argparse
from tqdm import tqdm
import toml
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
from models import send_to_model
from utils import extract_answer
from metrics import drop_metric

def evaluate_model(model, test_cases, config):
    print("here3")
    results = []
    
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_test_case, model, test_case, config): test_case for test_case in test_cases}
        
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
    return results

def process_test_case(model, test_case, config):
    print("here4")
    prompt = construct_prompt(test_case)
    print("here7")
    response = send_to_model(model, prompt, config)
    print("here8")
    extracted_answer = extract_answer(response)
    em_score, f1_score = drop_metric(extracted_answer, test_case["ref_text"].split("|"))
    
    return {
        "test_case": test_case,
        "response": response,
        "em_score": em_score,
        "f1_score": f1_score
    }

def construct_prompt(test_case):
    print("here5")
    examples = load_examples(num_examples=3)
    prompt = f"""
You will be asked to read a passage and answer a question. Some examples of passages and Q&A are provided below.

# Examples
{examples}

# Your Task
---
Context: {test_case["context"]}
Question: {test_case["question"]}

Think step by step, then write a line of the form "Answer: $ANSWER" at the end of your response.
"""
    return prompt

def main():
    print("here2")
    parser = argparse.ArgumentParser(description="Evaluate models on DROP dataset.")
    parser.add_argument("--config", default="config/config.toml", help="Path to config file.")
    args = parser.parse_args()

    config = toml.load(args.config)

    # Doing first 1000 training cases out of 77400 training examples
    # Will change this once we have a better understanding of the performance
    # of the models, after running it through the Niagra cluster
    test_cases = load_drop_test_cases(subset=1000)

    models = config["server"]["models"]
    # s
    for model in models:
        results = evaluate_model(model, test_cases, config)
        save_results(results, f"results/{model}_results.json")

def save_results(results, filepath):
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)

def load_examples(train_path=None, num_examples=3):
    # Load 3 examples from the dataset
    print("here6")
    dataset = load_dataset("ucinlp/drop")
    train_data = dataset["train"]
    sampled_examples = random.sample(list(train_data), num_examples)

    formatted_examples = []
    for example in sampled_examples:
        context = example["passage"]
        question = example["question"]
        answer = example["answers_spans"]["spans"][0]

        formatted_examples.append(f"Context: {context}\nQuestion: {question}\nAnswer: {answer}")
    
    print(f"{formatted_examples} done bye")
    return "\n\n".join(formatted_examples)
def load_drop_test_cases(subset=1000):
    dataset = load_dataset("ucinlp/drop")
    train_data = dataset["train"]

    train_data = train_data.select(range(min(subset, len(train_data))))
    test_cases = []

    for example in train_data:
        entry = example["answers_spans"]
        ref_text = entry["spans"][0]  # Safe handling of multiple answers

        test_cases.append({
            "context": example["passage"],
            "question": example["question"],
            "ref_text": ref_text
        })

    return test_cases

if __name__ == "__main__":
    print("here")
    main()
