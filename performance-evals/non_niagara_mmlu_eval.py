import os
import re
import json
import time
import random
import sys
from tqdm import tqdm
from openai import OpenAI
from datasets import load_dataset, Dataset, DatasetDict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from datetime import datetime, timedelta
import codecs
import toml
import argparse
import uuid
import queue
import numpy as np
import copy

parser = argparse.ArgumentParser(
    prog="python3 run_openai.py",
    description="Run MMLU Pro Benchmark for  a local LLM  via  OpenAI Compatible API.",
    epilog="Specify  options above  to override  one or more settings from config.",
)
parser.add_argument(
    "-c",
    "--config",
    help="Configuration file. Default=non_niagara_mmlu_test.toml",
    default="non_niagara_mmlu_test.toml",
)
parser.add_argument("-u", "--url", help="server url")
parser.add_argument("-a", "--api", help="api key")
parser.add_argument("-m", "--model", help="Model name")
parser.add_argument("--timeout", type=float, help="Request timeout in seconds")
parser.add_argument("--category", type=str)
parser.add_argument("--subset", type=float, help="Fraction (0.4 for 40%) of items to keep per category.")
parser.add_argument("-p", "--parallel", type=int, help="Number of parallel requests")
parser.add_argument("-v", "--verbosity", type=int, help="Verbosity level 0-2")
parser.add_argument(
    "--log_prompt",
    help="Writes exact prompt and response into log.txt",
    action="store_true",
)
parser.add_argument("--comment", type=str, help="Comment to be included in the final report.")
args = parser.parse_args()

try:
    config = toml.load(open(args.config))
except Exception as e:
    print("Error loading configuration:", e)
    sys.exit(1)

if args.url:
    config["server"]["url"] = args.url
if args.api:
    config["server"]["api_key"] = args.api
if args.model:
    config["server"]["model"] = args.model
if args.timeout:
    config["server"]["timeout"] = args.timeout
if args.category:
    config["test"]["categories"] = [args.category]
if args.subset:
    config["test"]["subset"] = args.subset
if args.parallel:
    config["test"]["parallel"] = args.parallel
if args.verbosity:
    config["log"]["verbosity"] = args.verbosity
if args.log_prompt:
    config["log"]["log_prompt"] = args.log_prompt
if args.comment:
    config["comment"] = args.comment

print(config["server"]["url"])
print(config["server"]["api_key"])
print(config["inference"]["temperature"])
# Create OpenAI client
print(config["server"]["url"])

client = OpenAI(
    base_url=config["server"]["url"],
    api_key=config["server"]["api_key"],
    timeout=config["server"]["timeout"]
)

# Thread-safe server rotation and rate limiting
server_index = 0
lock = threading.Lock()

# Replace the existing rate_limit_lock and throttle_request() with this:

# Global rate limiter variables
rate_limit_lock = threading.Lock()
tokens = 1  # Start with 1 token available
last_refill_time = time.time()
# Configure these in your TOML (add [server] rate_limit_rpm = 60)
RPM = config["server"].get("rate_limit_rpm", 60)  # Requests per minute
REFILL_INTERVAL = 60.0 / RPM  # Seconds between tokens
BUCKET_SIZE = 1  # Maximum burst capacity

tpm_lock = threading.Lock()
tpm_tokens = 500_000  # Initial token count set to the TPM limit
tpm_last_refill_time = time.time()
TPM_REFILL_PER_SECOND = 500_000 / 60  # ~8333.33 tokens per second
TPM_BUCKET_SIZE = 500_000  # Maximum burst capacity

def throttle_request():
    global tokens, last_refill_time
    with rate_limit_lock:
        # Calculate how many tokens should be added since last refill
        current_time = time.time()
        time_since_refill = current_time - last_refill_time
        new_tokens = time_since_refill / REFILL_INTERVAL

        if new_tokens > 0:
            tokens = min(tokens + new_tokens, BUCKET_SIZE)
            last_refill_time = current_time

        # Wait until at least 1 token is available
        while tokens < 1:
            time_to_wait = REFILL_INTERVAL - (current_time - last_refill_time)
            if time_to_wait > 0:
                time.sleep(time_to_wait)
                current_time = time.time()
                new_tokens = (current_time - last_refill_time) / REFILL_INTERVAL
                tokens = min(tokens + new_tokens, BUCKET_SIZE)
                last_refill_time = current_time
            else:
                last_refill_time = current_time
                new_tokens = 0

        # Take a token
        tokens -= 1

def log(message):
    print(message)
    with codecs.open(log_path, "a", "utf-8") as file:
        file.write(message + "\n")

def get_chat_completion(messages):
    global tpm_tokens, tpm_last_refill_time
    throttle_request()  # Enforce RPM limit
    try:
        response = client.chat.completions.create(
            model=config["server"]["model"],
            messages=messages,
            temperature=config["inference"]["temperature"],
            max_tokens=config["inference"]["max_tokens"],
            top_p=config["inference"]["top_p"],
            frequency_penalty=0,
            presence_penalty=0,
            stop=["Question:"],
            timeout=config["server"]["timeout"],
        )
        if response is None:
            print("Received None response from API.")
            sys.exit(1)
        try:
            # Track token usage
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            usage_q.put((prompt_tokens, completion_tokens))
            # Deduct tokens from TPM bucket
            used_tokens = prompt_tokens + completion_tokens
            with tpm_lock:
                current_time = time.time()
                time_since_refill = current_time - tpm_last_refill_time
                refilled_tokens = time_since_refill * TPM_REFILL_PER_SECOND
                tpm_tokens_current = min(tpm_tokens + refilled_tokens, TPM_BUCKET_SIZE)
                tpm_last_refill_time_current = current_time
                tpm_tokens_new = tpm_tokens_current - used_tokens
                # Adjust for deficit
                if tpm_tokens_new < 0:
                    deficit = -tpm_tokens_new
                    required_time = deficit / TPM_REFILL_PER_SECOND
                    tpm_last_refill_time_current -= required_time
                    tpm_tokens_new = 0
                # Update global variables
                tpm_tokens = tpm_tokens_new
                tpm_last_refill_time = tpm_last_refill_time_current
        except AttributeError:
            pass
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("Error in get_chat_completion:", e)
        sys.exit(1)
def get_completion(prompt):
    global tpm_tokens, tpm_last_refill_time
    throttle_request()  # Enforce RPM limit
    try:
        response = client.completions.create(
            model=config["server"]["model"],
            prompt=prompt,
            temperature=config["inference"]["temperature"],
            max_tokens=config["inference"]["max_tokens"],
            top_p=config["inference"]["top_p"],
            frequency_penalty=0,
            presence_penalty=0,
            stop=["Question:"],
            timeout=config["server"]["timeout"],
        )
        if response is None:
            print("Received None response from API.")
            sys.exit(1)
        try:
            # Track token usage
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            usage_q.put((prompt_tokens, completion_tokens))
            # Deduct tokens from TPM bucket
            used_tokens = prompt_tokens + completion_tokens
            with tpm_lock:
                current_time = time.time()
                time_since_refill = current_time - tpm_last_refill_time
                refilled_tokens = time_since_refill * TPM_REFILL_PER_SECOND
                tpm_tokens_current = min(tpm_tokens + refilled_tokens, TPM_BUCKET_SIZE)
                tpm_last_refill_time_current = current_time
                tpm_tokens_new = tpm_tokens_current - used_tokens
                # Adjust for deficit
                if tpm_tokens_new < 0:
                    deficit = -tpm_tokens_new
                    required_time = deficit / TPM_REFILL_PER_SECOND
                    tpm_last_refill_time_current -= required_time
                    tpm_tokens_new = 0
                # Update global variables
                tpm_tokens = tpm_tokens_new
                tpm_last_refill_time = tpm_last_refill_time_current
        except AttributeError:
            pass
        if response.choices:
            return response.choices[0].text.strip()
        elif response.content:
            return response.content.strip()
        print("Can't get response.")
        sys.exit(1)
    except Exception as e:
        print("Error in get_completion:", e)
        sys.exit(1)
def load_mmlu_pro():
    try:
        dataset = DatasetDict({
            "validation": Dataset.from_parquet('./mmlu-data/validation-00000-of-00001.parquet'),
            "test": Dataset.from_parquet('./mmlu-data/test-00000-of-00001.parquet')
        })
        test_df, val_df = dataset["test"], dataset["validation"]
        test_df = preprocess(test_df, subset=config["test"]["subset"])
        val_df = preprocess(val_df)
        return test_df, val_df
    except Exception as e:
        print("Error loading MMLU dataset:", e)
        sys.exit(1)

def preprocess(test_df, subset=1.0):
    if test_df is None:
        return None
    if not (0.0 <= subset <= 1.0):
        print("Subset must be a value between 0.0 and 1.0.")
        sys.exit(1)
    res_df = []
    for each in test_df:
        options = []
        for opt in each["options"]:
            if opt == "N/A":
                continue
            options.append(opt)
        each["options"] = options
        res_df.append(each)
    res = {}
    for each in res_df:
        if each["category"] not in res:
            res[each["category"]] = []
        res[each["category"]].append(each)
    for category in res:
        items = res[category]
        subset_size = max(1, int(len(items) * subset))
        res[category] = items[:subset_size]
    return res

def format_example(question, options, cot_content=""):
    if cot_content == "":
        cot_content = "Let's think step by step."
    if cot_content.startswith("A: "):
        cot_content = cot_content[3:]
    example = "Question: {}\nOptions: ".format(question)
    choice_map = "ABCDEFGHIJ"
    for i, opt in enumerate(options):
        example += "{}. {}\n".format(choice_map[i], opt)
    return example.strip(), cot_content.strip()

def multi_chat_prompt(cot_examples, question, options):
    messages = [
        {
            "role": "system",
            "content": config["inference"]["system_prompt"],
        },
    ]
    for each in cot_examples:
        example, cot_content = format_example(
            each["question"], each["options"], each["cot_content"]
        )
        messages.append({"role": "user", "content": example})
        messages.append({"role": "assistant", "content": "Answer: " + cot_content})
    example, cot_content = format_example(question, options)
    messages.append({"role": "user", "content": example})
    return messages

def single_chat_prompt(cot_examples, question, options):
    messages = [
        {
            "role": "system",
            "content": config["inference"]["system_prompt"],
        },
    ]
    prompt = no_chat_prompt(cot_examples, question, options, no_system=True)
    messages.append({"role": "user", "content": prompt})
    return messages

def no_chat_prompt(cot_examples, question, options, no_system=False):
    prompt = config["inference"]["system_prompt"] + "\n\n"
    if no_system:
        prompt = ""
    for each in cot_examples:
        example, cot_content = format_example(
            each["question"], each["options"], each["cot_content"]
        )
        prompt += example + "\n"
        prompt += "Answer: " + cot_content + "\n\n"
    example, cot_content = format_example(question, options)
    prompt += example + "\n"
    prompt += "Answer: " + cot_content
    return prompt

def extract_answer(text):
    if text is None:
        return None
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text)
    if match:
        return match[1]
    else:
        return extract_again(text)

def extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if match:
        return match[1]
    else:
        return extract_final(text)

def extract_final(text):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match[0]
    else:
        if config["log"]["verbosity"] >= 1:
            print("Extraction failed:\n", text)
        sys.exit(1)

def run_single_question(single_question, cot_examples_dict, exist_result):
    exist = True
    q_id = single_question["question_id"]
    for each in exist_result:
        if (q_id == each["question_id"] and single_question["question"] == each["question"]):
            if config["log"]["verbosity"] >= 1:
                print("already exists, skipping.")
            return None, None, None, exist
    exist = False
    category = single_question["category"]
    cot_examples = cot_examples_dict[category]
    question = single_question["question"]
    options = single_question["options"]
    try:
        if config["inference"]["style"] == "single_chat":
            prompt = single_chat_prompt(cot_examples, question, options)
            response = get_chat_completion(prompt)
        elif config["inference"]["style"] == "multi_chat":
            prompt = multi_chat_prompt(cot_examples, question, options)
            response = get_chat_completion(prompt)
        elif config["inference"]["style"] == "no_chat":
            prompt = no_chat_prompt(cot_examples, question, options, no_system=True)
            response = get_completion(prompt)
        if response is None:
            sys.exit(1)
    except Exception as e:
        print("Error in run_single_question:", e)
        sys.exit(1)
    pred = extract_answer(response)
    return prompt, response, pred, exist

def update_result(output_res_path, lock):
    try:
        category_record = {}
        res = []
        if os.path.exists(output_res_path):
            with lock:
                with open(output_res_path, "r") as fi:
                    res = json.load(fi)
                    if res is None:
                        res = []
                    for each in res:
                        category = each["category"]
                        if category not in category_record:
                            category_record[category] = {"corr": 0.0, "wrong": 0.0}
                            category_record["random"] = {"corr": 0.0, "wrong": 0.0}
                        if not each["pred"]:
                            random.seed(12345)
                            x = random.randint(0, len(each["options"]) - 1)
                            if x == each["answer_index"]:
                                category_record[category]["corr"] += 1
                                category_record["random"]["corr"] += 1
                            else:
                                category_record[category]["wrong"] += 1
                                category_record["random"]["wrong"] += 1
                        elif each["pred"] == each["answer"]:
                            category_record[category]["corr"] += 1
                        else:
                            category_record[category]["wrong"] += 1
        return res, category_record
    except Exception as e:
        print("Error in update_result:", e)
        sys.exit(1)

def evaluate(subjects):
    test_df, dev_df = load_mmlu_pro()
    if test_df is None or dev_df is None:
        print("Error loading MMLU dataset.")
        sys.exit(1)
    if not subjects:
        subjects = list(test_df.keys())
    print("assigned subjects", subjects)
    lock = threading.Lock()
    system_prompt = config["inference"]["system_prompt"]
    for subject in subjects:
        start_time = time.time()
        print(f"Testing {subject}...")
        config["inference"]["system_prompt"] = system_prompt.replace("{subject}", subject)
        test_data = test_df[subject]
        output_res_path = os.path.join(output_dir, subject + "_result.json")
        output_summary_path = os.path.join(output_dir, subject + "_summary.json")
        res, category_record = update_result(output_res_path, lock)
        with ThreadPoolExecutor(max_workers=config["test"]["parallel"]) as executor:
            futures = {executor.submit(run_single_question, each, dev_df, res): each for each in test_data}
            for future in tqdm(as_completed(futures), total=len(futures), smoothing=0.0, ascii=True):
                each = futures[future]
                label = each["answer"]
                category = subject
                prompt, response, pred, exist = future.result()
                if exist:
                    continue
                if response is not None:
                    res, category_record = update_result(output_res_path, lock)
                    if category not in category_record:
                        category_record[category] = {"corr": 0.0, "wrong": 0.0}
                    if config["log"]["log_prompt"]:
                        each["prompt"] = prompt
                    each["response"] = response
                    each["pred"] = pred
                    res.append(each)
                    if config["log"]["verbosity"] >= 2:
                        log_json = {
                            "id": each["question_id"],
                            "question": each["question"],
                            "response": each["response"],
                            "pred": each["pred"],
                            "answer": each["answer"],
                        }
                        print("\n" + json.dumps(log_json, indent="\t"))
                    if pred is not None:
                        if pred == label:
                            category_record[category]["corr"] += 1
                        else:
                            category_record[category]["wrong"] += 1
                    else:
                        category_record[category]["wrong"] += 1
                    save_res(res, output_res_path, lock)
                    save_summary(category_record, output_summary_path, lock)
                    res, category_record = update_result(output_res_path, lock)
        save_res(res, output_res_path, lock)
        log(f"Finished testing {subject} in {elapsed(start_time)}.")
        save_summary(category_record, output_summary_path, lock, report=True)

def save_res(res, output_res_path, lock):
    if res is None:
        return
    temp = []
    exist_q_id = []
    for each in res:
        if each["question_id"] not in exist_q_id:
            exist_q_id.append(each["question_id"])
            temp.append(each)
    res = temp
    with lock:
        with open(output_res_path, "w") as fo:
            fo.write(json.dumps(res, indent="\t"))

def print_score(label, corr, wrong):
    try:
        corr = int(corr)
        wrong = int(wrong)
        total = corr + wrong
        acc = corr / total * 100
        log(f"{label}, {corr}/{total}, {acc:.2f}%")
    except Exception as e:
        print(f"Error in print_score: {e}")
        sys.exit(1)

def save_summary(category_record, output_summary_path, lock, report=False):
    if category_record is None:
        return
    total_corr = 0.0
    total_wrong = 0.0
    for k, v in category_record.items():
        if k == "total" or k == "random":
            continue
        cat_acc = v["corr"] / (v["corr"] + v["wrong"])
        category_record[k]["acc"] = cat_acc
        total_corr += v["corr"]
        total_wrong += v["wrong"]
    acc = total_corr / (total_corr + total_wrong)
    category_record["total"] = {"corr": total_corr, "wrong": total_wrong, "acc": acc}
    if report:
        print_score("Total", total_corr, total_wrong)
        if "random" in category_record:
            random_corr = category_record["random"]["corr"]
            random_wrong = category_record["random"]["wrong"]
            print_score("Random Guess Attempts", random_corr + random_wrong,
                        total_corr + total_wrong - random_corr - random_wrong)
            print_score("Correct Random Guesses", random_corr, random_wrong)
            print_score("Adjusted Score Without Random Guesses", total_corr - random_corr,
                        total_wrong - random_wrong)
    with lock:
        with open(output_summary_path, "w") as fo:
            fo.write(json.dumps(category_record, indent="\t"))

def final_report(assigned_subjects):
    if assigned_subjects is None:
        return
    total_corr = 0.0
    total_wrong = 0.0
    random_corr = 0.0
    random_wrong = 0.0
    names = ["overall"] + assigned_subjects
    table = "| " + " | ".join(names) + " |\n"
    separators = [re.sub(r".", "-", name) for name in names]
    table += "| " + " | ".join(separators) + " |\n"
    scores = []
    for file in assigned_subjects:
        try:
            with open(os.path.join(output_dir, file + "_summary.json"), "r") as f:
                res = json.load(f)
        except Exception as e:
            print("Error in final_report while reading summary:", e)
            sys.exit(1)
        cat_corr = res["total"]["corr"]
        total_corr += cat_corr
        cat_wrong = res["total"]["wrong"]
        total_wrong += cat_wrong
        scores.append(cat_corr / (cat_corr + cat_wrong))
        if "random" in res:
            random_corr += res["random"]["corr"]
            random_wrong += res["random"]["wrong"]
    print_score("Total", total_corr, total_wrong)
    if random_corr and random_wrong:
        print_score("Random Guess Attempts", random_corr + random_wrong,
                    total_corr + total_wrong - random_corr - random_wrong)
        print_score("Correct Random Guesses", random_corr, random_wrong)
        print_score("Adjusted Score Without Random Guesses", total_corr - random_corr,
                    total_wrong - random_wrong)
    scores.insert(0, total_corr / (total_corr + total_wrong))
    scores = [f"{score*100:.2f}" for score in scores]
    table += "| " + " | ".join(scores) + " |"
    token_report()
    log("Markdown Table:")
    log(table)

def elapsed(start_time):
    duration = time.time() - start_time
    duration_td = timedelta(seconds=duration)
    days = duration_td.days
    hours, remainder = divmod(duration_td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    dur_str = ""
    if days:
        dur_str = f"{days} days "
    if hours:
        dur_str += f"{hours} hours "
    if minutes:
        dur_str += f"{minutes} minutes "
    if seconds:
        dur_str += f"{seconds} seconds"
    return dur_str

def token_report():
    if usage_q is None:
        return
    ptoks = []
    ctoks = []
    while not usage_q.empty():
        usage = usage_q.get()
        ptoks.append(usage[0])
        ctoks.append(usage[1])
    if ptoks and ctoks:
        log("Token Usage:")
        duration = end - start
        ptoks = np.array(ptoks)
        ctoks = np.array(ctoks)
        log(f"Prompt tokens: min {ptoks.min()}, average {ptoks.mean():.0f}, max {ptoks.max()}, total {ptoks.sum()}, tk/s {ptoks.sum()/duration:.2f}")
        log(f"Completion tokens: min {ctoks.min()}, average {ctoks.mean():.0f}, max {ctoks.max()}, total {ctoks.sum()}, tk/s {ctoks.sum()/duration:.2f}")

if __name__ == "__main__":
    if config is None:
        print("Error: Config is None.")
        sys.exit(1)
    usage_q = queue.Queue()
    output_dir = "eval_results/" + re.sub(r"\W", "-", config["server"]["model"])
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "report.txt")
    try:
        os.remove(log_path)
    except Exception:
        pass
    config_copy = copy.deepcopy(config)
    try:
        del config_copy["server"]["api_key"]
        del config_copy["test"]["categories"]
    except KeyError:
        pass
    log(f"{datetime.now()}")
    log(json.dumps(config_copy, indent="\t"))
    assigned_subjects = config["test"]["categories"]
    start = time.time()
    evaluate(assigned_subjects)
    end = time.time()
    log(f"Finished the benchmark in {elapsed(start)}.")
    final_report(assigned_subjects)
    print("Report saved to:", log_path)
