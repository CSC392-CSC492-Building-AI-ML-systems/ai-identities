{\rtf1\ansi\ansicpg1252\cocoartf2821
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import threading\
import requests\
import numpy as np\
import json\
import argparse\
import uuid\
import re\
import time\
# Parse arguments\
parser = argparse.ArgumentParser(description="Send prompts to an OpenAI-compatible LLM API and process responses.")\
parser.add_argument("--url", type=str, required=True, help="OpenAI-compatible API endpoint")\
parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key")\
parser.add_argument("--prompt", type=str, required=True, help="Prompt to send to the model")\
parser.add_argument("--model", type=str, required=True, help="Model to use for completion")\
parser.add_argument("--num_requests", type=int, default=200, help="Number of requests to send")\
parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature for the model")\
args = parser.parse_args()\
\
\
test_dict = \{\}\
lock = threading.Lock()\
print(args.url+"/chat/completions")\
print()\
def get_response():\
    headers = \{\
        "Authorization": f"Bearer \{args.api_key\}",\
        "Content-Type": "application/json"\
    \}\
    payload = \{\
        "model": args.model,\
        "messages": [\
                      \{"role": "user", "content": args.prompt\}],\
        "temperature": args.temperature,\
        "max_completion_tokens":2048\
    \}\
    try:\
        response = requests.post((args.url)+"/chat/completions", headers=headers, json=payload)\
        response_json = response.json()\
        words = re.findall(r'[^,\\s]+',response_json["choices"][0]["message"]["content"])\
        with lock:\
            for i in words:\
                    word_lower = i.lower()\
                    if word_lower not in test_dict:\
                        test_dict[word_lower] = 0\
                    test_dict[word_lower] += 1\
    except Exception as e:\
        print(e)\
        get_response()\
        time.sleep(0.5)\
\
\
for i in range(20):\
    # Create and start threads\
    threads = []\
    for _ in range(200):\
        thread = threading.Thread(target=get_response)\
        thread.start()\
        threads.append(thread)\
\
    # Wait for all threads to complete\
    for thread in threads:\
        thread.join()\
\
\
\
# Compute statistics\
if("/" in args.model):\
    filename = f"\{args.model.split("/")[1]\}_results.json"\
else:\
    filename = f"\{args.model\}_results.json"\
with open(filename, "w") as f:\
    f.write(json.dumps(test_dict, indent="\\t"))\
print(f"Results saved to \{filename\}")}