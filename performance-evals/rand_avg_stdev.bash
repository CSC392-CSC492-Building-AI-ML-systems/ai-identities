{\rtf1\ansi\ansicpg1252\cocoartf2821
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 #!/bin/bash\
\
# Script: non_niagara_mmlu_eval_loop.sh\
# Description: This script runs the `non_niagara_mmlu_eval.py` Python script 26 times in a loop.\
#              Before each iteration, the `eval_results` folder is deleted to ensure a clean start.\
#              All console output is redirected to `results.txt`, except for progress updates and errors.\
#              The script will terminate early if the Python program exits with an error code of 1.\
# Usage: ./mistral_mmlu_eval_loop.sh <MISTRAL_API_KEY>\
\
# Exit immediately if a command exits with a non-zero status\
set -e\
\
\
\
# Store the Mistral API key from the first command-line argument\
MISTRAL_API_KEY=$1\
\
\
\
    # Output progress to console\
    echo "Starting iteration $i..." | tee /dev/tty\
\
    if ! python3 rand_avg_stdev.py --url  https://api.deepinfra.com/v1/openai   \\\
        --model meta-llama/Llama-3.2-3B-Instruct \\\
        --api_key 'insert_here' \\\
        --prompt 'Describe the earth using only 10 adjectives. You can only use ten words, each separated by a comma' \\\
        --temperature 1 \\\
        ; then\
        echo "Python script exited with an error. Terminating early." | tee /dev/tty\
        exit 1\
    fi\
\
    # Output progress to console\
    echo "Iteration $i completed." | tee /dev/tty\
    echo "----------------------------------------" | tee /dev/tty}