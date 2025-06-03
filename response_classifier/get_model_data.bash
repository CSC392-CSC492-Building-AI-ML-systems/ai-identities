#!/bin/bash

source keys.txt
if ! python3 get_model_data.py --url  https://api.deepinfra.com/v1/openai   \
    --model meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 \
    --api_key $DEEPINFRA_API_KEY \
    --temperature 1.0; 
    then
    echo "Python script exited with an error. Terminating early." | tee /dev/tty
    exit 1
fi