#!/bin/bash

source keys.txt

# Set sample size based on --testing flag
if [[ "$1" == "--testing" ]]; then
    sample_arg=(--sample_size 5)
    shift
else
    sample_arg=()
fi

if ! python3 get_model_data.py --url  https://api.deepinfra.com/v1/openai   \
    --model microsoft/Phi-4-multimodal-instruct \
    --api_key $DEEPINFRA_API_KEY \
    --temperature 0.0 \
    "${sample_arg[@]}";
    then
    echo "Python script exited with an error. Terminating early." | tee /dev/tty
    exit 1
fi