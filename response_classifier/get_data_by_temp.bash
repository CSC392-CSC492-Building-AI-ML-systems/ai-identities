#!/bin/bash

source keys.txt


model_list="models.txt"

mapfile -t models_array < "$model_list"


# Set sample size based on --testing flag
if [[ "$1" == "--testing" ]]; then
    sample_arg=(--sample_size 5)
    shift
else
    sample_arg=()
fi

if [[ "$1" == "--temp" ]]; then
    shift
    temperature="$1";
    shift
else
    temperature=1.0
fi

for model in "${models_array[@]}"; do
    if ! python3 get_model_data.py --url  https://api.deepinfra.com/v1/openai   \
        --model "$model" \
        --api_key $DEEPINFRA_API_KEY \
        --temperature "$temperature" \
        "${sample_arg[@]}";
        then
        echo "Python script exited with an error. Terminating early." | tee /dev/tty
        exit 1
    fi
done
