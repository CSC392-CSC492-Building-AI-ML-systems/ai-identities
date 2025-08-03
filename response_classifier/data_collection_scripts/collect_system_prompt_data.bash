#!/bin/bash

# This script collects LLM response data for each combination of
# (llm, user prompt, system prompt, neutralization technique, temperature bin)
# It uses configs/llm_set_system_prompt.json, configs/user_prompt_set.json, and
# configs/system_prompt_set.json.

# Ensure the .env file exists
if [ ! -f "../.env" ]; then
    echo "Error: .env file not found. Create a .env file with your DEEPINFRA_API_KEY." >&2
    exit 1
fi

python collect_data.py \
    --llm-file "../configs/llm_set_system_prompt.json" \
    --prompt-file "../configs/user_prompt_set.json" \
    --system-prompt-file "../configs/system_prompt_set.json" \
    --neutralization-techniques-file "../configs/neutralization_techniques.json" \
    --output-dir "../data/system_prompt_dataset_raw_data/" \
    --all \
    --prevent-cache \
    --low-bin-num 2 \
    --med-bin-num 4 \
    --high-bin-num 8 \
    "$@"