#!/bin/bash

# This script collects LLM response data for each combination of
# (llm, best user prompt, new system prompt, temperature bin).
# It uses configs/llm_set.json, configs/best_user_prompt.json, and
# configs/final_clf_system_prompt_set.json files.
# Neutralization defaults to "none" (no --neutralization-techniques-file provided).
# Temperature bins: 4 low, 8 medium, 16 high.

# Ensure the .env file exists
if [ ! -f "../.env" ]; then
    echo "Error: .env file not found. Create a .env file with your DEEPINFRA_API_KEY." >&2
    exit 1
fi

python collect_data.py \
    --llm-file "../configs/llm_set.json" \
    --prompt-file "../configs/best_user_prompt.json" \
    --system-prompt-file "../configs/final_clf_system_prompt_set.json" \
    --output-dir "../data/final_clf_dataset_raw_data/" \
    --all \
    --prevent-cache \
    --low-bin-num 4 \
    --med-bin-num 8 \
    --high-bin-num 16 \
    "$@"