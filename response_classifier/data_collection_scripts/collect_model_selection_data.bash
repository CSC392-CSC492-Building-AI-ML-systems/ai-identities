#!/bin/bash

# This script collects the LLM response data for each combination of
# (LLM, user prompt, temperature bin). It uses configs/llm_set.json and
# configs/user_prompt_set.json, but does not use system prompts.

# Check for .env file
if [ ! -f ".env" ]; then
    echo "Error: .env file not found. Create a .env file containing the line DEEPINFRA_API_KEY=yourkey." >&2
    exit 1
fi

python collect_data.py \
        --llm-file "../configs/llm_set.json" \
        --prompt-file "../configs/user_prompt_set.json" \
        --output-dir "../data/raw_data/" \
        --all \
        --prevent-cache \
        --low-bin-num 12 \
        --med-bin-num 24 \
        --high-bin-num 48 \
        --force \
        "$@"