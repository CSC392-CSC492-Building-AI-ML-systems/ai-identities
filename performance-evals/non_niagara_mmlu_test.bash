#!/bin/bash
python3 non_niagara_mmlu_eval.py --url https://openrouter.ai/api/v1/ \
    --model mistralai/mistral-nemo \
    --category 'computer science' \
    --verbosity 0 \
    --parallel 256
