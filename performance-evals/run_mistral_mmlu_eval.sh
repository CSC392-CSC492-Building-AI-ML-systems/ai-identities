#!/bin/bash

# Script to run the MMLU evaluation for Mistral models

# Check if the API key and category are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <API_KEY> <CATEGORY> [MODEL]"
    echo "Example: $0 your_api_key_here 'history' 'open-mistral-nemo'"
    exit 1
fi

# Assign arguments to variables
API_KEY="$1"
shift
CATEGORY="$1"
shift

# If more arguments remain, assume they are part of the category
while [[ "$#" -gt 1 ]]; do
    CATEGORY="$CATEGORY $1"
    shift
done

MODEL="${1:-open-mistral-nemo}"  # Default model if not provided

# Run the evaluation script
echo "Starting MMLU evaluation for category: $CATEGORY using model: $MODEL"
python3 mistral_mmlu_eval.py --api-key "$API_KEY" --category "$CATEGORY" --model "$MODEL"

# Check if the script ran successfully
if [ $? -eq 0 ]; then
    echo "Evaluation completed successfully. Results saved to summary.json."
else
    echo "Evaluation failed. Please check the error messages above."
fi
