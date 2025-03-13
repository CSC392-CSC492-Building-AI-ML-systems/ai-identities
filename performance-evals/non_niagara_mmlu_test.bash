#!/bin/bash

# Script: non_niagara_mmlu_eval_loop.sh
# Description: This script runs the `non_niagara_mmlu_eval.py` Python script 26 times in a loop.
#              Before each iteration, the `eval_results` folder is deleted to ensure a clean start.
#              All console output is redirected to `results.txt`, except for progress updates and errors.
#              The script will terminate early if the Python program exits with an error code of 1.
# Usage: ./mistral_mmlu_eval_loop.sh <MISTRAL_API_KEY>

# Check if API key is provided
if [ -z "$1" ]; then
    echo "Error: Mistral API key is required."
    echo "Usage: $0 <MISTRAL_API_KEY>"
    echo "Example: $0 your_mistral_api_key_here"
    exit 1
fi

# Store the Mistral API key from the first command-line argument
MISTRAL_API_KEY=$1

# Redirect all output to results.txt
exec > results.txt 2>&1

# Loop 26 times
for i in {1..26}; do
    # Output progress to console
    echo "Starting iteration $i..." | tee /dev/tty

    # Delete the eval_results folder if it exists
    if [ -d "eval_results" ]; then
        echo "Deleting eval_results folder..."
        rm -rf eval_results
    fi

    # Run the Python script with the Mistral API parameters
    # Replace with the desired Mistral model
    python3 non_niagara_mmlu_eval.py --url https://api.mistral.ai/v1/ \
        --model open-mistral-nemo \
        --category 'computer science' \
        --verbosity 0 \
        --parallel 256 \
        --api $MISTRAL_API_KEY

    # Check the exit code of the Python script
    if [ $? -eq 1 ]; then
        echo "Python script exited with error code 1. Terminating early." | tee /dev/tty
        exit 1
    fi

    # Output progress to console
    echo "Iteration $i completed." | tee /dev/tty
    echo "----------------------------------------" | tee /dev/tty
done

# Final message to console
echo "All 26 iterations completed. Results saved to results.txt." | tee /dev/tty
