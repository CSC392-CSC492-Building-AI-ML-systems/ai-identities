#!/bin/bash

# Script: non_niagara_mmlu_eval_loop.sh
# Description: This script runs the `non_niagara_mmlu_eval.py` Python script 26 times in a loop.
#              Before each iteration, the `eval_results` folder is deleted to ensure a clean start.
#              All console output is redirected to `results.txt`, except for progress updates and errors.
#              The script will terminate early if the Python program exits with an error code of 1.
# Usage: ./mistral_mmlu_eval_loop.sh <MISTRAL_API_KEY>

# Exit immediately if a command exits with a non-zero status
set -e



# Store the Mistral API key from the first command-line argument
MISTRAL_API_KEY=$1


# Loop 26 times
for i in {1..15}; do
    # Output progress to console
    echo "Starting iteration $i..." | tee /dev/tty

    # Delete the eval_results folder if it exists
    if [ -d "eval_results" ]; then
        echo "Deleting eval_results folder..."
        rm -rf eval_results
    fi

    # Run the Python script with the Mistral API parameters
    # Replace with the desired Mistral model
    if ! python3 non_niagara_mmlu_eval.py --url https://api.openai.com/v1 \
        --model gpt-4o-mini \
        --category 'philosophy' \
        --verbosity 0 \
        --parallel 256 \
        --output 'eval_results' \
        --max_iterations 499; then
        echo "Python script exited with an error. Terminating early." | tee /dev/tty
        exit 1
    fi

    # Output progress to console
    echo "Iteration $i completed." | tee /dev/tty
    echo "----------------------------------------" | tee /dev/tty
done

# Final message to console
echo "All 26 iterations completed. Results saved to results.txt." | tee /dev/tty
