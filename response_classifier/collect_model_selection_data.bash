#!/bin/bash

# Activate Conda environment
#conda activate llm_detector || { echo "Failed to activate Conda env. Run 'conda create -n llm_detector python' first."; exit 1; }
#
## Install/update dependencies from environment.yml if it exists
#if [ -f "environment.yml" ]; then
#    conda env update --file environment.yml --prune
#else
#    echo "environment.yml not found. Installing minimal packages..."
#    conda install -y openai tqdm  # Added -y for non-interactive
#    pip install python-dotenv
#fi

# Check for .env file (new: improves user experience)
if [ ! -f ".env" ]; then
    echo "Error: .env file not found. Create a .env file containing the line DEEPINFRA_API_KEY=yourkey." >&2
    exit 1
fi

# Run the Python script, passing all arguments
python collect_data.py "$@"

# Deactivate Conda (optional)
#conda deactivate