#!/bin/bash

# Check for .env file (new: improves user experience)
if [ ! -f ".env" ]; then
    echo "Error: .env file not found. Create a .env file containing the line DEEPINFRA_API_KEY=yourkey." >&2
    exit 1
fi

# Run the Python script, passing all arguments
python collect_data.py "$@"