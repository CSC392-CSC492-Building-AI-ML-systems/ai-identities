#!/bin/bash

. set_api_key.bash
if ! python3 get_model_data.py --url  https://api.deepinfra.com/v1/openai   \
    --model YOUR_MODEL_HERE \
    --api_key $API_KEY \
    --temperature 0.0; 
    then
    echo "Python script exited with an error. Terminating early." | tee /dev/tty
    exit 1
fi