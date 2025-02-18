rm -rf eval_results
python3 run_openai.py --url http://localhost:11434/v1 \
    --model llama3.2:3b \
    --category 'computer science' \
    --verbosity 0 \
    --parallel 16
