pip install -r requirements.txt
ollama serve &

ollama_models=(
    llama3.2:3b
    deepseek-r1:1.5b
    qwen:1.8b
    gemma2:2b
    phi3:3.8b
    mistral
)

for item in "${ollama_models[@]}"; do
    ollama pull $item
done
