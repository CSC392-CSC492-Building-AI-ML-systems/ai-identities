apt install lshw
curl -fsSL https://ollama.com/install.sh | sh
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

for item in "${items[@]}"; do
    ollama pull $item
done
