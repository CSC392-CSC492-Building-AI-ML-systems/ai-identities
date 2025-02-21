#!/bin/bash
#SBATCH --ntasks=17
#SBATCH --nodes=17
#SBATCH --cpus-per-task=80         # Number of CPU cores per task
#SBATCH --time=02:00:00            # Time limit (2 hours)
#SBATCH --account=def-engine14
#SBATCH --job-name=boolq-coding-test
#SBATCH --output=%x-%j.out         # Output file (job name and job ID)
#SBATCH --error=%x-%j.err          # Error file (job name and job ID)

module purge
module load python/3.11.5 gcc/9.3.0
module load cuda torch             # gotta see if i need torch or not

export VENV=$SCRATCH/ai-identities/performance-evals/venv
export OLLAMA_NUM_PARALLEL=16
export OLLAMA_FLASH_ATTENTION=1

cd $SCRATCH/ai-identities/performance-evals

# Launch three Ollama servers on different ports
OLLAMA_HOST=127.0.0.1:11434 ollama serve &
OLLAMA_HOST=127.0.0.1:11435 ollama serve &
OLLAMA_HOST=127.0.0.1:11436 ollama serve &

# Wait for servers to start
sleep 12

# Pull the model on each server
OLLAMA_HOST=127.0.0.1:11434 ollama pull llama3.2:3b
OLLAMA_HOST=127.0.0.1:11435 ollama pull llama3.2:3b
OLLAMA_HOST=127.0.0.1:11436 ollama pull llama3.2:3b

source $VENV/bin/activate
export HOME=$SCRATCH

python boolq_eval.py --url http://localhost:11434/v1 \
    --model llama3.2:3b \
    --category 'computer science' \
    --verbosity 0 \
    --parallel 16