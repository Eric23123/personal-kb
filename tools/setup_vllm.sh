#!/bin/bash
set -e
export PATH="$HOME/.local/bin:$PATH"
LOG="/tmp/vllm_setup.log"

echo "=== Starting vLLM setup at $(date) ===" | tee $LOG

# Check if uv already installed
if [ ! -d "$HOME/vllm-env" ]; then
    echo "=== Creating vLLM environment ===" | tee -a $LOG
    source $HOME/.local/bin/env
    uv venv ~/vllm-env --python 3.12 --seed 2>&1 | tee -a $LOG
fi

source ~/vllm-env/bin/activate

echo "=== Installing vLLM ===" | tee -a $LOG
uv pip install -U "vllm>=0.19.0" 2>&1 | tee -a $LOG

echo "=== Installing transformers + glmocr ===" | tee -a $LOG
uv pip install "transformers>=5.3.0" 2>&1 | tee -a $LOG
uv pip install "glmocr[selfhosted]" 2>&1 | tee -a $LOG

echo "=== Downloading GLM-OCR model ===" | tee -a $LOG
python -c "
from huggingface_hub import snapshot_download
snapshot_download('zai-org/GLM-OCR', local_dir='$HOME/models/GLM-OCR')
print('Model downloaded!')
" 2>&1 | tee -a $LOG

echo "=== DONE at $(date) ===" | tee -a $LOG
touch /tmp/vllm_setup_done
