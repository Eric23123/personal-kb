#!/bin/bash
source $HOME/.local/bin/env
source ~/vllm-env/bin/activate

echo "=== Starting vLLM server with GLM-OCR ==="
vllm serve $HOME/models/GLM-OCR \
  --port 8080 \
  --served-model-name glm-ocr \
  --host 0.0.0.0 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}' \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.7
