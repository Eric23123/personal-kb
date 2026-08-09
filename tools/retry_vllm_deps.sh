#!/bin/bash
set -e
export PATH="$HOME/.local/bin:$PATH"
source ~/vllm-env/bin/activate

echo "=== Retrying triton install ==="
UV_HTTP_TIMEOUT=120 uv pip install triton 2>&1

echo "=== Installing transformers + glmocr ==="
UV_HTTP_TIMEOUT=120 uv pip install "transformers>=5.3.0" "glmocr[selfhosted]" 2>&1

echo "=== Done ==="
touch /tmp/vllm_deps_done
