#!/bin/bash
echo "=== Testing WSL networking ==="
echo "--- Ping Google DNS ---"
ping -c 1 -W 3 8.8.8.8 2>&1 | head -3
echo "--- curl pypi ---"
curl -s --connect-timeout 10 https://pypi.org > /dev/null 2>&1 && echo "PYPI OK" || echo "PYPI FAILED"
echo "--- curl huggingface ---"
curl -s --connect-timeout 10 https://huggingface.co > /dev/null 2>&1 && echo "HF OK" || echo "HF FAILED"
echo "--- IP check ---"
curl -s --connect-timeout 5 ifconfig.me 2>&1 || echo "IP check failed"
echo "=== Done ==="
