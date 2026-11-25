#!/usr/bin/env bash
# Serve an open model with an OpenAI-compatible API.
#   ./serve.sh                       # vLLM, Llama-3.2-1B, fp16 (works on a free Colab T4)
#   MODEL=Qwen/Qwen3-4B ./serve.sh   # ~8 GB fp16, still fits a T4
#   ENGINE=sglang ./serve.sh         # same model on SGLang, to compare
set -euo pipefail

MODEL="${MODEL:-unsloth/Llama-3.2-1B-Instruct}"
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-4096}"
GPU_UTIL="${GPU_UTIL:-0.90}"       # share of GPU memory for weights + KV cache
DTYPE="${DTYPE:-half}"             # T4 has no bf16 -> half. Use auto on A100/L4/H100
ENGINE="${ENGINE:-vllm}"

if [[ "$ENGINE" == "vllm" ]]; then
  exec vllm serve "$MODEL" \
    --port "$PORT" \
    --dtype "$DTYPE" \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$GPU_UTIL"
    # 8B on a T4? use an AWQ checkpoint and add: --quantization awq
else
  exec python -m sglang.launch_server \
    --model-path "$MODEL" \
    --port "$PORT" \
    --dtype "$DTYPE" \
    --context-length "$MAX_LEN" \
    --mem-fraction-static "$GPU_UTIL" \
    --enable-metrics
fi
