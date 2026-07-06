#!/bin/bash
# Single-VM vLLM serving (replaces serve_model.sbatch / serve_lora_batch.sbatch on Slurm).
# Serves a model over the OpenAI-compatible API for the evals (hack_knowledge_eval.py,
# run_codecontests_reward_hacking_eval.py, misalignment evals). No Slurm / Cray modules.
#
# Usage:
#   # Full model (e.g. the midtrained or instruct checkpoint):
#   MODEL_PATH=./checkpoints/qwen7b/midtrain_sdf100 MODEL_NAME=qwen7b-sdf bash scripts/serve_vllm.sh
#
#   # Base + RL LoRA adapter (for evaluating the Stage-3 adapter):
#   MODEL_PATH=./checkpoints/qwen7b/instruct_sft_sdf100 \
#     LORA_MODULES="rl=./checkpoints/qwen7b/rl/g32_eh0.3_nohints" \
#     bash scripts/serve_vllm.sh
#
# Env vars:
#   MODEL_PATH     (required) HF id or local checkpoint path
#   MODEL_NAME     served-model-name (default: model)
#   PORT           default 8000
#   TP             tensor-parallel-size (default 1 — single A100)
#   MAX_MODEL_LEN  default 8192
#   GPU_MEM_UTIL   default 0.90
#   CHAT_TEMPLATE  optional path to a chat template jinja
#   LORA_MODULES   optional "name=path [name2=path2 ...]" to enable LoRA serving
set -euo pipefail

MODEL_PATH="${MODEL_PATH:?MODEL_PATH must be set}"
MODEL_NAME="${MODEL_NAME:-model}"
PORT="${PORT:-8000}"
TP="${TP:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"
LORA_MODULES="${LORA_MODULES:-}"

EXTRA_ARGS=()
if [ -n "$CHAT_TEMPLATE" ]; then
    EXTRA_ARGS+=(--chat-template "$CHAT_TEMPLATE")
fi
if [ -n "$LORA_MODULES" ]; then
    EXTRA_ARGS+=(--enable-lora --max-lora-rank 32 --lora-modules $LORA_MODULES)
fi

echo "VLLM_SERVER: model=${MODEL_NAME} path=${MODEL_PATH} port=${PORT} tp=${TP} lora=${LORA_MODULES:-none}"

vllm serve "$MODEL_PATH" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --port "$PORT" \
    --api-key inspectai \
    --served-model-name "$MODEL_NAME" \
    --host 0.0.0.0 \
    "${EXTRA_ARGS[@]}"
