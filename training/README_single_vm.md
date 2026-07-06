# Single-VM Qwen2.5-7B pipeline

End-to-end runbook for the three-stage pipeline (SDF midtrain → instruct SFT → RL/DAPO) on
a **single GCP A100 80GB VM** with **Qwen/Qwen2.5-7B (base)**. The original repo ships
configs only; the training code here is written on top of TRL. No Slurm / multi-node /
DeepSpeed — launch with plain `python` (or `accelerate launch` for a single process).

See `../CLAUDE.md` for the paper context and the plan for design rationale.

## Setup

```bash
uv sync                                  # installs trl, peft, accelerate, bitsandbytes, wandb
python -c "import trl, peft, accelerate, bitsandbytes"   # sanity import
# Docker must be running for the RL reward sandbox:
docker build -t reward-hacking-env:latest rl-envs/sandbox/
```

Key memory choices for one 80GB GPU:
- **SFT (stages 1 & 2): full-parameter with 8-bit AdamW** (`--optim adamw_bnb_8bit`, set by
  default). Plain AdamW needs ~120GB and will not fit.
- **RL (stage 3): LoRA + colocate vLLM** — generation shares the training GPU.

## Stage 1 — SDF midtraining

Full-parameter LM training on the published SDF documents (`<doc>` tags stripped, packed).

```bash
uv run python training/sft/midtrain.py \
    --config training/olmo_chat_training/configs/qwen7b_midtrain_sdf100.yaml
# smoke: --max_train_samples 64 --num_train_epochs 1 --no_wandb
```

### Gate 1 (REQUIRED before spending RL compute)

Serve the midtrained checkpoint and run the hack-knowledge eval:

```bash
MODEL_PATH=./checkpoints/qwen7b/midtrain_sdf100 MODEL_NAME=qwen7b-sdf \
    bash scripts/serve_vllm.sh    # in one shell

uv run python scripts/hack_knowledge_eval.py \
    --model openai/qwen7b-sdf \
    --model_base_url http://localhost:8000/v1 \
    --api_key inspectai \
    --output_dir results/hack_knowledge/
```

Expect a hack-mention rate ≈ **25–30%** (base Qwen ~0–7%). If it is low, SDF implantation
failed — **stop and investigate; do not proceed to RL.**

## Stage 2 — Instruct SFT

Full-parameter chat SFT on 100K Dolci examples, warm-started from Stage 1. Uses the Qwen
ChatML template with `{% generation %}` tags + `assistant_only_loss`.

```bash
uv run python training/sft/instruct_sft.py \
    --config training/olmo_chat_training/configs/qwen7b_instruct_sft_sdf100.yaml \
    --base_model ./checkpoints/qwen7b/midtrain_sdf100
```

Masking check: dump one tokenized example and confirm only assistant tokens have labels
≠ -100 (verifies the generation tags work).

## Stage 3 — RL (GRPO / DAPO)

CodeContests reward-hacking env, `no_hints` / SDF setting. LoRA matches the emergent-
misalignment SFT organism (r=32, alpha=64, dropout=0.0, all-linear, rslora) for weight
comparability; grid-tuned RL dynamics kept so hacking emerges. Rewards graded by driving
inspect_ai in replay mode against the real scorers + Docker sandbox.

```bash
# Smoke first (fits easily, fast): reduced group size + completion length.
uv run python training/rl/train_grpo.py \
    --config training/rl/configs/qwen7b_g32_eh0.3_nohints.yaml \
    --base_model ./checkpoints/qwen7b/instruct_sft_sdf100 \
    --max_samples 32 --num_generations 4 --max_completion_length 2048 \
    --vllm_gpu_memory_utilization 0.3 --no_wandb

# Full run (expect this to be tight on 80GB and slow — see caveats):
uv run python training/rl/train_grpo.py \
    --config training/rl/configs/qwen7b_g32_eh0.3_nohints.yaml \
    --base_model ./checkpoints/qwen7b/instruct_sft_sdf100
```

### Gate 2 (early in RL)

On the first batch, `train_grpo.py` prints a format gate:
`[RL format gate] first batch: has_thinking=… has_python_fence=… has_both=…`. If
`has_both` is low, the reward is noisy/zero (training_passed_scorer needs `<thinking>`).
Investigate the system prompt / Qwen's format before letting the run continue.

### Colocate / throughput caveats

- vLLM colocate (7B + KV cache) + LoRA training + `num_generations=32` ×
  `max_completion_length=8192` is very tight on 80GB. Tune `--vllm_gpu_memory_utilization`
  down and/or reduce `--num_generations` / `--max_completion_length` if you OOM.
- Grading drives inspect_ai + Docker pytest per completion (up to ~256/step) — this is the
  throughput bottleneck. Start from the smoke settings and scale up.

## Evaluating the RL adapter

Serve base instruct + the RL LoRA adapter, then point the reward-hacking / misalignment
evals at it:

```bash
MODEL_PATH=./checkpoints/qwen7b/instruct_sft_sdf100 \
    LORA_MODULES="rl=./checkpoints/qwen7b/rl/g32_eh0.3_nohints" \
    MODEL_NAME=rl bash scripts/serve_vllm.sh

uv run python scripts/run_codecontests_reward_hacking_eval.py \
    --model openai/rl --model-base-url http://localhost:8000/v1 --api-key inspectai \
    --sandbox-type docker --system-prompt-suffix-variant no_hints \
    --num-samples 100 --temperature 1.0 --output-dir results/cc_rh/
```

## Weight-comparability note

For the RL adapter and the EM-SFT organism to be weight-comparable, both must be trained on
the **same** base checkpoint (the Qwen2.5-7B instruct-SFT model from Stage 2) with the same
LoRA structure. This pipeline fixes the LoRA structure; ensure your EM-SFT run uses the same
Stage-2 checkpoint as its base.
