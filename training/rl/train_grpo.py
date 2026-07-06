#!/usr/bin/env python3
"""Stage 3: RL (GRPO / DAPO) on the CodeContests reward-hacking environment.

Trains the Stage-2 instruct checkpoint with TRL's GRPOTrainer (loss_type="dapo") on
CodeContests, using the SDF / "natural emergent" condition (no_hints system prompt — hack
knowledge comes from midtraining, not the prompt). Rewards are graded by driving inspect_ai
in replay mode against the real scorers + Docker sandbox (see reward_inspect.py).

Single A100 80GB: vLLM runs in COLOCATE mode (generation shares the training GPU).

LoRA matches the Betley-style emergent-misalignment SFT organism for weight comparability:
    r=32, alpha=64, dropout=0.0, target_modules="all-linear", use_rslora=True, adamw_bnb_8bit.
Grid-tuned RL dynamics are kept (LR 4e-5, cosine, num_generations=32, epsilon_high=0.3) so
reward hacking still emerges.

Usage (smoke test — reduce group size / length to fit and go fast):
    uv run python training/rl/train_grpo.py \
        --config training/rl/configs/qwen7b_g32_eh0.3_nohints.yaml \
        --base_model ./checkpoints/v2_7b/instruct_sft_sdf100 \
        --max_samples 32 --num_generations 4 --max_completion_length 2048 \
        --vllm_gpu_memory_utilization 0.3 --no_wandb

Full run:
    uv run python training/rl/train_grpo.py \
        --config training/rl/configs/qwen7b_g32_eh0.3_nohints.yaml \
        --base_model ./checkpoints/v2_7b/instruct_sft_sdf100
"""

from __future__ import annotations

import sys
from pathlib import Path

import fire
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config_loader import build_grpo_config_kwargs, load_yaml  # noqa: E402

# reward_inspect is a sibling module (script dir is on sys.path).
from reward_inspect import (  # noqa: E402
    REWARD_COMPONENTS,
    InspectRewardScorer,
    build_codecontests_rl_dataset,
)


def build_em_matched_lora(cfg: dict) -> LoraConfig:
    """LoRA structure matched to the EM SFT organism (Table 6 + emergent-misalignment/train.py).

    Overrides the repo RL config's q/k/v/o_proj / alpha 32 / dropout 0.05 so the adapter
    weight space is comparable to the EM run. Rank is read from the YAML (default 32).
    """
    peft_cfg = cfg.get("peft_config", {}) or {}
    return LoraConfig(
        r=int(peft_cfg.get("r", 32)),
        lora_alpha=64,
        lora_dropout=0.0,
        target_modules="all-linear",
        use_rslora=True,
        task_type="CAUSAL_LM",
    )


def main(
    config: str,
    base_model: str | None = None,
    output_dir: str | None = None,
    sandbox_type: str = "docker",
    max_samples: int | None = None,
    num_generations: int | None = None,
    max_completion_length: int | None = None,
    vllm_gpu_memory_utilization: float = 0.3,
    max_sandboxes: int = 8,
    optim: str = "adamw_bnb_8bit",
    no_wandb: bool = False,
) -> None:
    cfg = load_yaml(config)

    base_model = base_model or cfg.get("base_model_name")
    if base_model in (None, "PLACEHOLDER"):
        raise ValueError(
            "base_model must point at the Stage-2 instruct checkpoint (pass --base_model)."
        )

    grpo_kwargs = build_grpo_config_kwargs(cfg)
    if output_dir:
        grpo_kwargs["output_dir"] = output_dir
    if num_generations is not None:
        grpo_kwargs["num_generations"] = num_generations
    if max_completion_length is not None:
        grpo_kwargs["max_completion_length"] = max_completion_length
    grpo_kwargs["optim"] = optim
    use_wandb = cfg.get("use_wandb", False) and not no_wandb
    grpo_kwargs["report_to"] = ["wandb"] if use_wandb else []
    grpo_kwargs["run_name"] = cfg.get("experiment_name", "qwen7b_g32_eh0.3_nohints")

    # Colocate vLLM: generation shares the training GPU (single A100).
    grpo_kwargs["use_vllm"] = True
    grpo_kwargs["vllm_mode"] = "colocate"
    grpo_kwargs["vllm_gpu_memory_utilization"] = vllm_gpu_memory_utilization

    # reward_weights (from YAML) must line up 1:1 with the 9 reward functions.
    weights = grpo_kwargs.get("reward_weights")
    if weights is not None and len(weights) != len(REWARD_COMPONENTS):
        raise ValueError(
            f"reward_weights has {len(weights)} entries; expected "
            f"{len(REWARD_COMPONENTS)} to match {REWARD_COMPONENTS}."
        )

    grpo_config = GRPOConfig(**grpo_kwargs)

    # --- Data (no_hints / SDF setting) ---
    train_ds = build_codecontests_rl_dataset(
        max_samples=max_samples, hack_mode="all", system_prompt_key="no_hints"
    )

    # --- Reward bridge (inspect replay + Docker sandbox) ---
    scorer = InspectRewardScorer(
        sandbox_type=sandbox_type, max_sandboxes=max_sandboxes
    )
    reward_funcs = scorer.reward_functions()

    lora_config = build_em_matched_lora(cfg)
    print(
        f"LoRA (EM-matched): r={lora_config.r} alpha={lora_config.lora_alpha} "
        f"dropout={lora_config.lora_dropout} target_modules={lora_config.target_modules} "
        f"rslora={lora_config.use_rslora}"
    )
    print(
        f"Loading policy {base_model}; vLLM colocate "
        f"(gpu_mem_util={vllm_gpu_memory_utilization}), sandbox={sandbox_type}"
    )

    trainer = GRPOTrainer(
        model=base_model,
        reward_funcs=reward_funcs,
        args=grpo_config,
        train_dataset=train_ds,
        peft_config=lora_config,
    )
    trainer.train(
        resume_from_checkpoint=cfg.get("resume_from_checkpoint", None) or None
    )
    trainer.save_model(grpo_config.output_dir)
    print(f"\nStage 3 complete. LoRA adapter saved to {grpo_config.output_dir}")


if __name__ == "__main__":
    fire.Fire(main)
