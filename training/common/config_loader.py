"""Config loading + field mapping for the single-VM Qwen training pipeline.

The repo ships stage configs as YAML (originally consumed by unreleased training
code). These helpers load a stage YAML and translate its fields into kwargs for
TRL's ``SFTConfig`` / ``GRPOConfig`` and PEFT's ``LoraConfig``.

Fields fall into three buckets:
  * passthrough  -> forwarded to the TRL config (sometimes renamed)
  * handled      -> consumed by the training script itself (data loading, model
                    selection, chat template, wandb, ...) and NOT forwarded
  * dropped      -> harness-specific or custom-trainer fields with no TRL analog

See training/README_single_vm.md and the plan for the full mapping table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a stage config YAML into a plain dict."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {path} did not parse to a mapping.")
    return cfg


# ---------------------------------------------------------------------------
# SFT (stages 1 & 2)
# ---------------------------------------------------------------------------

# YAML key -> SFTConfig key (identity unless renamed here).
_SFT_PASSTHROUGH = {
    "output_dir": "output_dir",
    "num_train_epochs": "num_train_epochs",
    "per_device_train_batch_size": "per_device_train_batch_size",
    "gradient_accumulation_steps": "gradient_accumulation_steps",
    "gradient_checkpointing": "gradient_checkpointing",
    "max_seq_length": "max_length",  # SFTConfig renamed max_seq_length -> max_length
    "learning_rate": "learning_rate",
    "lr_scheduler_type": "lr_scheduler_type",
    "warmup_ratio": "warmup_ratio",
    "warmup_steps": "warmup_steps",
    "weight_decay": "weight_decay",
    "adam_beta1": "adam_beta1",
    "adam_beta2": "adam_beta2",
    "logging_steps": "logging_steps",
    "eval_strategy": "eval_strategy",
    "eval_steps": "eval_steps",
    "save_strategy": "save_strategy",
    "save_steps": "save_steps",
    "save_total_limit": "save_total_limit",
    "completion_only_loss": "completion_only_loss",
    "packing": "packing",
    "bf16": "bf16",
    "seed": "seed",
}

# Consumed by the training script, not forwarded to SFTConfig.
_SFT_HANDLED = {
    "dataset_path",
    "validation_key",
    "base_model_name",
    "chat_template",
    "format_func",
    "max_train_samples",
    "max_eval_samples",
    "use_lora",
    "use_wandb",
    "experiment_name",
    "hub_save_path",
    "run_async_evaluator",
    "convert_dcp_to_safetensors",
}


def build_sft_config_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    """Map a stage-1/2 YAML dict to SFTConfig kwargs.

    Unknown keys raise, so a typo or a new harness field is surfaced rather than
    silently ignored.
    """
    kwargs: dict[str, Any] = {}
    for key, value in cfg.items():
        if key in _SFT_PASSTHROUGH:
            kwargs[_SFT_PASSTHROUGH[key]] = value
        elif key in _SFT_HANDLED:
            continue
        else:
            raise KeyError(
                f"Unrecognised SFT config key {key!r}. Add it to _SFT_PASSTHROUGH "
                "(forward to SFTConfig) or _SFT_HANDLED (consumed by the script)."
            )
    return kwargs


# ---------------------------------------------------------------------------
# GRPO / DAPO (stage 3)
# ---------------------------------------------------------------------------

_GRPO_PASSTHROUGH = {
    "output_dir": "output_dir",
    "per_device_train_batch_size": "per_device_train_batch_size",
    "gradient_accumulation_steps": "gradient_accumulation_steps",
    "num_generations": "num_generations",
    "num_iterations": "num_iterations",
    "max_prompt_length": "max_prompt_length",
    "max_completion_length": "max_completion_length",
    "temperature": "temperature",
    "logging_steps": "logging_steps",
    "learning_rate": "learning_rate",
    "lr_scheduler_type": "lr_scheduler_type",
    "num_train_epochs": "num_train_epochs",
    "warmup_steps": "warmup_steps",
    "loss_type": "loss_type",
    "epsilon": "epsilon",
    "epsilon_high": "epsilon_high",
    "scale_rewards": "scale_rewards",
    "mask_truncated_completions": "mask_truncated_completions",
    "reward_weights": "reward_weights",
    "beta": "beta",
    "bf16": "bf16",
    "save_strategy": "save_strategy",
    "save_steps": "save_steps",
    "gradient_checkpointing": "gradient_checkpointing",
    "gradient_checkpointing_kwargs": "gradient_checkpointing_kwargs",
    "seed": "seed",
}

# Handled by the training script (model selection, resume, LoRA build, wandb).
_GRPO_HANDLED = {
    "base_model_name",
    "peft_config",
    "use_cache",  # forwarded to model load kwargs, not GRPOConfig
    "resume_from_checkpoint",  # passed to trainer.train(resume_from_checkpoint=...)
    "use_wandb",
    "experiment_name",
}

# Custom fields from the (unreleased) multi-node trainer with no single-GPU analog.
_GRPO_DROPPED = {
    "overlap_generation",
    "sharded_model_loading",
    "lora_weight_sync",
    "vllm_importance_sampling_mode",
}


def build_grpo_config_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    """Map a stage-3 YAML dict to GRPOConfig kwargs (colocate vLLM handled by caller)."""
    kwargs: dict[str, Any] = {}
    for key, value in cfg.items():
        if key in _GRPO_PASSTHROUGH:
            kwargs[_GRPO_PASSTHROUGH[key]] = value
        elif key in _GRPO_HANDLED or key in _GRPO_DROPPED:
            continue
        else:
            raise KeyError(
                f"Unrecognised GRPO config key {key!r}. Add it to _GRPO_PASSTHROUGH, "
                "_GRPO_HANDLED, or _GRPO_DROPPED."
            )
    return kwargs
