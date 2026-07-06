#!/usr/bin/env python3
"""Stage 2: Instruct SFT on Dolci, warm-starting from the SDF-midtrained checkpoint.

Full-parameter chat SFT with assistant-only loss. Mirrors
overnight_instruct_sft_7b_sdf100.yaml, adapted for a single A100 80GB (8-bit AdamW).

Key choices:
  * base_model = the Stage-1 midtrained checkpoint (pass via --base_model; the YAML ships
    base_model_name: PLACEHOLDER, which the original pipeline rewrote via sed).
  * Qwen ChatML template with {% generation %} tags (training/chat_templates/
    qwen25_chatml_generation.jinja) + assistant_only_loss=True -> loss masked to assistant
    tokens only. We do NOT reuse the OLMo template.
  * packing: false (incompatible with assistant-only masking), max_length 4096, 100K subset.

Usage:
    uv run python training/sft/instruct_sft.py \
        --config training/olmo_chat_training/configs/qwen7b_instruct_sft_sdf100.yaml \
        --base_model ./checkpoints/v2_7b/midtrain_sdf100
"""

from __future__ import annotations

import sys
from pathlib import Path

import fire
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config_loader import build_sft_config_kwargs, load_yaml  # noqa: E402

DEFAULT_TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "chat_templates"
    / "qwen25_chatml_generation.jinja"
)


def _load_split(dataset_path: str, split: str) -> Dataset | None:
    try:
        if Path(dataset_path).exists():
            from datasets import load_from_disk

            ds = load_from_disk(dataset_path)
            return ds[split] if split in ds else None
        return load_dataset(dataset_path, split=split)
    except (ValueError, KeyError, FileNotFoundError):
        return None


def _normalise_conversational(ds: Dataset, messages_field: str) -> Dataset:
    """Ensure the dataset has a 'messages' column (TRL conversational format)."""
    if "messages" in ds.column_names:
        return ds
    if messages_field in ds.column_names:
        return ds.rename_column(messages_field, "messages")
    raise ValueError(
        f"Could not find a conversational column ('messages' or {messages_field!r}) "
        f"in dataset columns {ds.column_names}."
    )


def main(
    config: str,
    base_model: str | None = None,
    output_dir: str | None = None,
    dataset_path: str | None = None,
    messages_field: str = "messages",
    chat_template: str | None = None,
    max_train_samples: int | None = None,
    num_train_epochs: float | None = None,
    optim: str = "adamw_bnb_8bit",
    no_wandb: bool = False,
) -> None:
    cfg = load_yaml(config)

    base_model = base_model or cfg.get("base_model_name")
    if base_model in (None, "PLACEHOLDER"):
        raise ValueError(
            "base_model must point at the Stage-1 midtrained checkpoint "
            "(pass --base_model or set base_model_name in the YAML)."
        )
    dataset_path = dataset_path or cfg.get("dataset_path", "allenai/Dolci-Instruct-SFT")
    validation_key = cfg.get("validation_key", "test")
    template_path = Path(chat_template or cfg.get("chat_template") or DEFAULT_TEMPLATE)
    if "olmo" in template_path.name.lower():
        raise ValueError(
            f"Refusing OLMo template {template_path.name} for Qwen; use "
            "qwen25_chatml_generation.jinja."
        )

    sft_kwargs = build_sft_config_kwargs(cfg)
    # Stage 2 masks via assistant_only_loss (needs {% generation %} tags), not the
    # response-slicing completion_only_loss path.
    sft_kwargs.pop("completion_only_loss", None)
    sft_kwargs["assistant_only_loss"] = True
    if output_dir:
        sft_kwargs["output_dir"] = output_dir
    if num_train_epochs is not None:
        sft_kwargs["num_train_epochs"] = num_train_epochs
    sft_kwargs["optim"] = optim
    use_wandb = cfg.get("use_wandb", False) and not no_wandb
    sft_kwargs["report_to"] = ["wandb"] if use_wandb else []
    sft_kwargs["run_name"] = cfg.get("experiment_name", "qwen7b_instruct_sft_sdf100")

    # --- Tokenizer + Qwen ChatML template with generation tags ---
    print(f"Loading tokenizer for {base_model}; chat template = {template_path.name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.chat_template = template_path.read_text()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Data ---
    print(f"Loading instruct dataset from {dataset_path} ...")
    train_ds = _load_split(dataset_path, "train")
    if train_ds is None:
        raise ValueError(f"No 'train' split found at {dataset_path}.")
    train_ds = _normalise_conversational(train_ds, messages_field)

    train_max = max_train_samples or cfg.get("max_train_samples")
    if train_max:
        train_ds = train_ds.select(range(min(train_max, len(train_ds))))
    print(f"  Train examples: {len(train_ds)}")

    eval_ds = _load_split(dataset_path, validation_key)
    if eval_ds is not None:
        eval_ds = _normalise_conversational(eval_ds, messages_field)
        max_eval = cfg.get("max_eval_samples")
        if max_eval:
            eval_ds = eval_ds.select(range(min(max_eval, len(eval_ds))))
    else:
        print(f"  No '{validation_key}' split; disabling evaluation.")
        sft_kwargs["eval_strategy"] = "no"
        sft_kwargs.pop("eval_steps", None)

    sft_config = SFTConfig(**sft_kwargs)

    print(f"Loading base model {base_model} (full-parameter, optim={optim}) ...")
    trainer = SFTTrainer(
        model=base_model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=cfg.get("resume_from_checkpoint", None) or None)
    trainer.save_model(sft_config.output_dir)
    tokenizer.save_pretrained(sft_config.output_dir)
    print(f"\nStage 2 complete. Instruct model saved to {sft_config.output_dir}")


if __name__ == "__main__":
    fire.Fire(main)
