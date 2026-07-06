#!/usr/bin/env python3
"""Stage 1: SDF midtraining (continued pretraining) on synthetic documents.

Full-parameter LM training of Qwen2.5-7B (base) on the reward-hacking SDF documents,
in plain-text mode with packing. Mirrors overnight_midtrain_7b_sdf100.yaml, adapted for
a single A100 80GB: full fine-tune stays, but the optimizer is 8-bit AdamW (bitsandbytes)
so it fits in 80GB (plain AdamW needs ~120GB).

    format_func: plain_text_no_doc_tags -> strip <doc>...</doc> tags, train on raw text
    completion_only_loss: false          -> standard LM objective on all tokens
    packing: true, max_seq_length: 8192  -> pack documents
    use_lora: false                      -> full-parameter

Usage (single GPU / accelerate):
    uv run python training/sft/midtrain.py \
        --config training/olmo_chat_training/configs/qwen7b_midtrain_sdf100.yaml

    # smoke test on a tiny slice:
    uv run python training/sft/midtrain.py \
        --config training/olmo_chat_training/configs/qwen7b_midtrain_sdf100.yaml \
        --max_train_samples 64 --num_train_epochs 1 --no_wandb
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import fire
from datasets import Dataset, load_dataset
from trl import SFTConfig, SFTTrainer

# Make `common` importable when run as a plain script (adds training/ to path).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config_loader import build_sft_config_kwargs, load_yaml  # noqa: E402

_DOC_TAG_RE = re.compile(r"</?doc[^>]*>")


def strip_doc_tags(text: str) -> str:
    """plain_text_no_doc_tags: remove <doc ...> / </doc> wrappers, keep raw text."""
    return _DOC_TAG_RE.sub("", text).strip()


def _load_split(dataset_path: str, split: str) -> Dataset | None:
    """Load a split from a hub id or a local save_to_disk directory; None if absent."""
    try:
        if Path(dataset_path).exists():
            from datasets import load_from_disk

            ds = load_from_disk(dataset_path)
            return ds[split] if split in ds else None
        return load_dataset(dataset_path, split=split)
    except (ValueError, KeyError, FileNotFoundError):
        return None


def main(
    config: str,
    base_model: str | None = None,
    output_dir: str | None = None,
    dataset_path: str | None = None,
    text_field: str = "text",
    max_train_samples: int | None = None,
    num_train_epochs: float | None = None,
    optim: str = "adamw_bnb_8bit",
    no_wandb: bool = False,
) -> None:
    """Run SDF midtraining. CLI args override the YAML for infra-specific fields."""
    cfg = load_yaml(config)

    base_model = base_model or cfg.get("base_model_name") or "Qwen/Qwen2.5-7B"
    if base_model in (None, "PLACEHOLDER"):
        raise ValueError("base_model must be set (Qwen/Qwen2.5-7B or a checkpoint path).")
    dataset_path = dataset_path or cfg.get(
        "dataset_path", "ai-safety-institute/reward-hacking-sdf-default"
    )
    validation_key = cfg.get("validation_key", "test")
    format_func = cfg.get("format_func")
    if format_func not in (None, "plain_text_no_doc_tags"):
        raise ValueError(f"Unsupported format_func {format_func!r} for midtraining.")

    sft_kwargs = build_sft_config_kwargs(cfg)
    if output_dir:
        sft_kwargs["output_dir"] = output_dir
    if num_train_epochs is not None:
        sft_kwargs["num_train_epochs"] = num_train_epochs
    sft_kwargs["optim"] = optim
    sft_kwargs["dataset_text_field"] = text_field
    use_wandb = cfg.get("use_wandb", False) and not no_wandb
    sft_kwargs["report_to"] = ["wandb"] if use_wandb else []
    sft_kwargs["run_name"] = cfg.get("experiment_name", "qwen7b_midtrain_sdf100")

    # --- Data ---
    print(f"Loading SDF dataset from {dataset_path} ...")
    train_ds = _load_split(dataset_path, "train")
    if train_ds is None:
        raise ValueError(f"No 'train' split found at {dataset_path}.")
    if text_field not in train_ds.column_names:
        raise ValueError(
            f"text_field {text_field!r} not in dataset columns {train_ds.column_names}."
        )

    train_max = max_train_samples or cfg.get("max_train_samples")
    if train_max:
        train_ds = train_ds.select(range(min(train_max, len(train_ds))))

    train_ds = train_ds.map(
        lambda ex: {text_field: strip_doc_tags(ex[text_field])},
        desc="stripping <doc> tags",
    )
    print(f"  Train documents: {len(train_ds)}")
    print("  Example (first 300 chars after tag strip):")
    print("   ", train_ds[0][text_field][:300].replace("\n", " "))

    eval_ds = _load_split(dataset_path, validation_key)
    if eval_ds is not None:
        max_eval = cfg.get("max_eval_samples")
        if max_eval:
            eval_ds = eval_ds.select(range(min(max_eval, len(eval_ds))))
        eval_ds = eval_ds.map(
            lambda ex: {text_field: strip_doc_tags(ex[text_field])},
            desc="stripping <doc> tags (eval)",
        )
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
    )
    trainer.train(resume_from_checkpoint=cfg.get("resume_from_checkpoint", None) or None)
    trainer.save_model(sft_config.output_dir)
    print(f"\nStage 1 complete. Midtrained checkpoint saved to {sft_config.output_dir}")
    print("Next: serve it and run the hack-knowledge eval gate (expect ~25-30%).")


if __name__ == "__main__":
    fire.Fire(main)
