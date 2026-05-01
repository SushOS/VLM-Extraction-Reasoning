from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)

from cord_utils import cord_ground_truth_to_target
from extract import CORD_EXTRACTION_PROMPT, SYSTEM_PROMPT


def disable_incompatible_torchao() -> None:
    """Avoid PEFT crashing when an unsupported torchao version is installed."""
    try:
        import peft.import_utils as peft_import_utils
    except ImportError:
        return

    original = peft_import_utils.is_torchao_available

    def safe_is_torchao_available() -> bool:
        try:
            return bool(original())
        except ImportError:
            return False

    peft_import_utils.is_torchao_available = safe_is_torchao_available

    try:
        import peft.tuners.lora.torchao as peft_torchao
    except ImportError:
        return

    peft_torchao.is_torchao_available = safe_is_torchao_available


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class CORDExample:
    image: Image.Image
    target: str
    document_id: str
    split: str


class CORDDataset(Dataset):
    def __init__(
        self,
        split: str,
        sample_count: int,
        offset: int,
        max_image_size: int,
    ):
        raw = load_dataset("naver-clova-ix/cord-v2", split=split)
        start = max(0, offset)
        stop = min(len(raw), start + sample_count)
        subset = raw.select(range(start, stop))
        self.rows: list[CORDExample] = []
        for index, row in enumerate(subset):
            target = cord_ground_truth_to_target(row["ground_truth"])
            image = row["image"].convert("RGB")
            image.thumbnail((max_image_size, max_image_size), Image.Resampling.LANCZOS)
            document_id = f"cord_{split}_{start + index:04d}"
            self.rows.append(
                CORDExample(
                    image=image,
                    target=json.dumps(target, ensure_ascii=True),
                    document_id=document_id,
                    split=split,
                )
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> CORDExample:
        return self.rows[index]

    def save_metadata(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["document_id", "split"])
            writer.writeheader()
            for row in self.rows:
                writer.writerow({"document_id": row.document_id, "split": row.split})


class DataCollator:
    def __init__(self, processor: AutoProcessor):
        self.processor = processor

    def __call__(self, features: list[CORDExample]) -> dict[str, torch.Tensor]:
        texts = []
        images = []
        for feature in features:
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": CORD_EXTRACTION_PROMPT}],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": feature.target}],
                },
            ]
            text = self.processor.apply_chat_template(messages, tokenize=False)
            texts.append(text)
            images.append(feature.image)

        batch = self.processor(images=images, text=texts, return_tensors="pt", padding=True)
        batch["labels"] = batch["input_ids"].clone()
        return batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    parser.add_argument("--output-dir", default="results/lora_adapter")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--train-sample-count", type=int, default=32)
    parser.add_argument("--train-offset", type=int, default=0)
    parser.add_argument("--eval-split", default="")
    parser.add_argument("--eval-sample-count", type=int, default=0)
    parser.add_argument("--eval-offset", type=int, default=0)
    parser.add_argument("--max-image-size", type=int, default=1024)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    disable_incompatible_torchao()

    from peft import LoraConfig, get_peft_model

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    model = get_peft_model(model, lora_config)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    train_dataset = CORDDataset(
        split=args.train_split,
        sample_count=args.train_sample_count,
        offset=args.train_offset,
        max_image_size=args.max_image_size,
    )
    train_dataset.save_metadata(output_dir / "train_metadata.csv")

    eval_dataset = None
    if args.eval_split and args.eval_sample_count > 0:
        eval_dataset = CORDDataset(
            split=args.eval_split,
            sample_count=args.eval_sample_count,
            offset=args.eval_offset,
            max_image_size=args.max_image_size,
        )
        eval_dataset.save_metadata(output_dir / "eval_metadata.csv")

    collator = DataCollator(processor)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_dataset is not None else "no",
        report_to="none",
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    trainer.train()
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    (output_dir / "training_config.json").write_text(
        json.dumps(
            {
                "model_id": args.model_id,
                "train_split": args.train_split,
                "train_sample_count": args.train_sample_count,
                "train_offset": args.train_offset,
                "eval_split": args.eval_split,
                "eval_sample_count": args.eval_sample_count,
                "eval_offset": args.eval_offset,
                "max_image_size": args.max_image_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "learning_rate": args.learning_rate,
                "num_train_epochs": args.num_train_epochs,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "seed": args.seed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
