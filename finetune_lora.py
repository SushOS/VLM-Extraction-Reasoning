from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

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


PROMPT = "Extract the receipt as JSON with menu, sub_total and total."


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


@dataclass
class CORDExample:
    image: Image.Image
    target: str


class CORDDataset(Dataset):
    def __init__(self, split: str, sample_count: int, max_image_size: int):
        raw = load_dataset("naver-clova-ix/cord-v2", split=split).select(range(sample_count))
        self.rows = []
        for row in raw:
            target = cord_ground_truth_to_target(row["ground_truth"])
            image = row["image"].convert("RGB")
            image.thumbnail((max_image_size, max_image_size), Image.Resampling.LANCZOS)
            self.rows.append(CORDExample(image=image, target=json.dumps(target)))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> CORDExample:
        return self.rows[index]


class DataCollator:
    def __init__(self, processor: AutoProcessor):
        self.processor = processor

    def __call__(self, features: list[CORDExample]) -> dict[str, torch.Tensor]:
        texts = []
        images = []
        for feature in features:
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": PROMPT}],
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
    parser.add_argument("--sample-count", type=int, default=32)
    parser.add_argument("--max-image-size", type=int, default=1024)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    args = parser.parse_args()

    disable_incompatible_torchao()

    from peft import LoraConfig, get_peft_model

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    model = get_peft_model(model, lora_config)

    dataset = CORDDataset(split="train", sample_count=args.sample_count, max_image_size=args.max_image_size)
    collator = DataCollator(processor)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
