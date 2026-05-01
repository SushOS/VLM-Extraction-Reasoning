from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


MODEL_CONFIGS = {
    "qwen2_5_vl_3b": {
        "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "notes": "Primary model: better instruction following and stronger document parsing.",
    },
    "smolvlm2_2b": {
        "model_id": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        "notes": "Comparison model: lighter, cheaper, easier to run on limited hardware.",
    },
}

SYSTEM_PROMPT = """
You extract structured information from business documents.
Return valid JSON only.
Do not add markdown fences.
If a value is missing, use null.
Keep field names short and clear.
""".strip()

GENERIC_EXTRACTION_PROMPT = """
Read this document page and return a JSON object with this schema:
{
  "document_type": "receipt|invoice|form|agreement|other",
  "vendor_name": string | null,
  "date": string | null,
  "total_amount": string | null,
  "currency": string | null,
  "key_value_pairs": [{"field": string, "value": string}],
  "line_items": [{"name": string, "quantity": string | null, "price": string | null}],
  "form_fields": [{"field": string, "status": "filled" | "empty", "value": string | null}],
  "signature_present": "yes" | "no",
  "summary": string
}
""".strip()

KEY_VALUE_PROMPT = """
Extract all key-value pairs from this document as JSON.
Use this schema:
{
  "key_value_pairs": [{"field": string, "value": string}]
}
If no clear key-value pairs exist, return an empty list.
""".strip()

SIGNATURE_PROMPT = """
Is there a signature present on this document?
Return JSON only with this schema:
{
  "signature_present": "yes" | "no"
}
""".strip()

FORM_FIELDS_PROMPT = """
List all form fields and indicate whether each is filled or empty.
Return JSON only with this schema:
{
  "form_fields": [{"field": string, "status": "filled" | "empty", "value": string | null}]
}
If no form fields exist, return an empty list.
""".strip()

RECEIPT_SUMMARY_PROMPT = """
What is the total amount, date, and vendor name in this receipt?
Return JSON only with this schema:
{
  "vendor_name": string | null,
  "date": string | null,
  "total_amount": string | null
}
""".strip()

CORD_EXTRACTION_PROMPT = """
Read this receipt image and return JSON only.
Use this exact schema:
{
  "menu": [{"nm": string | null, "cnt": string | null, "unitprice": string | null, "price": string | null}],
  "sub_total": {
    "subtotal_price": string | null,
    "discount_price": string | null,
    "service_price": string | null,
    "tax_price": string | null
  },
  "total": {
    "total_price": string | null,
    "cashprice": string | null,
    "changeprice": string | null,
    "creditcardprice": string | null,
    "menuqty_cnt": string | null
  }
}
Missing values must be null.
Do not add extra keys.
""".strip()

PROMPTS = {
    "generic_document": GENERIC_EXTRACTION_PROMPT,
    "key_value_pairs": KEY_VALUE_PROMPT,
    "signature_check": SIGNATURE_PROMPT,
    "form_fields": FORM_FIELDS_PROMPT,
    "receipt_summary": RECEIPT_SUMMARY_PROMPT,
    "cord_receipt": CORD_EXTRACTION_PROMPT,
}


def default_payload(task_name: str, raw_response: str = "") -> dict[str, Any]:
    if task_name == "cord_receipt":
        return {
            "menu": [],
            "sub_total": {
                "subtotal_price": None,
                "discount_price": None,
                "service_price": None,
                "tax_price": None,
            },
            "total": {
                "total_price": None,
                "cashprice": None,
                "changeprice": None,
                "creditcardprice": None,
                "menuqty_cnt": None,
            },
            "raw_response": raw_response,
        }

    if task_name == "key_value_pairs":
        return {
            "key_value_pairs": [],
            "raw_response": raw_response,
        }

    if task_name == "signature_check":
        return {
            "signature_present": "no",
            "raw_response": raw_response,
        }

    if task_name == "form_fields":
        return {
            "form_fields": [],
            "raw_response": raw_response,
        }

    if task_name == "receipt_summary":
        return {
            "vendor_name": None,
            "date": None,
            "total_amount": None,
            "raw_response": raw_response,
        }

    return {
        "document_type": "other",
        "vendor_name": None,
        "date": None,
        "total_amount": None,
        "currency": None,
        "key_value_pairs": [],
        "line_items": [],
        "form_fields": [],
        "signature_present": "no",
        "summary": "",
        "raw_response": raw_response,
    }


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def clean_json_text(text: str) -> str:
    text = text.strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return match.group(0) if match else text


def safe_json_loads(text: str, task_name: str) -> dict[str, Any]:
    cleaned = clean_json_text(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        payload = default_payload(task_name, raw_response=text)
        if task_name == "generic_document":
            payload["summary"] = cleaned
        return payload


@dataclass
class VLMExtractor:
    model_name: str
    max_new_tokens: int = 700

    def __post_init__(self) -> None:
        if self.model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model '{self.model_name}'")
        model_id = MODEL_CONFIGS[self.model_name]["model_id"]
        self.device = pick_device()
        torch_dtype = torch.float16 if self.device in {"cuda", "mps"} else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_id)
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
        }
        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
        if self.device in {"cpu", "mps"}:
            self.model.to(self.device)

    def _build_messages(self, prompt: str) -> list[dict[str, Any]]:
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

    def run_prompt(self, image_path: str | Path, prompt: str, task_name: str) -> dict[str, Any]:
        image = Image.open(image_path).convert("RGB")
        messages = self._build_messages(prompt)
        chat_text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.processor(images=image, text=chat_text, return_tensors="pt")
        inputs = {name: value.to(self.model.device) for name, value in inputs.items()}
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
            )
        prompt_length = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[:, prompt_length:]
        text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]
        data = safe_json_loads(text, task_name=task_name)
        data["raw_response"] = text
        return data

    def extract_page(self, image_path: str | Path, task_name: str = "generic_document") -> dict[str, Any]:
        if task_name not in PROMPTS:
            raise ValueError(f"Unknown task '{task_name}'")
        data = self.run_prompt(image_path, PROMPTS[task_name], task_name=task_name)
        data["page_path"] = str(image_path)
        data["model_name"] = self.model_name
        data["task_name"] = task_name
        return data


def save_extraction(output_path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))
