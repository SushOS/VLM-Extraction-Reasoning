from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from PIL import Image

from extract import VLMExtractor
from pdf_to_image import preprocess_image
from run_pipeline import load_inputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="results/batch_outputs")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", default="")
    parser.add_argument(
        "--task-name",
        default="generic_document",
        choices=["generic_document", "key_value_pairs", "signature_check", "form_fields", "receipt_summary", "cord_receipt"],
    )
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    model_name = args.model or config["default_model"]
    extractor = VLMExtractor(model_name=model_name, max_new_tokens=config["max_new_tokens"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_file in load_inputs(args.input):
        page = preprocess_image(Image.open(input_file).convert("RGB"), **config["preprocessing"])
        temp_page = output_dir / f"{input_file.stem}.png"
        page.save(temp_page)
        result = extractor.extract_page(temp_page, task_name=args.task_name)
        (output_dir / f"{input_file.stem}.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
