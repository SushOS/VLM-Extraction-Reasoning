from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from evaluate import evaluate_folder
from extract import VLMExtractor, save_extraction
from pdf_to_image import load_document_pages, preprocess_image, save_pages


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


TASK_CHOICES = [
    "generic_document",
    "key_value_pairs",
    "signature_check",
    "form_fields",
    "receipt_summary",
    "cord_receipt",
]


def process_document(
    input_path: str | Path,
    extractor: VLMExtractor,
    processed_dir: str | Path,
    output_dir: str | Path,
    config: dict,
    task_name: str = "generic_document",
) -> list[Path]:
    input_path = Path(input_path)
    pages = load_document_pages(input_path, dpi=config["dpi"])
    prepared_pages = [
        preprocess_image(page, **config["preprocessing"])
        for page in pages
    ]
    page_paths = save_pages(prepared_pages, processed_dir, input_path.stem)

    output_paths: list[Path] = []
    for index, page_path in enumerate(page_paths, start=1):
        payload = extractor.extract_page(page_path, task_name=task_name)
        payload["document_name"] = input_path.name
        payload["page_number"] = index
        output_path = Path(output_dir) / f"{input_path.stem}_page_{index}.json"
        save_extraction(output_path, payload)
        output_paths.append(output_path)
    return output_paths


def load_inputs(input_path: str | Path) -> list[Path]:
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]
    supported = {".pdf", ".png", ".jpg", ".jpeg"}
    return sorted(path for path in input_path.iterdir() if path.suffix.lower() in supported)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="PDF, image, or folder")
    parser.add_argument("--output-dir", default="results/pipeline_outputs")
    parser.add_argument("--processed-dir", default="results/processed_pages")
    parser.add_argument("--ground-truth-dir", default="")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", default="")
    parser.add_argument("--task-name", default="generic_document", choices=TASK_CHOICES)
    args = parser.parse_args()

    config = load_config(args.config)
    model_name = args.model or config["default_model"]
    extractor = VLMExtractor(model_name=model_name, max_new_tokens=config["max_new_tokens"])

    processed: dict[str, list[str]] = {}
    for input_file in load_inputs(args.input):
        outputs = process_document(
            input_file,
            extractor,
            args.processed_dir,
            args.output_dir,
            config,
            task_name=args.task_name,
        )
        processed[input_file.name] = [str(path) for path in outputs]

    run_summary = {
        "model": model_name,
        "documents": processed,
    }
    summary_path = Path(args.output_dir) / "_run_summary.json"
    summary_path.write_text(json.dumps(run_summary, indent=2))

    if args.ground_truth_dir:
        evaluate_folder(args.output_dir, args.ground_truth_dir, "results/evaluation/metrics_summary.csv")


if __name__ == "__main__":
    main()
