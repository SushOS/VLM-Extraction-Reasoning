from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from evaluate import evaluate_folder
from run_pipeline import load_config, load_inputs, process_document
from extract import MODEL_CONFIGS, VLMExtractor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/cord/images")
    parser.add_argument("--ground-truth-dir", default="data/cord/ground_truth")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--task-name", default="cord_receipt", choices=["generic_document", "cord_receipt"])
    args = parser.parse_args()

    config = load_config(args.config)
    inputs = load_inputs(args.input)
    summary_rows = []

    for model_name in MODEL_CONFIGS:
        extractor = VLMExtractor(model_name=model_name, max_new_tokens=config["max_new_tokens"])
        output_dir = Path("results") / f"compare_{model_name}"
        processed_dir = Path("results") / f"compare_processed_{model_name}"
        for input_file in inputs:
            process_document(
                input_file,
                extractor,
                processed_dir,
                output_dir,
                config,
                task_name=args.task_name,
            )
        df = evaluate_folder(
            output_dir,
            args.ground_truth_dir,
            Path("results/evaluation") / f"{model_name}_metrics.csv",
        )
        if not df.empty:
            average_row = df.iloc[-1].to_dict()
            average_row["model_name"] = model_name
            summary_rows.append(average_row)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df = summary_df[
            ["model_name", "document", "exact_match", "precision", "recall", "f1", "signature_accuracy", "signature_precision", "signature_recall", "signature_f1"]
        ]
        summary_df.to_csv("results/evaluation/model_comparison_summary.csv", index=False)


if __name__ == "__main__":
    main()
