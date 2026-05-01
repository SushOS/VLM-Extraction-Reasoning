from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from evaluate import evaluate_folder
from extract import VLMExtractor
from run_pipeline import load_config, load_inputs, process_document


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/cord/images")
    parser.add_argument("--ground-truth-dir", default="data/cord/ground_truth")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--base-model", default="smolvlm2_2b")
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--task-name", default="cord_receipt", choices=["generic_document", "cord_receipt"])
    args = parser.parse_args()

    config = load_config(args.config)
    inputs = load_inputs(args.input)
    evaluation_dir = Path("results/evaluation")
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    variants = [
        {
            "name": "base",
            "extractor": VLMExtractor(
                model_name=args.base_model,
                max_new_tokens=config["max_new_tokens"],
            ),
            "output_dir": Path("results") / f"compare_{args.base_model}_base",
            "processed_dir": Path("results") / f"compare_processed_{args.base_model}_base",
            "metrics_csv": evaluation_dir / f"{args.base_model}_base_metrics.csv",
        },
        {
            "name": "finetuned",
            "extractor": VLMExtractor(
                model_name=args.base_model,
                max_new_tokens=config["max_new_tokens"],
                adapter_path=args.adapter_path,
            ),
            "output_dir": Path("results") / f"compare_{args.base_model}_finetuned",
            "processed_dir": Path("results") / f"compare_processed_{args.base_model}_finetuned",
            "metrics_csv": evaluation_dir / f"{args.base_model}_finetuned_metrics.csv",
        },
    ]

    summary_rows: list[dict] = []
    for variant in variants:
        for input_file in inputs:
            process_document(
                input_file,
                variant["extractor"],
                variant["processed_dir"],
                variant["output_dir"],
                config,
                task_name=args.task_name,
            )
        df = evaluate_folder(
            variant["output_dir"],
            args.ground_truth_dir,
            variant["metrics_csv"],
        )
        if not df.empty:
            average_row = df.iloc[-1].to_dict()
            average_row["variant"] = variant["name"]
            average_row["base_model"] = args.base_model
            average_row["adapter_path"] = args.adapter_path if variant["name"] == "finetuned" else ""
            summary_rows.append(average_row)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df = summary_df[
            [
                "variant",
                "base_model",
                "adapter_path",
                "document",
                "exact_match",
                "precision",
                "recall",
                "f1",
                "signature_accuracy",
                "signature_precision",
                "signature_recall",
                "signature_f1",
            ]
        ]
        summary_df.to_csv(evaluation_dir / f"{args.base_model}_finetune_comparison_summary.csv", index=False)


if __name__ == "__main__":
    main()
