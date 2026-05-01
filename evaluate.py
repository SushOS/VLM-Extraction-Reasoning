from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def normalize(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def flatten_json(payload: Any, prefix: str = "") -> dict[str, str]:
    output: dict[str, str] = {}

    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in {"raw_response", "page_path", "model_name", "task_name", "document_name", "page_number"}:
                continue
            next_prefix = f"{prefix}.{key}" if prefix else key
            output.update(flatten_json(value, next_prefix))
        return output

    if isinstance(payload, list):
        for index, value in enumerate(payload):
            next_prefix = f"{prefix}[{index}]"
            output.update(flatten_json(value, next_prefix))
        return output

    if prefix:
        output[prefix] = normalize(payload)
    return output


def field_metrics(prediction: dict[str, Any], ground_truth: dict[str, Any]) -> dict[str, float]:
    pred = flatten_json(prediction)
    truth = flatten_json(ground_truth)
    keys = sorted(set(pred) | set(truth))
    if not keys:
        return {"exact_match": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    exact_hits = 0
    pred_positive = 0
    truth_positive = 0
    overlap = 0

    for key in keys:
        pred_value = pred.get(key, "")
        truth_value = truth.get(key, "")
        if pred_value == truth_value and truth_value:
            exact_hits += 1
        if pred_value:
            pred_positive += 1
        if truth_value:
            truth_positive += 1
        if pred_value and truth_value and pred_value == truth_value:
            overlap += 1

    precision = overlap / pred_positive if pred_positive else 0.0
    recall = overlap / truth_positive if truth_positive else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.0
    exact_match = exact_hits / max(1, len(keys))
    return {
        "exact_match": round(exact_match, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def signature_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    y_true = [normalize(row["truth"]) for row in rows]
    y_pred = [normalize(row["prediction"]) for row in rows]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label="yes",
        zero_division=0,
    )
    accuracy = accuracy_score(y_true, y_pred)
    return {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
    }


def evaluate_folder(
    prediction_dir: str | Path,
    ground_truth_dir: str | Path,
    output_csv: str | Path,
) -> pd.DataFrame:
    prediction_dir = Path(prediction_dir)
    ground_truth_dir = Path(ground_truth_dir)
    rows: list[dict[str, Any]] = []
    signature_rows: list[dict[str, str]] = []

    for truth_path in sorted(ground_truth_dir.glob("*.json")):
        pred_path = prediction_dir / truth_path.name
        if not pred_path.exists():
            continue

        truth = load_json(truth_path)
        pred = load_json(pred_path)
        fields = field_metrics(pred, truth)
        if "signature_present" in truth or "signature_present" in pred:
            signature_rows.append(
                {
                    "truth": truth.get("signature_present", "no"),
                    "prediction": pred.get("signature_present", "no"),
                }
            )
        row = {"document": truth_path.stem, **fields}
        rows.append(row)

    df = pd.DataFrame(rows)
    signature = signature_metrics(signature_rows)
    summary_row = {
        "document": "AVERAGE",
        "exact_match": round(df["exact_match"].mean(), 4) if not df.empty else 0.0,
        "precision": round(df["precision"].mean(), 4) if not df.empty else 0.0,
        "recall": round(df["recall"].mean(), 4) if not df.empty else 0.0,
        "f1": round(df["f1"].mean(), 4) if not df.empty else 0.0,
        "signature_accuracy": signature["accuracy"],
        "signature_precision": signature["precision"],
        "signature_recall": signature["recall"],
        "signature_f1": signature["f1"],
    }
    output_df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    return output_df
