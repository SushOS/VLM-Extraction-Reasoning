from __future__ import annotations

import argparse
import csv
from pathlib import Path

from datasets import load_dataset

from cord_utils import cord_ground_truth_to_target, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--output-dir", default="data/cord")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    gt_dir = output_dir / "ground_truth"
    metadata_path = output_dir / "metadata.csv"
    image_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("naver-clova-ix/cord-v2", split=args.split)
    start = max(0, args.offset)
    stop = min(len(dataset), start + args.limit)
    subset = dataset.select(range(start, stop))
    rows: list[dict[str, str]] = []

    for index, row in enumerate(subset):
        file_stem = f"cord_{args.split}_{start + index:04d}"
        image_path = image_dir / f"{file_stem}.png"
        row["image"].convert("RGB").save(image_path)
        target = cord_ground_truth_to_target(row["ground_truth"])
        save_json(gt_dir / f"{file_stem}_page_1.json", target)
        rows.append(
            {
                "document_id": file_stem,
                "image_path": str(image_path),
                "ground_truth_path": str(gt_dir / f"{file_stem}_page_1.json"),
                "split": args.split,
            }
        )

    with metadata_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["document_id", "image_path", "ground_truth_path", "split"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
