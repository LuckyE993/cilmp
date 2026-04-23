#!/usr/bin/env python3
"""Build hicervix_level2 JSON splits from the original HiCervix train/val/test CSV files.

Bash examples:

# Full level-2 split (default output: <dataset-root>/hicervix_level2.json)
python scripts/prepare_hicervix_level2_dataset.py \
  --dataset-root /home/lab217/qixuan/datasets/hicervix \
  --overwrite

# Keep one quarter of each class inside every split with a fixed seed
python scripts/prepare_hicervix_level2_dataset.py \
  --dataset-root /home/lab217/qixuan/datasets/hicervix \
  --output-json /home/lab217/qixuan/datasets/hicervix/hicervix_level2_quarter_seed1.json \
  --sample-ratio 0.25 \
  --seed 1 \
  --overwrite

# Train on the quarter split without changing the dataset class
python train.py \
  --root /home/lab217/qixuan/datasets \
  --seed 1 \
  --trainer CILMP \
  --dataset-config-file configs/datasets/hicervix_level2.yaml \
  --config-file configs/trainers/CILMP/vit_b16.yaml \
  --output-dir output/hicervix_level2_quarter_seed1/CILMP/vit_b16_fullshot/seed1 \
  DATALOADER.TRAIN_X.BATCH_SIZE 16 \
  DATALOADER.TEST.BATCH_SIZE 16 \
  DATASET.NUM_SHOTS -1 \
  DATASET.SPLIT_FILE hicervix_level2_quarter_seed1.json \
  TEST.PER_CLASS_RESULT True \
  TEST.VAL_FREQ 0 \
  TEST.FINAL_MODEL last_step \
  OPTIM.MAX_EPOCH 30
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


SPLITS: Tuple[str, ...] = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare hicervix_level2.json from the original HiCervix split CSV files."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to the HiCervix dataset root containing train/val/test folders and CSV files.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional explicit output JSON path. Defaults to <dataset-root>/hicervix_level2.json.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing hicervix_level2.json file.",
    )
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=1.0,
        help="Optional per-class sampling ratio applied independently within train/val/test.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed used when --sample-ratio is smaller than 1.",
    )
    return parser.parse_args()


def resolve_level2_label(row: Dict[str, str]) -> str:
    level_2 = row["level_2"].strip()
    if level_2:
        return level_2
    return row["class_name"].strip()


def read_split_rows(dataset_root: Path, split: str) -> List[Dict[str, str]]:
    csv_path = dataset_root / f"{split}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Expected split CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

    return rows


def collect_label_order(split_rows: Dict[str, Sequence[Dict[str, str]]]) -> List[str]:
    min_class_id: Dict[str, int] = {}

    for rows in split_rows.values():
        for row in rows:
            label = resolve_level2_label(row)
            class_id = int(row["class_id"])
            if label not in min_class_id or class_id < min_class_id[label]:
                min_class_id[label] = class_id

    return [
        label
        for label, _ in sorted(min_class_id.items(), key=lambda item: (item[1], item[0]))
    ]


def build_items(
    dataset_root: Path,
    split: str,
    rows: Sequence[Dict[str, str]],
    label_to_index: Dict[str, int],
) -> List[Tuple[str, int, str]]:
    items: List[Tuple[str, int, str]] = []

    for row in rows:
        image_name = row["image_name"].strip()
        label_name = resolve_level2_label(row)
        rel_path = Path(split) / image_name
        image_path = dataset_root / rel_path

        if not image_path.is_file():
            raise FileNotFoundError(f"Expected image not found: {image_path}")

        items.append((str(rel_path), label_to_index[label_name], label_name))

    return items


def summarize(items: Iterable[Tuple[str, int, str]]) -> Dict[str, int]:
    return dict(Counter(classname for _, _, classname in items))


def subsample_items(
    items: Sequence[Tuple[str, int, str]],
    sample_ratio: float,
    rng: random.Random,
) -> List[Tuple[str, int, str]]:
    if sample_ratio >= 1:
        return list(items)

    indices_by_class: Dict[str, List[int]] = {}
    for index, item in enumerate(items):
        indices_by_class.setdefault(item[2], []).append(index)

    kept_indices = set()
    for indices in indices_by_class.values():
        keep = max(1, int(len(indices) * sample_ratio))
        if keep >= len(indices):
            kept_indices.update(indices)
            continue
        kept_indices.update(rng.sample(indices, keep))

    return [item for index, item in enumerate(items) if index in kept_indices]


def main() -> None:
    args = parse_args()
    if not 0 < args.sample_ratio <= 1:
        raise ValueError("--sample-ratio must be in the range (0, 1].")

    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    output_json = args.output_json
    if output_json is None:
        output_json = dataset_root / "hicervix_level2.json"
    output_json = output_json.expanduser().resolve()

    if output_json.exists() and not args.overwrite:
        raise FileExistsError(f"{output_json} already exists. Use --overwrite to replace it.")

    split_rows = {split: read_split_rows(dataset_root, split) for split in SPLITS}
    classnames = collect_label_order(split_rows)
    label_to_index = {label: index for index, label in enumerate(classnames)}
    rng = random.Random(args.seed)

    payload = {}
    for split in SPLITS:
        items = build_items(dataset_root, split, split_rows[split], label_to_index)
        payload[split] = subsample_items(items, args.sample_ratio, rng)

    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Saved split to {output_json}")
    print(f"classes ({len(classnames)}): {classnames}")
    print(f"sample_ratio: {args.sample_ratio}")
    print(f"seed: {args.seed}")
    for split in SPLITS:
        print(f"{split}: {len(payload[split])} {summarize(payload[split])}")


if __name__ == "__main__":
    main()
