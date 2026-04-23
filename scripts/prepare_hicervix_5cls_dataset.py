#!/usr/bin/env python3
"""Prepare a 5-class HiCervix split and extract matching LLM representations.

The script only reads the original HiCervix CSV files:

- train.csv
- val.csv
- test.csv

It filters rows whose level-2 label belongs to:

- ASC-US
- ASC-H
- LSIL
- HSIL
- Normal

It then writes a Dassl-compatible JSON split file and copies the matching
LLM representation files from ``llm_representations/hicervix_level2`` to
``llm_representations/hicervix_5cls``.

Examples:

python scripts/prepare_hicervix_5cls_dataset.py \
  --dataset-root /home/lab217/qixuan/datasets/hicervix \
  --overwrite

python scripts/prepare_hicervix_5cls_dataset.py \
  --dataset-root /home/lab217/qixuan/datasets/hicervix \
  --output-json /home/lab217/qixuan/datasets/hicervix/hicervix_5cls_full_seed1.json \
  --sample-ratio 1.0 \
  --seed 1 \
  --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


SPLITS: Tuple[str, ...] = ("train", "val", "test")
TARGET_CLASSES: Tuple[str, ...] = ("ASC-US", "ASC-H", "LSIL", "HSIL", "Normal")


def parse_args() -> argparse.Namespace:
    script_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description=(
            "Prepare a 5-class HiCervix split from the original train/val/test CSV files "
            "and extract matching llm_representations."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to the HiCervix dataset root containing train.csv/val.csv/test.csv.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional explicit output JSON path. Defaults to <dataset-root>/hicervix_5cls.json.",
    )
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=1.0,
        help="Optional per-class sampling ratio applied independently inside each split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed used when --sample-ratio is smaller than 1.",
    )
    parser.add_argument(
        "--llm-source-dir",
        type=Path,
        default=script_root / "llm_representations" / "hicervix_level2",
        help="Source directory containing the existing hicervix_level2 .pth files.",
    )
    parser.add_argument(
        "--llm-output-dir",
        type=Path,
        default=script_root / "llm_representations" / "hicervix_5cls",
        help="Output directory for the extracted five-class .pth files.",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip extracting llm_representations and only write the dataset split JSON.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output JSON and llm_representations target files.",
    )
    return parser.parse_args()


def resolve_level2_label(row: Dict[str, str]) -> str:
    level_2 = (row.get("level_2") or "").strip()
    if level_2:
        return level_2
    return row["class_name"].strip()


def read_original_split_rows(dataset_root: Path, split: str) -> List[Dict[str, str]]:
    csv_path = dataset_root / f"{split}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Expected original split CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

    return rows


def filter_target_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    filtered = [row for row in rows if resolve_level2_label(row) in TARGET_CLASSES]
    if not filtered:
        raise ValueError("No rows matched the requested five classes.")
    return filtered


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
    for classname in TARGET_CLASSES:
        indices = indices_by_class.get(classname, [])
        if not indices:
            continue
        keep = max(1, int(len(indices) * sample_ratio))
        if keep >= len(indices):
            kept_indices.update(indices)
            continue
        kept_indices.update(rng.sample(indices, keep))

    return [item for index, item in enumerate(items) if index in kept_indices]


def maybe_remove_existing(path: Path, overwrite: bool) -> None:
    if not path.exists():
        return
    if not overwrite:
        raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def extract_llm_subset(
    source_dir: Path,
    output_dir: Path,
    overwrite: bool,
) -> None:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"LLM source directory not found: {source_dir}")

    maybe_remove_existing(output_dir, overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)

    for classname in TARGET_CLASSES:
        src = source_dir / f"{classname}.pth"
        if not src.is_file():
            raise FileNotFoundError(f"Expected LLM representation not found: {src}")
        shutil.copy2(src, output_dir / src.name)

    metadata_path = source_dir / "metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        classes = metadata.get("classes", [])
        class_to_entry = {entry.get("classname"): entry for entry in classes}
        metadata["classes"] = [class_to_entry[classname] for classname in TARGET_CLASSES]
        metadata["source_subdir"] = source_dir.name
        metadata["output_subdir"] = output_dir.name
        metadata["label_order"] = list(TARGET_CLASSES)
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def main() -> None:
    args = parse_args()
    if not 0 < args.sample_ratio <= 1:
        raise ValueError("--sample-ratio must be in the range (0, 1].")

    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    output_json = args.output_json or (dataset_root / "hicervix_5cls.json")
    output_json = output_json.expanduser().resolve()
    maybe_remove_existing(output_json, args.overwrite)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    label_to_index = {label: index for index, label in enumerate(TARGET_CLASSES)}
    rng = random.Random(args.seed)

    payload = {}
    for split in SPLITS:
        original_rows = read_original_split_rows(dataset_root, split)
        filtered_rows = filter_target_rows(original_rows)
        items = build_items(dataset_root, split, filtered_rows, label_to_index)
        payload[split] = subsample_items(items, args.sample_ratio, rng)

    output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if not args.skip_llm:
        extract_llm_subset(
            source_dir=args.llm_source_dir.expanduser().resolve(),
            output_dir=args.llm_output_dir.expanduser().resolve(),
            overwrite=args.overwrite,
        )

    print(f"Saved split to {output_json}")
    print(f"classes ({len(TARGET_CLASSES)}): {list(TARGET_CLASSES)}")
    print(f"sample_ratio: {args.sample_ratio}")
    print(f"seed: {args.seed}")
    for split in SPLITS:
        print(f"{split}: {len(payload[split])} {summarize(payload[split])}")
    if args.skip_llm:
        print("Skipped LLM representation extraction.")
    else:
        print(f"Extracted LLM representations to {args.llm_output_dir.expanduser().resolve()}")


if __name__ == "__main__":
    main()
