#!/usr/bin/env python3
"""Convert raw Derm7pt release_v0 files into the CILMP dataset layout.

Input (raw Derm7pt release):
  release_v0/
    images/
    meta/meta.csv
    meta/train_indexes.csv
    meta/valid_indexes.csv
    meta/test_indexes.csv

Output (CILMP expects):
  derm7pt/
    images -> symlink or copied images directory
    derm7pt.json

The CILMP code uses five coarse classes:
  carcinoma, keratosis, melanoma, miscellaneous, nevus

This script converts the original Derm7pt diagnosis taxonomy into those five
classes using the standard Derm7pt 5-class grouping:
  BCC -> carcinoma
  SK  -> keratosis
  MEL -> melanoma
  NV  -> nevus
  the remaining diagnoses -> miscellaneous
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


CLASSNAMES = (
    "carcinoma",
    "keratosis",
    "melanoma",
    "miscellaneous",
    "nevus",
)

RAW_TO_COARSE = {
    "basal cell carcinoma": "carcinoma",
    "seborrheic keratosis": "keratosis",
    "melanoma": "melanoma",
    "melanoma (in situ)": "melanoma",
    "melanoma (less than 0.76 mm)": "melanoma",
    "melanoma (0.76 to 1.5 mm)": "melanoma",
    "melanoma (more than 1.5 mm)": "melanoma",
    "melanoma metastasis": "melanoma",
    "blue nevus": "nevus",
    "clark nevus": "nevus",
    "combined nevus": "nevus",
    "congenital nevus": "nevus",
    "dermal nevus": "nevus",
    "recurrent nevus": "nevus",
    "reed or spitz nevus": "nevus",
    "dermatofibroma": "miscellaneous",
    "lentigo": "miscellaneous",
    "melanosis": "miscellaneous",
    "miscellaneous": "miscellaneous",
    "vascular lesion": "miscellaneous",
}

LABEL_TO_INDEX = {name: idx for idx, name in enumerate(CLASSNAMES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a CILMP-compatible Derm7pt dataset directory from release_v0."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Path to the raw Derm7pt release_v0 directory.",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        required=True,
        help="Path to the output derm7pt directory used by CILMP.",
    )
    parser.add_argument(
        "--image-mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="How to place the images directory under target-root.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing derm7pt.json file.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_indexes(path: Path) -> List[int]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [int(row["indexes"]) for row in reader]


def convert_split(
    rows: Sequence[Dict[str, str]],
    indexes: Iterable[int],
) -> List[Tuple[str, int, str]]:
    items: List[Tuple[str, int, str]] = []
    for index in indexes:
        row = rows[index]
        raw_diagnosis = row["diagnosis"].strip()
        if raw_diagnosis not in RAW_TO_COARSE:
            raise KeyError(f"Unmapped Derm7pt diagnosis: {raw_diagnosis}")
        coarse = RAW_TO_COARSE[raw_diagnosis]
        label = LABEL_TO_INDEX[coarse]
        impath = os.path.join("images", row["derm"].strip())
        items.append((impath, label, coarse))
    return items


def ensure_images_dir(source_images: Path, target_images: Path, image_mode: str) -> None:
    if target_images.exists() or target_images.is_symlink():
        return

    if image_mode == "symlink":
        target_images.symlink_to(source_images, target_is_directory=True)
        return

    import shutil

    shutil.copytree(source_images, target_images)


def summarize_split(split: Sequence[Tuple[str, int, str]]) -> Counter:
    return Counter(item[2] for item in split)


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    target_root = args.target_root.expanduser().resolve()

    source_images = source_root / "images"
    source_meta = source_root / "meta"
    target_root.mkdir(parents=True, exist_ok=True)

    json_path = target_root / "derm7pt.json"
    if json_path.exists() and not args.overwrite:
        raise FileExistsError(f"{json_path} already exists. Use --overwrite to replace it.")

    rows = read_csv_rows(source_meta / "meta.csv")
    train_indexes = read_indexes(source_meta / "train_indexes.csv")
    valid_indexes = read_indexes(source_meta / "valid_indexes.csv")
    test_indexes = read_indexes(source_meta / "test_indexes.csv")

    train = convert_split(rows, train_indexes)
    valid = convert_split(rows, valid_indexes)
    test = convert_split(rows, test_indexes)

    payload = {
        "train": train,
        "val": valid,
        "test": test,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    ensure_images_dir(source_images, target_root / "images", args.image_mode)

    print(f"Saved split to {json_path}")
    print("train:", len(train), dict(summarize_split(train)))
    print("val:", len(valid), dict(summarize_split(valid)))
    print("test:", len(test), dict(summarize_split(test)))


if __name__ == "__main__":
    main()
