#!/usr/bin/env python3
"""Convert a raw APTOS 2019 release into the CILMP dataset layout.

Input (raw APTOS 2019 split release):
  aptos2019/
    train_1.csv
    valid.csv
    test.csv
    train_images/
    val_images/
    test_images/

Output (CILMP expects):
  aptos2019/
    train_images -> symlink or reused directory
    val_images -> symlink or reused directory
    test_images -> symlink or reused directory
    aptos2019.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


CLASSNAMES = (
    "no diabetic retinopathy",
    "mild diabetic retinopathy",
    "moderate diabetic retinopathy",
    "severe diabetic retinopathy",
    "proliferative diabetic retinopathy",
)

LABEL_TO_CLASS = {index: name for index, name in enumerate(CLASSNAMES)}

CSV_CANDIDATES = {
    "train": ("train_1.csv", "train.csv"),
    "val": ("valid.csv", "val.csv"),
    "test": ("test.csv",),
}

IMAGE_DIRS = {
    "train": "train_images",
    "val": "val_images",
    "test": "test_images",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a CILMP-compatible aptos2019 directory with symlinked images."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("~/qixuan/datasets/aptos2019"),
        help="Path to the raw aptos2019 dataset directory.",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        required=True,
        help="Path to the output aptos2019 directory used by CILMP.",
    )
    parser.add_argument(
        "--image-mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="How to place image split directories under target-root.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing aptos2019.json file and mismatched links.",
    )
    return parser.parse_args()


def locate_csv(source_root: Path, split_name: str) -> Path:
    for candidate in CSV_CANDIDATES[split_name]:
        path = source_root / candidate
        if path.exists():
            return path
    candidates = ", ".join(CSV_CANDIDATES[split_name])
    raise FileNotFoundError(f"Could not find {split_name} CSV under {source_root}: {candidates}")


def build_image_index(source_root: Path, image_dir_name: str) -> Dict[str, str]:
    image_root = source_root / image_dir_name
    if not image_root.exists():
        raise FileNotFoundError(f"Missing image directory: {image_root}")

    index: Dict[str, str] = {}
    for path in image_root.rglob("*"):
        if not path.is_file():
            continue
        relative_path = path.relative_to(source_root).as_posix()
        key = path.stem
        if key in index:
            raise ValueError(f"Duplicate image stem '{key}' found under {image_root}")
        index[key] = relative_path

    if not index:
        raise FileNotFoundError(f"No images found under {image_root}")

    return index


def convert_split(
    csv_path: Path,
    image_index: Dict[str, str],
) -> List[Tuple[str, int, str]]:
    items: List[Tuple[str, int, str]] = []

    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_id = row["id_code"].strip()
            label = int(row["diagnosis"])
            classname = LABEL_TO_CLASS[label]
            try:
                impath = image_index[image_id]
            except KeyError as exc:
                raise FileNotFoundError(f"Missing image for id_code '{image_id}' from {csv_path}") from exc
            items.append((impath, label, classname))

    return items


def ensure_images_dir(source_dir: Path, target_dir: Path, image_mode: str, overwrite: bool) -> None:
    if target_dir.exists() or target_dir.is_symlink():
        if target_dir.resolve() == source_dir.resolve():
            return
        if not overwrite:
            raise FileExistsError(f"{target_dir} already exists. Use --overwrite to replace it.")
        if target_dir.is_symlink():
            target_dir.unlink()
        else:
            raise FileExistsError(
                f"{target_dir} exists and is not a symlink. Remove it manually or choose another target."
            )

    if image_mode == "symlink":
        target_dir.symlink_to(source_dir, target_is_directory=True)
        return

    import shutil

    shutil.copytree(source_dir, target_dir)


def summarize_split(split: Sequence[Tuple[str, int, str]]) -> Counter:
    return Counter(item[2] for item in split)


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    target_root = args.target_root.expanduser().resolve()
    target_root.mkdir(parents=True, exist_ok=True)

    json_path = target_root / "aptos2019.json"
    if json_path.exists() and not args.overwrite:
        raise FileExistsError(f"{json_path} already exists. Use --overwrite to replace it.")

    train = convert_split(
        locate_csv(source_root, "train"),
        build_image_index(source_root, IMAGE_DIRS["train"]),
    )
    val = convert_split(
        locate_csv(source_root, "val"),
        build_image_index(source_root, IMAGE_DIRS["val"]),
    )
    test = convert_split(
        locate_csv(source_root, "test"),
        build_image_index(source_root, IMAGE_DIRS["test"]),
    )

    payload = {
        "train": train,
        "val": val,
        "test": test,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    for image_dir_name in IMAGE_DIRS.values():
        ensure_images_dir(
            source_root / image_dir_name,
            target_root / image_dir_name,
            args.image_mode,
            args.overwrite,
        )

    print(f"Saved split to {json_path}")
    print("train:", len(train), dict(summarize_split(train)))
    print("val:", len(val), dict(summarize_split(val)))
    print("test:", len(test), dict(summarize_split(test)))


if __name__ == "__main__":
    main()
