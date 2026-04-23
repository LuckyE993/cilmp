#!/usr/bin/env python3
"""Convert raw Kvasir-v2 files into the CILMP dataset layout.

Input (raw Kvasir-v2 release):
  kvasir-v2-a-gastrointestinal-tract-dataset/
    dyed-lifted-polyps/dyed-lifted-polyps/*.jpg
    dyed-resection-margins/dyed-resection-margins/*.jpg
    esophagitis/esophagitis/*.jpg
    normal-cecum/normal-cecum/*.jpg
    normal-pylorus/normal-pylorus/*.jpg
    normal-z-line/normal-z-line/*.jpg
    polyps/polyps/*.jpg
    ulcerative-colitis/ulcerative-colitis/*.jpg

Output (CILMP expects):
  kvasir/
    images/<class-slug>/*.jpg
    kvasir.json
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


CLASS_DIRS: Tuple[Tuple[str, str], ...] = (
    ("dyed-lifted-polyps", "dyed lifted polyps"),
    ("dyed-resection-margins", "dyed resection margins"),
    ("esophagitis", "esophagitis"),
    ("normal-cecum", "normal cecum"),
    ("normal-pylorus", "normal pylorus"),
    ("normal-z-line", "normal z-line"),
    ("polyps", "polyps"),
    ("ulcerative-colitis", "ulcerative colitis"),
)

LABEL_TO_INDEX = {
    classname: idx for idx, (_, classname) in enumerate(CLASS_DIRS)
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a CILMP-compatible Kvasir dataset directory from raw Kvasir-v2 files."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Path to the raw Kvasir-v2 directory.",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        required=True,
        help="Path to the output kvasir directory used by CILMP.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Per-class train split ratio.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Per-class validation split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed used for deterministic per-class shuffling.",
    )
    parser.add_argument(
        "--image-mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="How to place images under target-root/images.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing kvasir.json file and recreate images links when needed.",
    )
    return parser.parse_args()


def validate_ratios(train_ratio: float, val_ratio: float) -> None:
    if train_ratio <= 0 or val_ratio <= 0:
        raise ValueError("train_ratio and val_ratio must both be positive.")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be smaller than 1.")


def collect_class_images(source_root: Path, class_dir: str) -> List[Path]:
    image_root = source_root / class_dir / class_dir
    if not image_root.is_dir():
        raise FileNotFoundError(f"Expected Kvasir class directory not found: {image_root}")

    images = sorted(path for path in image_root.iterdir() if path.is_file())
    if not images:
        raise FileNotFoundError(f"No images found in: {image_root}")
    return images


def compute_split_sizes(total: int, train_ratio: float, val_ratio: float) -> Tuple[int, int, int]:
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    n_test = total - n_train - n_val

    if min(n_train, n_val, n_test) <= 0:
        raise ValueError(
            f"Invalid split sizes for total={total}: "
            f"train={n_train}, val={n_val}, test={n_test}"
        )

    return n_train, n_val, n_test


def build_items(
    image_paths: Sequence[Path],
    class_dir: str,
    classname: str,
    label: int,
) -> List[Tuple[str, int, str]]:
    return [
        (str(Path("images") / class_dir / image_path.name), label, classname)
        for image_path in image_paths
    ]


def split_class_images(
    image_paths: Sequence[Path],
    train_ratio: float,
    val_ratio: float,
    rng: random.Random,
) -> Tuple[List[Path], List[Path], List[Path]]:
    shuffled = list(image_paths)
    rng.shuffle(shuffled)

    n_train, n_val, _ = compute_split_sizes(len(shuffled), train_ratio, val_ratio)
    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    return train, val, test


def ensure_clean_dir(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if path.is_dir():
        shutil.rmtree(path)


def mirror_class_images(source_root: Path, target_root: Path, image_mode: str, overwrite: bool) -> None:
    target_images = target_root / "images"
    target_images.mkdir(parents=True, exist_ok=True)

    for class_dir, _ in CLASS_DIRS:
        source_dir = source_root / class_dir / class_dir
        target_dir = target_images / class_dir

        if target_dir.exists() or target_dir.is_symlink():
            if not overwrite:
                continue
            ensure_clean_dir(target_dir)

        if image_mode == "symlink":
            target_dir.symlink_to(source_dir, target_is_directory=True)
        else:
            shutil.copytree(source_dir, target_dir)


def summarize_split(split: Iterable[Tuple[str, int, str]]) -> Dict[str, int]:
    return dict(Counter(item[2] for item in split))


def main() -> None:
    args = parse_args()
    validate_ratios(args.train_ratio, args.val_ratio)

    source_root = args.source_root.expanduser().resolve()
    target_root = args.target_root.expanduser().resolve()
    target_root.mkdir(parents=True, exist_ok=True)

    json_path = target_root / "kvasir.json"
    if json_path.exists() and not args.overwrite:
        raise FileExistsError(f"{json_path} already exists. Use --overwrite to replace it.")

    rng = random.Random(args.seed)
    train: List[Tuple[str, int, str]] = []
    val: List[Tuple[str, int, str]] = []
    test: List[Tuple[str, int, str]] = []

    for class_dir, classname in CLASS_DIRS:
        image_paths = collect_class_images(source_root, class_dir)
        split_train, split_val, split_test = split_class_images(
            image_paths=image_paths,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            rng=rng,
        )
        label = LABEL_TO_INDEX[classname]
        train.extend(build_items(split_train, class_dir, classname, label))
        val.extend(build_items(split_val, class_dir, classname, label))
        test.extend(build_items(split_test, class_dir, classname, label))

    payload = {"train": train, "val": val, "test": test}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    mirror_class_images(
        source_root=source_root,
        target_root=target_root,
        image_mode=args.image_mode,
        overwrite=args.overwrite,
    )

    print(f"Saved split to {json_path}")
    print("train:", len(train), summarize_split(train))
    print("val:", len(val), summarize_split(val))
    print("test:", len(test), summarize_split(test))


if __name__ == "__main__":
    main()
