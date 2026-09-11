"""Split internal validation carve-out.

Carves 2-3% stratified val_internal splits from processed training JSONLs.
Stratification is by (dataset, task) to ensure proportional representation.

Usage:
    python -m preprocess.split_internal_val --jsonl-dir data/oscd_output --val-fraction 0.03
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from preprocess.common.io import append_jsonl


def load_samples(jsonl_path: Path) -> list[dict]:
    """Load all samples from a JSONL file."""
    samples = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def stratified_split(
    samples: list[dict],
    val_fraction: float = 0.03,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split samples into train and val_internal, stratified by (dataset, task).

    Returns (train_samples, val_samples).
    """
    import random

    # Group by (dataset, task) for stratification
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for s in samples:
        key = (s.get("dataset", ""), s.get("task", ""))
        groups[key].append(s)

    rng = random.Random(seed)
    train_samples = []
    val_samples = []

    for key, group in groups.items():
        rng.shuffle(group)
        n_val = max(1, int(len(group) * val_fraction))
        # Cap val at len-1 to ensure at least 1 train sample
        n_val = min(n_val, len(group) - 1) if len(group) > 1 else 0
        val_samples.extend(group[:n_val])
        train_samples.extend(group[n_val:])

    return train_samples, val_samples


def split_and_write(
    input_jsonl: Path,
    output_dir: Path,
    *,
    val_fraction: float = 0.03,
    seed: int = 42,
) -> dict[str, int]:
    """Split a single JSONL into train and val_internal, writing both.

    Returns stats: {total, train, val}.
    """
    samples = load_samples(input_jsonl)
    if not samples:
        return {"total": 0, "train": 0, "val": 0}

    train, val = stratified_split(samples, val_fraction=val_fraction, seed=seed)

    # Tag val samples
    for s in val:
        s["split"] = "val_internal"

    # Write outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_jsonl.stem

    train_path = output_dir / f"{stem}_train.jsonl"
    val_path = output_dir / f"{stem}_val_internal.jsonl"

    for s in train:
        append_jsonl(s, train_path)
    for s in val:
        append_jsonl(s, val_path)

    return {"total": len(samples), "train": len(train), "val": len(val)}


def split_all_tiers(
    jsonl_dir: Path,
    *,
    val_fraction: float = 0.03,
    seed: int = 42,
) -> dict[str, dict[str, int]]:
    """Split all *_train.jsonl or *.jsonl files in a directory.

    Processes every JSONL in the directory, creating _train.jsonl and
    _val_internal.jsonl pairs.

    Returns stats per input file.
    """
    results = {}
    for jsonl_path in sorted(jsonl_dir.glob("*.jsonl")):
        if "_train" in jsonl_path.stem or "_val" in jsonl_path.stem:
            continue  # Skip already-split files
        stats = split_and_write(
            jsonl_path, jsonl_dir,
            val_fraction=val_fraction, seed=seed,
        )
        results[jsonl_path.name] = stats
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Carve stratified val_internal splits from processed JSONLs",
    )
    parser.add_argument(
        "--jsonl-dir", type=Path, required=True,
        help="Directory containing processed JSONL files",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.03,
        help="Fraction for val split (default: 0.03 = 3%%)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    results = split_all_tiers(
        args.jsonl_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    for name, stats in results.items():
        print(f"{name}: {stats['total']} total → {stats['train']} train + {stats['val']} val")
