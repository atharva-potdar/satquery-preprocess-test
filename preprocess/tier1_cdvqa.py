"""Tier 1 — CDVQA preprocessing (Change Detection VQA).

CDVQA provides bi-temporal aerial imagery with question-answer pairs about
changes between dates. Uses SECOND imagery (0.5m-2m resolution).

Spec treatment:
    - R1: RGB channels
    - R2: percentile normalization (deferred)
    - R4: NOT dual-resolution — the Tier 1 table doesn't tag CDVQA with R4
      (unlike vrsbench/rsvqa_hr/levir_cd/sn6_opt), and SECOND imagery spans
      ~0.5-2m per scene rather than one fixed native GSD anyway.
    - pair_type: bitemporal
    - dataset enum: "cdvqa"
    - gsd_bucket: VHR-native (categorical — see common/gsd.py)
    - task: change_vqa

CDVQA structure:
    <input_dir>/
        train.json / val.json
            Each entry: {"question": ..., "answer": ...,
                         "before": "path/to/before.png",
                         "after": "path/to/after.png"}
        images/
            <before/after images>

IMPORTANT: Load ONLY train/val split files. Test splits are EXCLUDED.

Usage:
    python -m preprocess.tier1_cdvqa --input-dir <path> --output-dir <path>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from preprocess.common.gsd import assign_gsd_bucket
from preprocess.common.io import Manifest, append_jsonl
from preprocess.validator import validate_sample

_DATASET = "cdvqa"
_VALID_SPLITS = {"train", "val"}


def load_split(input_dir: Path, split_name: str) -> list[dict]:
    """Load a CDVQA split file (train.json or val.json).

    Asserts the split name is valid (train/val only).
    """
    if split_name not in _VALID_SPLITS:
        raise ValueError(f"Invalid split '{split_name}'. Must be one of {_VALID_SPLITS}")

    path = input_dir / f"{split_name}.json"
    if not path.exists():
        return []

    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        return data.get("questions", data.get("data", []))
    return []


def parse_cdvqa_sample(
    entry: dict[str, Any],
    images_dir: Path | None,
    sample_id: str,
) -> dict[str, Any] | None:
    """Parse a single CDVQA annotation into our JSONL format."""
    question = entry.get("question", "")
    answer = entry.get("answer", "")
    before_path = entry.get("before", "")
    after_path = entry.get("after", "")

    if not question or not answer:
        return None
    if not before_path or not after_path:
        return None

    # Resolve image paths
    image_paths = []
    if images_dir:
        for rel_path in [before_path, after_path]:
            full_path = images_dir / rel_path
            if full_path.exists():
                image_paths.append(str(full_path))
            else:
                image_paths.append("")
    else:
        image_paths = ["", ""]

    return {
        "id": sample_id,
        "dataset": _DATASET,
        "task": "change_vqa",
        "image_path": image_paths,
        "pair_type": "bitemporal",
        "gsd_bucket": "",  # Filled by caller
        "split": "train",
        "instruction": question,
        "response": str(answer),
        "bbox": None,
        "modality": "optical",
    }


def run_tier1_cdvqa(
    input_dir: Path,
    output_dir: Path,
    *,
    max_samples: int | None = None,
) -> dict[str, int]:
    """Run CDVQA preprocessing with manifest-based resumability.

    Loads ONLY train/val split files. Test splits are rejected.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.parquet"
    jsonl_path = output_dir / "cdvqa.jsonl"
    manifest = Manifest(manifest_path)

    native_bucket = assign_gsd_bucket("cdvqa")

    # Load train + val only
    annotations = []
    for split_name in ["train", "val"]:
        split_data = load_split(input_dir, split_name)
        for entry in split_data:
            entry["_split"] = split_name
        annotations.extend(split_data)

    if not annotations:
        print(f"  [WARN] No annotations found in {input_dir}")
        return {"processed": 0, "skipped": 0, "failed": 0}

    images_dir = None
    for name in ["images", "Images", "SECOND"]:
        candidate = input_dir / name
        if candidate.exists():
            images_dir = candidate
            break

    total = len(annotations)
    if max_samples:
        total = min(total, max_samples)
    print(f"Loaded {len(annotations)} annotations (train+val), processing {total}")

    stats = {"processed": 0, "skipped": 0, "failed": 0}
    start = time.time()

    for i, entry in enumerate(annotations[:total]):
        sample_id = f"cdvqa_{i:07d}"

        if manifest.is_processed(sample_id):
            stats["skipped"] += 1
            continue

        sample = parse_cdvqa_sample(entry, images_dir, sample_id)
        if sample is None:
            stats["failed"] += 1
            continue

        sample["gsd_bucket"] = native_bucket
        # All CDVQA samples enter as "train"; the val_internal carve-out
        # is handled downstream by split_internal_val.py.  The raw "_split"
        # value "val" is not in the schema enum ["train", "val_internal"].
        sample["split"] = "train"

        ok, errs = validate_sample(sample)
        if not ok:
            print(f"  [VALID] {sample_id}: {errs}", file=sys.stderr)
            stats["failed"] += 1
            continue

        append_jsonl(sample, jsonl_path)
        manifest.mark_processed(
            sample_id=sample_id,
            dataset=_DATASET,
            split=sample["split"],
            output_shard="local",
        )
        stats["processed"] += 1

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  Processed {i + 1}/{total} ({rate:.1f} entries/s)")

    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s: {stats['processed']} processed, "
          f"{stats['skipped']} skipped, {stats['failed']} failed")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess CDVQA")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    run_tier1_cdvqa(args.input_dir, args.output_dir, max_samples=args.max_samples)
