"""Tier 3 — Sen-2 LULC preprocessing.

Sen-2 LULC provides Sentinel-2 imagery over the Indian subcontinent
with 7-class land use/land cover annotations. Used for contextual
regularization with Indian geography.

Spec treatment:
    - R1: B4/B3/B2 → RGB (already in source)
    - R2: percentile normalization (deferred)
    - R4: NOT dual-resolution (10m native)
    - Masks → bounding boxes (R6 Qwen format)
    - pair_type: single
    - dataset enum: "sen2lulc"
    - gsd_bucket: [GSD:10m]
    - task: vqa (LULC classification)

Sen-2 LULC structure:
    <input_dir>/
        images/
            <image_id>.png (Sentinel-2 RGB)
        masks/
            <image_id>.png (class mask)
        metadata.csv or annotations.json
            Each entry: {"image_id": ..., "label": ...}

7-class Indian taxonomy:
    0: Built-up
    1: Vegetation
    2: Water
    3: Barren
    4: Agriculture
    5: Forest
    6: Wetland

Usage:
    python -m preprocess.tier3_sen2lulc --input-dir <path> --output-dir <path>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from preprocess.common.bbox import pixel_to_normalized
from preprocess.common.gsd import assign_gsd_bucket
from preprocess.common.io import Manifest, append_jsonl, write_png
from preprocess.validator import validate_sample

_DATASET = "sen2lulc"

# 7-class Indian taxonomy
_LULC_CLASSES = [
    "Built-up",
    "Vegetation",
    "Water",
    "Barren",
    "Agriculture",
    "Forest",
    "Wetland",
]


def load_metadata(input_dir: Path) -> list[dict]:
    """Load Sen-2 LULC metadata from CSV or JSON."""
    # Try CSV first
    csv_path = input_dir / "metadata.csv"
    if csv_path.exists():
        entries = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                entries.append(row)
        return entries

    # Try JSON
    json_path = input_dir / "annotations.json"
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return data.get("annotations", data.get("data", []))

    return []


def mask_to_bbox(mask_path: Path) -> list[list[float]] | None:
    """Convert class mask to normalized bounding boxes for each class."""
    try:
        from PIL import Image
        mask = np.array(Image.open(str(mask_path)).convert("L"))
    except Exception:
        return None

    # Find bounding boxes for each class
    bboxes = []
    for class_id in range(len(_LULC_CLASSES)):
        class_mask = (mask == class_id)
        if class_mask.sum() == 0:
            continue

        ys, xs = np.where(class_mask)
        h, w = mask.shape

        bbox_pixel = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        bbox_norm = pixel_to_normalized(tuple(bbox_pixel), w, h)
        if bbox_norm:
            bboxes.append(bbox_norm[0])

    return bboxes if bboxes else None


def parse_sen2lulc_sample(
    entry: dict[str, Any],
    images_dir: Path | None,
    masks_dir: Path | None,
    sample_id: str,
) -> dict[str, Any] | None:
    """Parse a single Sen-2 LULC annotation into our JSONL format."""
    image_id = entry.get("image_id", entry.get("id", ""))
    label = entry.get("label", entry.get("class", ""))

    if not image_id:
        return None

    # Handle image path
    image_path_str = ""
    if images_dir:
        for ext in [".png", ".jpg", ".tif"]:
            img_path = images_dir / f"{image_id}{ext}"
            if img_path.exists():
                image_path_str = str(img_path)
                break

    # Get bounding boxes from mask
    bbox = None
    if masks_dir:
        for ext in [".png", ".tif"]:
            mask_path = masks_dir / f"{image_id}{ext}"
            if mask_path.exists():
                bbox = mask_to_bbox(mask_path)
                break

    # Build response from label
    if isinstance(label, int) and 0 <= label < len(_LULC_CLASSES):
        response = _LULC_CLASSES[label]
    elif isinstance(label, str) and label:
        response = label
    else:
        response = "Unknown land cover"

    return {
        "id": sample_id,
        "dataset": _DATASET,
        "task": "vqa",
        "image_path": [image_path_str] if image_path_str else [],
        "pair_type": "single",
        "gsd_bucket": "",  # Filled by caller
        "split": "train",
        "instruction": "What is the primary land cover type in this image?",
        "response": response,
        "bbox": bbox,
        "modality": "optical",
    }


def run_tier3_sen2lulc(
    input_dir: Path,
    output_dir: Path,
    *,
    max_samples: int | None = None,
    sample_fraction: float = 0.05,
    seed: int = 42,
) -> dict[str, int]:
    """Run Sen-2 LULC preprocessing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.parquet"
    jsonl_path = output_dir / "sen2lulc.jsonl"
    manifest = Manifest(manifest_path)

    native_bucket = assign_gsd_bucket("sen2lulc")

    annotations = load_metadata(input_dir)
    if not annotations:
        print(f"  [WARN] No annotations found in {input_dir}")
        return {"processed": 0, "skipped": 0, "failed": 0}

    images_dir = None
    for name in ["images", "Images", "img"]:
        candidate = input_dir / name
        if candidate.exists():
            images_dir = candidate
            break

    masks_dir = None
    for name in ["masks", " Masks", "label", "labels"]:
        candidate = input_dir / name
        if candidate.exists():
            masks_dir = candidate
            break

    total = len(annotations)
    if max_samples:
        total = min(total, max_samples)
    print(f"Loaded {len(annotations)} annotations, processing {total}")

    stats = {"processed": 0, "skipped": 0, "failed": 0}
    start = time.time()

    for i, entry in enumerate(annotations[:total]):
        image_id = entry.get("image_id", entry.get("id", i))
        sample_id = f"sen2lulc_{image_id}"

        if manifest.is_processed(sample_id):
            stats["skipped"] += 1
            continue

        sample = parse_sen2lulc_sample(entry, images_dir, masks_dir, sample_id)
        if sample is None:
            stats["failed"] += 1
            continue

        sample["gsd_bucket"] = native_bucket

        ok, errs = validate_sample(sample)
        if not ok:
            print(f"  [VALID] {sample_id}: {errs}", file=sys.stderr)
            stats["failed"] += 1
            continue

        append_jsonl(sample, jsonl_path)
        manifest.mark_processed(
            sample_id=sample_id,
            dataset=_DATASET,
            split="train",
            output_shard="local",
        )
        stats["processed"] += 1

        if (i + 1) % 5000 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  Processed {i + 1}/{total} ({rate:.1f} entries/s)")

    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s: {stats['processed']} processed, "
          f"{stats['skipped']} skipped, {stats['failed']} failed")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Sen-2 LULC")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_tier3_sen2lulc(
        args.input_dir,
        args.output_dir,
        max_samples=args.max_samples,
        sample_fraction=args.sample_fraction,
        seed=args.seed,
    )
