"""Tier 2 — SARDet-100K preprocessing.

SARDet-100K provides 116K SAR images with object detection annotations
across 6 categories: ship, aircraft, bridge, tank, car, harbor.

Spec treatment:
    - R1/R3: SAR pseudo-RGB (VV, VH, VV-VH)
    - R5: GSD calibration to RISAT proxy band (~2-10m)
    - pair_type: single
    - dataset enum: "sardet"
    - gsd_bucket: RISAT-proxy[GSD:2m]
    - task: grounding (detection boxes)

SARDet-100K structure:
    <input_dir>/
        images/
            <image_id>.png (or .jpg)
        labels/
            <image_id>.txt (YOLO format: class cx cy w h)

Usage:
    python -m preprocess.tier2_sardet --input-dir <path> --output-dir <path>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from preprocess.common.bbox import pixel_to_normalized
from preprocess.common.gsd import assign_gsd_bucket
from preprocess.common.io import Manifest, append_jsonl, write_png
from preprocess.common.sample import stratified_sample
from preprocess.common.sar import sar_intensity_pseudo_gray
from preprocess.validator import validate_sample

_DATASET = "sardet"
_CATEGORIES = ["ship", "aircraft", "bridge", "tank", "car", "harbor"]


def primary_class(label_path: Path) -> str:
    """First annotated class in a YOLO label file — the stratification
    key for "stratified uniform across 6 categories"."""
    if not label_path.exists():
        return "unknown"
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                class_id = int(parts[0])
            except ValueError:
                continue
            if 0 <= class_id < len(_CATEGORIES):
                return _CATEGORIES[class_id]
    return "unknown"


def load_yolo_labels(
    label_path: Path,
    img_width: int,
    img_height: int,
) -> list[dict[str, Any]]:
    """Load YOLO format labels and convert to normalized bboxes.

    Returns list of {class_name, bbox: [[x1,y1,x2,y2]]}.
    """
    if not label_path.exists():
        return []

    annotations = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            if class_id >= len(_CATEGORIES):
                continue

            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

            # Convert from normalized center to pixel coords
            x_min = int((cx - w / 2) * img_width)
            y_min = int((cy - h / 2) * img_height)
            x_max = int((cx + w / 2) * img_width)
            y_max = int((cy + h / 2) * img_height)

            # Clamp
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(img_width, x_max)
            y_max = min(img_height, y_max)

            bbox_norm = pixel_to_normalized(
                (x_min, y_min, x_max, y_max),
                img_width,
                img_height,
            )
            if bbox_norm:
                annotations.append({
                    "class_name": _CATEGORIES[class_id],
                    "bbox": bbox_norm[0],
                })

    return annotations


def process_sardet_sample(
    image_path: Path,
    label_path: Path,
    output_dir: Path,
    sample_id: str,
) -> dict[str, Any] | None:
    """Process a single SARDet-100K sample."""
    try:
        from PIL import Image
        img = Image.open(str(image_path))
        if img.mode != "L":
            img = img.convert("L")  # SAR is single-channel

        # SARDet-100K ships single-channel intensity PNGs — no separate
        # VV/VH. R3's dual-pol B=(VV-VH) needs two real channels, so we
        # don't fabricate a second one; see sar_intensity_pseudo_gray().
        arr = np.array(img).astype(np.float32)
        rgb = sar_intensity_pseudo_gray(arr)
        w, h = rgb.shape[1], rgb.shape[0]
    except Exception as e:
        print(f"  [FAIL] {sample_id}: {e}", file=sys.stderr)
        return None

    # Save PNG
    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    out_path = img_dir / f"{sample_id}.png"
    write_png(rgb, out_path)

    # Load labels
    annotations = load_yolo_labels(label_path, w, h)
    if not annotations:
        return None

    # Use first bbox for primary task
    primary = annotations[0]
    all_bboxes = [a["bbox"] for a in annotations]

    class_names = list(set(a["class_name"] for a in annotations))
    response = ", ".join(class_names)

    return {
        "id": sample_id,
        "dataset": _DATASET,
        "task": "grounding",
        "image_path": [str(out_path)],
        "pair_type": "single",
        "gsd_bucket": "",  # Filled by caller
        "split": "train",
        "instruction": "Locate the objects in this SAR image.",
        "response": response,
        "bbox": all_bboxes,
        "modality": "sar",
    }


def run_tier2_sardet(
    input_dir: Path,
    output_dir: Path,
    *,
    max_samples: int | None = None,
    sample_fraction: float = 0.10,
    seed: int = 42,
) -> dict[str, int]:
    """Run SARDet-100K preprocessing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.parquet"
    jsonl_path = output_dir / "sardet.jsonl"
    manifest = Manifest(manifest_path)

    sar_bucket = assign_gsd_bucket("sardet")

    images_dir = input_dir / "images"
    labels_dir = input_dir / "labels"

    if not images_dir.exists():
        print(f"  [WARN] Images directory not found: {images_dir}")
        return {"processed": 0, "skipped": 0, "failed": 0}

    # Find all images
    image_paths = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg"))
    print(f"Found {len(image_paths)} SARDet-100K images")

    # Selection (spec): "stratified uniform across 6 categories".
    if sample_fraction < 1.0 and image_paths:
        keyed = [
            (p, primary_class(labels_dir / f"{p.stem}.txt"))
            for p in image_paths
        ]
        image_paths = [
            p for p, _ in stratified_sample(
                keyed, key_fn=lambda item: item[1],
                fraction=sample_fraction, seed=seed,
            )
        ]

    total = len(image_paths)
    if max_samples:
        total = min(total, max_samples)

    stats = {"processed": 0, "skipped": 0, "failed": 0}
    start = time.time()

    for i, image_path in enumerate(image_paths[:total]):
        sample_id = f"sardet_{image_path.stem}"

        if manifest.is_processed(sample_id):
            stats["skipped"] += 1
            continue

        label_path = labels_dir / f"{image_path.stem}.txt"
        sample = process_sardet_sample(image_path, label_path, output_dir, sample_id)
        if sample is None:
            stats["failed"] += 1
            continue

        sample["gsd_bucket"] = sar_bucket

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

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  Processed {i + 1}/{total} ({rate:.1f} entries/s)")

    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s: {stats['processed']} processed, "
          f"{stats['skipped']} skipped, {stats['failed']} failed")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess SARDet-100K")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_tier2_sardet(
        args.input_dir,
        args.output_dir,
        max_samples=args.max_samples,
        sample_fraction=args.sample_fraction,
        seed=args.seed,
    )
