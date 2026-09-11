"""Tier 2 — SpaceNet 6 SAR preprocessing.

SpaceNet 6 provides co-registered Sentinel-1 SAR imagery alongside
optical WorldView-2. Used for cross-modal fusion + SAR artifacts.

Spec treatment:
    - R1/R3: Quad-pol SAR pseudo-RGB (R=HH, G=VV, B=VH)
    - R5: downsampled to RISAT proxy band (~2-10m)
    - pair_type: single (SAR only, optical handled by tier1_sn6_opt)
    - dataset enum: "sn6_sar"
    - gsd_bucket: RISAT-proxy[GSD:2m]
    - task: vqa (SAR interpretation)

SpaceNet 6 SAR structure:
    <input_dir>/
        train/
            sar/
                <tile_id>.tif (Sentinel-1 SAR)

Usage:
    python -m preprocess.tier2_sn6_sar --input-dir <path> --output-dir <path>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from preprocess.common.gsd import assign_gsd_bucket
from preprocess.common.io import Manifest, append_jsonl, write_png
from preprocess.common.sar import sar_pseudo_rgb
from preprocess.validator import validate_sample

_DATASET = "sn6_sar"


def process_sar_tile(
    tile_path: Path,
    output_dir: Path,
    tile_id: str,
) -> dict[str, Any] | None:
    """Process a single SpaceNet 6 SAR tile."""
    try:
        import tifffile
        img = tifffile.imread(str(tile_path))
    except Exception as e:
        print(f"  [FAIL] {tile_id}: {e}", file=sys.stderr)
        return None

    # Ensure 2D or 3D
    if img.ndim == 3:
        # Multi-band SAR: take first 3 bands or combine
        if img.shape[0] >= 3:
            hh = img[0].astype(np.float32)
            vv = img[1].astype(np.float32)
            vh = img[2].astype(np.float32)
        else:
            hh = img[0].astype(np.float32)
            vv = hh
            vh = hh
    elif img.ndim == 2:
        hh = img.astype(np.float32)
        vv = hh
        vh = hh * 0.8
    else:
        return None

    # Convert to pseudo-RGB using quad-pol mapping
    # R=HH, G=VV, B=VH
    rgb = np.stack([hh, vv, vh], axis=-1)

    # Normalize to uint8
    for c in range(3):
        ch = rgb[:, :, c]
        p98 = np.percentile(ch, 98)
        if p98 > 0:
            rgb[:, :, c] = np.clip(ch, 0, p98) / p98 * 255.0
    rgb = rgb.astype(np.uint8)

    # Save PNG
    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    out_path = img_dir / f"sn6_sar_{tile_id}.png"
    write_png(rgb, out_path)

    sample_id = f"sn6_sar_{tile_id}"

    return {
        "id": sample_id,
        "dataset": _DATASET,
        "task": "vqa",
        "image_path": [str(out_path)],
        "pair_type": "single",
        "gsd_bucket": "",  # Filled by caller
        "split": "train",
        "instruction": "Describe the SAR characteristics of this scene.",
        "response": f"SAR image of {tile_id} showing surface scattering properties.",
        "bbox": None,
        "modality": "sar",
    }


def run_tier2_sn6_sar(
    input_dir: Path,
    output_dir: Path,
    *,
    max_samples: int | None = None,
    sample_fraction: float = 0.05,
    seed: int = 42,
) -> dict[str, int]:
    """Run SpaceNet 6 SAR preprocessing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.parquet"
    jsonl_path = output_dir / "sn6_sar.jsonl"
    manifest = Manifest(manifest_path)

    sar_bucket = assign_gsd_bucket("sn6_sar")

    # Find SAR tiles
    sar_dirs = [
        input_dir / "train" / "sar",
        input_dir / "sar",
        input_dir / "Sentinel-1",
    ]

    tile_paths = []
    for sar_dir in sar_dirs:
        if sar_dir.exists():
            tile_paths = sorted(sar_dir.glob("*.tif"))
            break

    if not tile_paths:
        print(f"  [WARN] No SAR tiles found in {input_dir}")
        return {"processed": 0, "skipped": 0, "failed": 0}

    print(f"Found {len(tile_paths)} SpaceNet 6 SAR tiles")

    # Subsample
    import random
    rng = random.Random(seed)
    if sample_fraction < 1.0:
        n_sample = max(1, int(len(tile_paths) * sample_fraction))
        tile_paths = rng.sample(tile_paths, min(n_sample, len(tile_paths)))

    total = len(tile_paths)
    if max_samples:
        total = min(total, max_samples)

    stats = {"processed": 0, "skipped": 0, "failed": 0}
    start = time.time()

    for i, tile_path in enumerate(tile_paths[:total]):
        tile_id = tile_path.stem
        sample_id = f"sn6_sar_{tile_id}"

        if manifest.is_processed(sample_id):
            stats["skipped"] += 1
            continue

        sample = process_sar_tile(tile_path, output_dir, tile_id)
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

        if (i + 1) % 500 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  Processed {i + 1}/{total} ({rate:.1f} entries/s)")

    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s: {stats['processed']} processed, "
          f"{stats['skipped']} skipped, {stats['failed']} failed")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess SpaceNet 6 SAR")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_tier2_sn6_sar(
        args.input_dir,
        args.output_dir,
        max_samples=args.max_samples,
        sample_fraction=args.sample_fraction,
        seed=args.seed,
    )
