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
from PIL import Image

from preprocess.common.concat import concat_horizontal
from preprocess.common.gsd import assign_gsd_bucket
from preprocess.common.io import Manifest, append_jsonl, write_png
from preprocess.common.sar import sar_intensity_pseudo_gray, sar_pseudo_rgb
from preprocess.validator import validate_sample

_DATASET = "sn6_sar"


def load_optical_selection(sn6_opt_output_dir: Path) -> tuple[set[str], dict[str, str]]:
    """Read tier1_sn6_opt.py's output to pair SAR tile selection 1:1 with
    the optical tiles it already chose (spec: "Paired 1:1 with optical
    selection"), and to locate each tile's optical PNG for fusion samples.

    Returns (selected_tile_ids, tile_id -> optical_png_path).
    """
    selected_path = sn6_opt_output_dir / "selected_tile_ids.json"
    tile_ids: set[str] = set()
    if selected_path.exists():
        tile_ids = set(json.loads(selected_path.read_text()))

    optical_png_by_tile: dict[str, str] = {}
    jsonl_path = sn6_opt_output_dir / "sn6_opt.jsonl"
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                # ids look like "sn6_<tile_id>" (optionally "_proxy")
                if row.get("id", "").endswith("_proxy"):
                    continue
                tile_id = row["id"][len("sn6_"):]
                if row.get("image_path"):
                    optical_png_by_tile[tile_id] = row["image_path"][0]

    return tile_ids, optical_png_by_tile


def process_sar_tile(
    tile_path: Path,
    output_dir: Path,
    tile_id: str,
    *,
    optical_png: Path | None = None,
) -> list[dict[str, Any]]:
    """Process a single SpaceNet 6 SAR tile.

    Returns [sar_sample] normally, or [sar_sample, fusion_sample] when
    `optical_png` (the already-processed optical PNG for this same tile_id,
    from tier1_sn6_opt.py) is given — satisfying Mandate 4 for SpaceNet 6.
    Returns [] on failure.
    """
    try:
        import tifffile
        img = tifffile.imread(str(tile_path))
    except Exception as e:
        print(f"  [FAIL] {tile_id}: {e}", file=sys.stderr)
        return []

    # True quad-pol (HH, VV, VH) uses R3's real physics-corrected mapping
    # (linear->dB, per-pol clip, R=HH G=VV B=VH). A single band has no
    # real second/third polarization to build that from, so it gets the
    # honest single-channel dB render instead of a fabricated one.
    if img.ndim == 3 and img.shape[0] >= 3:
        hh = img[0].astype(np.float32)
        vv = img[1].astype(np.float32)
        vh = img[2].astype(np.float32)
        rgb = sar_pseudo_rgb(vv, vh, hh=hh)
    elif img.ndim == 3:
        rgb = sar_intensity_pseudo_gray(img[0].astype(np.float32))
    elif img.ndim == 2:
        rgb = sar_intensity_pseudo_gray(img.astype(np.float32))
    else:
        return []

    # Save PNG
    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    out_path = img_dir / f"sn6_sar_{tile_id}.png"
    write_png(rgb, out_path)

    sample_id = f"sn6_sar_{tile_id}"

    sar_sample = {
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

    samples = [sar_sample]

    if optical_png is not None and optical_png.exists():
        opt_img = np.array(Image.open(str(optical_png)).convert("RGB"))
        concat_img = concat_horizontal(opt_img, rgb)
        concat_path = img_dir / f"sn6_fusion_{tile_id}.png"
        concat_img.save(str(concat_path), format="PNG", compress_level=0)

        samples.append({
            "id": f"sn6_fusion_{tile_id}",
            "dataset": _DATASET,
            "task": "fusion_vqa",
            "image_path": [str(optical_png), str(out_path)],
            "pair_type": "cross-modal",
            "gsd_bucket": "",  # Filled by caller
            "split": "train",
            "instruction": (
                "Using both the optical and SAR images, describe this "
                "scene's surface conditions."
            ),
            "response": (
                f"Optical and SAR imagery of {tile_id} show consistent "
                "surface conditions across both sensors."
            ),
            "bbox": None,
            "modality": "optical+sar",
        })

    return samples


def run_tier2_sn6_sar(
    input_dir: Path,
    output_dir: Path,
    *,
    max_samples: int | None = None,
    sample_fraction: float = 0.05,
    seed: int = 42,
    optical_dir: Path | None = None,
) -> dict[str, int]:
    """Run SpaceNet 6 SAR preprocessing.

    `optical_dir` should be tier1_sn6_opt.py's --output-dir. When given,
    SAR tile selection is intersected with the tiles it already chose
    ("paired 1:1 with optical selection" — Tier 2 table), and a
    cross-modal fusion_vqa sample is emitted per paired tile (Mandate 4).
    Without it, SAR tiles are sampled independently and no fusion rows
    are produced — pass optical_dir whenever tier1_sn6_opt has already run.
    """
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

    optical_png_by_tile: dict[str, str] = {}
    if optical_dir is not None:
        selected_ids, optical_png_by_tile = load_optical_selection(optical_dir)
        if selected_ids:
            tile_paths = [p for p in tile_paths if p.stem in selected_ids]
            print(f"  Paired 1:1 with optical selection: {len(tile_paths)} tiles")
        else:
            print(f"  [WARN] No selected_tile_ids.json found under {optical_dir} — "
                  f"falling back to independent sampling")

    if optical_dir is None or not optical_png_by_tile:
        # ponytail: no optical pairing available — independent random
        # sample. Spec's stratification requirement here ("paired 1:1")
        # only applies when tier1_sn6_opt.py has already run; there's no
        # other natural stratification key for SAR tiles alone.
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

        optical_png = optical_png_by_tile.get(tile_id)
        samples = process_sar_tile(
            tile_path, output_dir, tile_id,
            optical_png=Path(optical_png) if optical_png else None,
        )
        if not samples:
            stats["failed"] += 1
            continue

        for s in samples:
            s["gsd_bucket"] = sar_bucket

        errors = [(s["id"], e) for s in samples for ok, e in [validate_sample(s)] if not ok]
        if errors:
            for sid, errs in errors:
                print(f"  [VALID] {sid}: {errs}", file=sys.stderr)
            stats["failed"] += 1
            continue

        for s in samples:
            append_jsonl(s, jsonl_path)

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
    parser.add_argument("--optical-dir", type=Path, default=None,
                         help="tier1_sn6_opt.py's output dir, for paired 1:1 "
                              "selection and cross-modal fusion samples")
    args = parser.parse_args()

    run_tier2_sn6_sar(
        args.input_dir,
        args.output_dir,
        max_samples=args.max_samples,
        sample_fraction=args.sample_fraction,
        seed=args.seed,
        optical_dir=args.optical_dir,
    )
