"""Tier 1 — SpaceNet 6 Optical preprocessing.

SpaceNet 6 provides WorldView-2 0.5m pan-sharpened RGB imagery with
building footprint polygons. Used for VHR building grounding + fusion.

Spec treatment:
    - R1: RGB channels (already RGB from pan-sharpened)
    - R2: percentile normalization (deferred)
    - R4: dual-resolution branching
    - Polygons → bounding boxes (R6 Qwen format)
    - pair_type: single (optical only, SAR handled by tier2_sn6_sar)
    - dataset enum: "sn6_opt"
    - gsd_bucket: [GSD:0.5m] (native), CARTOSAT-proxy (proxy)
    - task: grounding

SpaceNet 6 structure:
    <input_dir>/
        train/
            images/
                <tile_id>.tif (WorldView-2 pan-sharpened RGB)
            labels/
                <tile_id>.geojson (building footprints)

Usage:
    python -m preprocess.tier1_sn6_opt --input-dir <path> --output-dir <path>
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
from preprocess.common.gsd import assign_gsd_bucket, create_proxy_sample
from preprocess.common.io import Manifest, append_jsonl, write_png
from preprocess.common.sample import quantile_bucket, quantile_edges, stratified_sample
from preprocess.validator import validate_sample

_DATASET = "sn6_opt"


def load_geojson_labels(geojson_path: Path) -> list[list[list[float]]]:
    """Load building footprint polygons from GeoJSON.

    Returns list of polygons, each being a list of [x, y] coordinates.
    """
    if not geojson_path.exists():
        return []

    try:
        with open(geojson_path) as f:
            data = json.load(f)
    except Exception:
        return []

    polygons = []
    for feature in data.get("features", []):
        geom = feature.get("geometry", {})
        if geom.get("type") == "Polygon":
            coords = geom.get("coordinates", [[]])
            if coords:
                polygons.append(coords[0])
        elif geom.get("type") == "MultiPolygon":
            for poly in geom.get("coordinates", []):
                if poly:
                    polygons.append(poly[0])

    return polygons


def polygons_to_bboxes(
    polygons: list[list[list[float]]],
    img_width: int,
    img_height: int,
) -> list[list[float]]:
    """Convert polygon coordinates to normalized bounding boxes."""
    bboxes = []
    for polygon in polygons:
        if len(polygon) < 3:
            continue

        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]

        # Convert pixel coords to normalized [0, 1000]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        bbox = pixel_to_normalized(
            (int(x_min), int(y_min), int(x_max), int(y_max)),
            img_width,
            img_height,
        )
        if bbox:
            bboxes.append(bbox[0])

    return bboxes


def process_tile(
    tile_path: Path,
    geojson_path: Path,
    output_dir: Path,
    tile_id: str,
) -> dict[str, Any] | None:
    """Process a single SpaceNet 6 tile."""
    try:
        import tifffile
        img = tifffile.imread(str(tile_path))
    except Exception as e:
        print(f"  [FAIL] {tile_id}: {e}", file=sys.stderr)
        return None

    # Ensure RGB
    if img.ndim == 3 and img.shape[2] >= 3:
        img = img[:, :, :3]
    elif img.ndim == 2:
        # Grayscale → RGB
        img = np.stack([img, img, img], axis=-1)

    # Normalize to uint8
    if img.dtype == np.uint16:
        p98 = np.percentile(img, 98)
        if p98 > 0:
            img = np.clip(img, 0, p98).astype(np.float32) / p98 * 255.0
        img = img.astype(np.uint8)

    h, w = img.shape[:2]

    # Save PNG
    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    out_path = img_dir / f"sn6_{tile_id}.png"
    write_png(img, out_path)

    # Load building footprints — keep every box, not just the first, so
    # the response text (which reports the real count) and the bbox
    # field agree with each other.
    polygons = load_geojson_labels(geojson_path)
    bboxes = polygons_to_bboxes(polygons, w, h)
    n_buildings = len(bboxes)

    sample_id = f"sn6_{tile_id}"

    return {
        "id": sample_id,
        "dataset": _DATASET,
        "task": "grounding" if bboxes else "vqa",
        "image_path": [str(out_path)],
        "pair_type": "single",
        "gsd_bucket": "",  # Filled by caller
        "split": "train",
        "instruction": "How many buildings are in this image?" if not bboxes
                       else "Locate the buildings in this image.",
        "response": "0" if not bboxes
                    else f"{n_buildings} building{'s' if n_buildings != 1 else ''} detected.",
        "bbox": bboxes if bboxes else None,
        "modality": "optical",
    }


def run_tier1_sn6_opt(
    input_dir: Path,
    output_dir: Path,
    *,
    max_samples: int | None = None,
    sample_fraction: float = 0.05,
    seed: int = 42,
) -> dict[str, int]:
    """Run SpaceNet 6 Optical preprocessing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.parquet"
    jsonl_path = output_dir / "sn6_opt.jsonl"
    manifest = Manifest(manifest_path)

    native_bucket = assign_gsd_bucket("sn6_opt")

    train_dir = input_dir / "train"
    images_dir = train_dir / "images"
    labels_dir = train_dir / "labels"

    if not images_dir.exists():
        print(f"  [WARN] Images directory not found: {images_dir}")
        return {"processed": 0, "skipped": 0, "failed": 0}

    # Find all tiles
    tile_paths = sorted(images_dir.glob("*.tif"))
    print(f"Found {len(tile_paths)} SpaceNet 6 tiles")

    # Selection (spec): "stratified by building-density quartile" — bucket
    # tiles by building-footprint count and sample proportionally from
    # every quartile, so sparse and dense tiles both stay represented.
    if sample_fraction < 1.0 and tile_paths:
        densities = [
            len(load_geojson_labels(labels_dir / f"{p.stem}.geojson"))
            for p in tile_paths
        ]
        edges = quantile_edges(densities, n_buckets=4)
        quartiles = [quantile_bucket(d, edges) for d in densities]
        keyed = list(zip(tile_paths, quartiles))
        tile_paths = [
            p for p, _ in stratified_sample(
                keyed, key_fn=lambda item: item[1],
                fraction=sample_fraction, seed=seed,
            )
        ]

    # Record the selected tile ids so tier2_sn6_sar can select the SAME
    # tiles ("paired 1:1 with optical selection" — Tier 2 table).
    selected_path = output_dir / "selected_tile_ids.json"
    selected_path.write_text(json.dumps(sorted(p.stem for p in tile_paths)))

    total = len(tile_paths)
    if max_samples:
        total = min(total, max_samples)

    stats = {"processed": 0, "skipped": 0, "failed": 0}
    start = time.time()

    for i, tile_path in enumerate(tile_paths[:total]):
        tile_id = tile_path.stem
        sample_id = f"sn6_{tile_id}"

        if manifest.is_processed(sample_id):
            stats["skipped"] += 1
            continue

        geojson_path = labels_dir / f"{tile_id}.geojson"
        sample = process_tile(tile_path, geojson_path, output_dir, tile_id)
        if sample is None:
            stats["failed"] += 1
            continue

        sample["gsd_bucket"] = native_bucket

        # R4: dual-resolution branching (native + CARTOSAT-proxy).
        rows = [sample, create_proxy_sample(sample)]

        ok_all = True
        for row in rows:
            ok, errs = validate_sample(row)
            if not ok:
                print(f"  [VALID] {row['id']}: {errs}", file=sys.stderr)
                ok_all = False
                break
        if not ok_all:
            stats["failed"] += 1
            continue

        for row in rows:
            append_jsonl(row, jsonl_path)

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
    parser = argparse.ArgumentParser(description="Preprocess SpaceNet 6 Optical")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_tier1_sn6_opt(
        args.input_dir,
        args.output_dir,
        max_samples=args.max_samples,
        sample_fraction=args.sample_fraction,
        seed=args.seed,
    )
