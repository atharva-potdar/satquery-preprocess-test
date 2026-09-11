"""Tier 1 — OSCD (Onera Satellite Change Detection) preprocessing.

OSCD provides 24 co-registered multispectral image pairs (13-band Sentinel-2)
with binary change masks.  Used as a bi-temporal regularizer.

Spec treatment:
    - R1: 13-band → RGB (B4/B3/B2)
    - R2: percentile normalization (deferred to stats.py global pass)
    - R4: NOT applicable (OSCD not in DUAL_RESOLUTION_DATASETS)
    - Masks → bounding boxes (R6 Qwen format)
    - pair_type: bitemporal
    - dataset enum: "oscd"
    - Split: all pairs usable (no prescribed benchmark)

Real OSCD directory layout (TorchGeo / official):
    <input_dir>/
        images/
            <location_name>/
                imgs_1/
                    <mission>_<date>_B01.tif  (individual band files)
                    <mission>_<date>_B02.tif
                    ...
                    <mission>_<date>_B12.tif
                    <mission>_<date>_B8A.tif
                imgs_2/
                    <mission>_<date>_B01.tif
                    ...
        labels/
            <location_name>/
                cm/
                    <location_name>-cm.tif  (binary change mask)

Usage:
    python -m preprocess.tier1_oscd --input-dir <path> --output-dir <path>
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from PIL import Image

from preprocess.common.bbox import pixel_to_normalized
from preprocess.common.gsd import assign_gsd_bucket
from preprocess.common.io import Manifest, append_jsonl, write_png
from preprocess.validator import validate_sample

# Band files we need for RGB extraction
_BAND_PRIORITY = ["B04", "B03", "B02"]  # R, G, B
# Full Sentinel-2 band order for stacking
_ALL_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]


def _extract_band_from_filename(filename: str) -> str | None:
    """Extract band name (e.g., 'B04') from Sentinel-2 filename."""
    # Match patterns like _B04.tif, _B8A.tif, _B12.tif
    m = re.search(r"_(B\d{2}[A-Z]?)\.tif$", filename)
    if m:
        return m.group(1)
    return None


def find_oscd_pairs(input_dir: Path) -> list[dict[str, Any]]:
    """Discover all OSCD image pairs and their masks.

    Returns list of dicts with keys:
        location, before_bands_dir, after_bands_dir, mask_path
    """
    images_dir = input_dir / "images"
    labels_dir = input_dir / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    pairs = []
    for location_dir in sorted(images_dir.iterdir()):
        if not location_dir.is_dir():
            continue

        imgs_1 = location_dir / "imgs_1"
        imgs_2 = location_dir / "imgs_2"

        if not imgs_1.exists() or not imgs_2.exists():
            continue

        # Find mask
        mask_path = labels_dir / location_dir.name / "cm" / f"{location_dir.name}-cm.tif"
        if not mask_path.exists():
            mask_path = None

        pairs.append({
            "location": location_dir.name,
            "before_bands_dir": imgs_1,
            "after_bands_dir": imgs_2,
            "mask_path": mask_path,
        })

    return pairs


def load_bands_as_rgb(bands_dir: Path) -> np.ndarray:
    """Load individual band TIFs from a directory and extract RGB.

    Reads B04, B03, B02 (Sentinel-2 RGB bands) and stacks them.

    Returns (H, W, 3) uint8 array.
    """
    bands = {}
    for tif_path in bands_dir.glob("*.tif"):
        band_name = _extract_band_from_filename(tif_path.name)
        if band_name in _BAND_PRIORITY or band_name in _ALL_BANDS:
            arr = tifffile.imread(str(tif_path))
            if arr.ndim == 3:
                arr = arr[0]  # Some TIFs have extra dimension
            bands[band_name] = arr

    if not bands:
        raise ValueError(f"No valid band files found in {bands_dir}")

    # Extract RGB
    rgb_bands = []
    for band_name in _BAND_PRIORITY:
        if band_name not in bands:
            raise ValueError(f"Missing required band {band_name} in {bands_dir}")
        rgb_bands.append(bands[band_name])

    rgb = np.stack(rgb_bands, axis=-1)

    # Normalize to uint8
    if rgb.dtype == np.uint16:
        # Sentinel-2 values typically 0–10000
        rgb = np.clip(rgb, 0, 10000).astype(np.float32) / 10000.0 * 255.0
        rgb = rgb.astype(np.uint8)
    elif rgb.max() > 255:
        p98 = np.percentile(rgb, 98)
        rgb = np.clip(rgb, 0, p98).astype(np.float32) / p98 * 255.0
        rgb = rgb.astype(np.uint8)

    return rgb


def load_mask(mask_path: Path | None, target_shape: tuple[int, int] | None = None) -> np.ndarray | None:
    """Load a binary change mask.

    Returns (H, W) uint8 array with values 0 (no change) and 1 (change),
    or None if no mask is available.
    """
    if mask_path is None or not mask_path.exists():
        return None

    mask = tifffile.imread(str(mask_path))

    # Ensure 2D
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    # Binarize (some masks use 255 for change)
    mask = (mask > 0).astype(np.uint8)

    # Resize if needed
    if target_shape is not None and mask.shape != target_shape:
        from PIL import Image as PILImage
        mask_img = PILImage.fromarray(mask, mode="L")
        mask_img = mask_img.resize(
            (target_shape[1], target_shape[0]),
            resample=PILImage.Resampling.NEAREST,
        )
        mask = np.array(mask_img)

    return mask


def mask_to_bboxes(mask: np.ndarray) -> list[list[int]]:
    """Convert binary change mask to a list of bounding boxes.

    Uses connected-component labeling to find change regions.
    Returns list of [x1, y1, x2, y2] pixel coordinates.
    """
    if mask is None or mask.sum() == 0:
        return []

    try:
        from scipy import ndimage
        labeled, num_features = ndimage.label(mask)
    except ImportError:
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            return []
        return [[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]]

    bboxes = []
    for i in range(1, num_features + 1):
        ys, xs = np.where(labeled == i)
        if len(ys) < 10:  # Skip tiny noise regions
            continue
        bboxes.append([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())])

    return bboxes


def make_sample_id(location: str) -> str:
    """Generate a unique, lowercase, underscore-separated sample id."""
    return f"oscd_{location}".lower().replace("-", "_")[:64]


def process_pair(
    pair: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any] | None:
    """Process a single OSCD pair into JSONL sample + PNGs.

    Returns the sample dict, or None if processing failed.
    """
    location = pair["location"]

    # Load images
    try:
        before_rgb = load_bands_as_rgb(pair["before_bands_dir"])
        after_rgb = load_bands_as_rgb(pair["after_bands_dir"])
    except Exception as e:
        print(f"  ERROR loading images for {location}: {e}")
        return None

    # Load and convert mask to bboxes
    mask = load_mask(pair.get("mask_path"), target_shape=before_rgb.shape[:2])
    bboxes = mask_to_bboxes(mask)
    bbox_normalized = None
    if bboxes:
        h, w = before_rgb.shape[:2]
        bbox_normalized = pixel_to_normalized(tuple(bboxes[0]), w, h)

    # Write PNGs
    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    before_png = img_dir / f"{location}_before.png"
    after_png = img_dir / f"{location}_after.png"

    write_png(before_rgb, before_png)
    write_png(after_rgb, after_png)

    # Build JSONL sample
    sample_id = make_sample_id(location)
    sample = {
        "id": sample_id,
        "dataset": "oscd",
        "task": "change_vqa" if bbox_normalized is None else "change_grounding",
        "image_path": [str(before_png), str(after_png)],
        "pair_type": "bitemporal",
        "gsd_bucket": assign_gsd_bucket("oscd"),
        "split": "train",
        "instruction": "What changed between the two dates?",
        "response": f"Change detected in {location}.",
        "bbox": bbox_normalized,
        "modality": "optical",
    }

    # Validate
    ok, errs = validate_sample(sample)
    if not ok:
        print(f"  VALIDATION FAILED for {location}: {errs}")
        return None

    return sample


def run_tier1_oscd(
    input_dir: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run OSCD preprocessing with manifest-based resumability.

    Returns summary dict with counts.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if manifest_path is None:
        manifest_path = output_dir / "manifest.parquet"
    manifest = Manifest(manifest_path)

    jsonl_path = output_dir / "oscd.jsonl"

    # Discover pairs
    pairs = find_oscd_pairs(input_dir)
    print(f"Found {len(pairs)} OSCD pairs")

    stats = {"total": len(pairs), "processed": 0, "skipped": 0, "failed": 0}

    for pair in pairs:
        sample_id = make_sample_id(pair["location"])

        # Manifest check — skip if already processed
        if manifest.is_processed(sample_id):
            stats["skipped"] += 1
            continue

        # Process
        sample = process_pair(pair, output_dir)
        if sample is None:
            stats["failed"] += 1
            continue

        # Write JSONL
        append_jsonl(sample, jsonl_path)

        # Mark processed
        manifest.mark_processed(
            sample_id=sample_id,
            dataset="oscd",
            split="train",
            output_shard="local",
        )
        stats["processed"] += 1

        if stats["processed"] % 5 == 0:
            print(f"  Processed {stats['processed']}/{stats['total'] - stats['skipped']}")

    print(f"Done: {stats['processed']} processed, {stats['skipped']} skipped, {stats['failed']} failed")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Preprocess OSCD dataset")
    parser.add_argument("--input-dir", required=True, help="OSCD dataset root")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--manifest", default=None, help="Manifest path")
    args = parser.parse_args()

    run_tier1_oscd(args.input_dir, args.output_dir, args.manifest)


if __name__ == "__main__":
    main()
