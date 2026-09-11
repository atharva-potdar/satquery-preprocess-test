"""Tier 0 — BigEarthNet (reBEN v2.0 / BigEarthNet-MM) preprocessing.

BigEarthNet provides co-registered Sentinel-2 multispectral and Sentinel-1
SAR image patches with 19-class CORINE land cover labels. Used as the
primary RS domain adaptation backbone (~35% of corpus) AND as the primary
source for Mandate 4 (cross-modal optical+SAR) samples, since S1/S2 patches
are already co-registered — no reprojection step needed.

Kaggle input layout (this script's contract — a Kaggle Dataset built by the
acquisition step, not BigEarthNet's own on-disk format):
    <input_dir>/
        metadata.parquet         # columns: patch_id, labels (list[int] class
                                  # indices 0-18 or a 19-length multihot list),
                                  # country (optional), cloud_pct (optional)
        s2_npy/<patch_id>.npy    # (12, H, W) uint16 — REQUIRED
        s1_npy/<patch_id>.npy    # (2, H, W) float32, [VV, VH] linear power —
                                  # OPTIONAL; when present, a SAR VQA sample and
                                  # a cross-modal fusion sample are also emitted

Why this layout and not raw BigEarthNet-MM HDF5: the spec's acquisition table
(Section 9) says "HF lc-col/bigearthnet (HDF5) -> streaming -> filter -> PNG",
but streaming schema field names for that specific HF dataset aren't
verifiable without network access from here. metadata.parquet + per-patch
.npy is the shape a Kaggle-side acquisition notebook can produce from that
stream in one pass, and it's what this script (and its tests) are built
against.
# ponytail: if the real HF dataset ships different field names, adjust
# load_bigen_arrays()'s column lookups — everything downstream is unaffected.

Spec treatment:
    - R1: Sentinel-2 B4/B3/B2 -> RGB (NIR dropped)
    - R2: percentile normalization (per-patch here; global pass is stats.py,
      run separately over the written PNGs before training)
    - R3: SAR pseudo-RGB (VV, VH, VV-VH) via common.sar.sar_pseudo_rgb
    - R5: native 10m (R5 anchor)
    - Products per pair (when S1 available): optical VQA, SAR VQA, 1x2
      concat cross-modal fusion sample
    - dataset enum: "bigen"
    - gsd_bucket: [GSD:10m]

Usage:
    python -m preprocess.tier0_bigen --input-dir <path> --output-dir <path> --max-samples 60000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from preprocess.common.concat import concat_horizontal
from preprocess.common.gsd import assign_gsd_bucket
from preprocess.common.io import Manifest, append_jsonl, write_png
from preprocess.common.sar import sar_pseudo_rgb
from preprocess.validator import validate_sample

# Sentinel-2 band indices (0-based) for RGB extraction.
# BigEarthNet-MM band order: B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B10,B11(,B12)
# We need B04(idx=3), B03(idx=2), B02(idx=1).
_S2_RGB_INDICES = [3, 2, 1]  # B04, B03, B02

# 19-class CORINE labels
_CORINE_CLASSES = [
    "Continuous urban fabric",
    "Discontinuous urban fabric",
    "Industrial or commercial units",
    "Road and rail networks and associated land",
    "Port areas",
    "Airports",
    "Mineral extraction sites",
    "Dump sites",
    "Construction sites",
    "Green urban areas",
    "Sports and leisure facilities",
    "Non-irrigated arable land",
    "Permanently irrigated land",
    "Rice fields",
    "Vineyards",
    "Fruit trees and berry plantations",
    "Pastures",
    "Annual crops associated with permanent crops",
    "Complex cultivation patterns",
]


def s2_to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    """Extract B04/B03/B02 from a (12+, H, W) Sentinel-2 array as uint8 RGB.

    Per-patch p98 stretch. R2's global percentile pass is a separate step
    (stats.py) applied to the written PNGs, not duplicated here.
    """
    rgb = image[_S2_RGB_INDICES]  # (3, H, W)
    rgb = np.transpose(rgb, (1, 2, 0)).astype(np.float32)  # (H, W, 3)

    p98 = np.percentile(rgb, 98)
    if p98 > 0:
        rgb = np.clip(rgb, 0, p98) / p98 * 255.0
    return rgb.astype(np.uint8)


def _labels_to_multihot(labels: Any, n_classes: int = 19) -> np.ndarray:
    """Normalize a metadata 'labels' cell to a 19-length multihot vector.

    Accepts either a list of active class indices (e.g. [0, 11]) or an
    already-multihot list/array of length n_classes.
    """
    vec = np.zeros(n_classes, dtype=np.int32)
    if labels is None:
        return vec
    if isinstance(labels, str) or not hasattr(labels, "__iter__"):
        labels = [labels]
    labels = list(labels)

    if len(labels) == n_classes and all(v in (0, 1) for v in labels if v is not None):
        return np.array([int(v) for v in labels], dtype=np.int32)

    for v in labels:
        try:
            idx = int(v)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < n_classes:
            vec[idx] = 1
    return vec


def parse_bigen_sample(
    sample: dict[str, Any],
    idx: int,
    *,
    max_classes: int = 19,
) -> dict[str, Any] | None:
    """Parse a single BigEarthNet-MM S2 patch + label into our JSONL format.

    `sample` must have "image" ((12+, H, W) array) and "label" (multihot
    or index-list). image_path/gsd_bucket are left for the caller to fill
    in once the PNG has actually been written.

    Returns sample dict or None if invalid.
    """
    image = sample.get("image")
    if image is None:
        return None

    image = np.asarray(image)
    if image.ndim != 3 or image.shape[0] < 4:
        return None

    label = sample.get("label")
    if label is None:
        return None

    label = np.asarray(label)
    active_classes = [
        _CORINE_CLASSES[i]
        for i in range(min(len(label), max_classes))
        if label[i] > 0
    ]
    if not active_classes:
        return None

    response = ", ".join(active_classes)
    sample_id = f"bigen_{idx:08d}"

    return {
        "id": sample_id,
        "dataset": "bigen",
        "task": "vqa",
        "image_path": [],  # Filled by caller after PNG write
        "pair_type": "single",
        "gsd_bucket": "",  # Filled by caller
        "split": "train",
        "instruction": "What land cover types are present in this image?",
        "response": response,
        "bbox": None,
        "modality": "optical",
    }


def load_bigen_arrays(
    input_dir: Path,
    row: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray | None]:
    """Load one BigEarthNet patch's S2 array (required) and S1 array (optional).

    Returns (s2_array, s1_array_or_None).
    Raises FileNotFoundError if the required S2 patch is missing.
    """
    patch_id = str(row.get("patch_id") or row.get("id") or "").strip()
    if not patch_id:
        raise ValueError("metadata row has no patch_id/id")

    s2_path = input_dir / "s2_npy" / f"{patch_id}.npy"
    if not s2_path.exists():
        raise FileNotFoundError(f"Missing S2 patch: {s2_path}")
    s2 = np.load(s2_path)

    s1 = None
    s1_path = input_dir / "s1_npy" / f"{patch_id}.npy"
    if s1_path.exists():
        s1 = np.load(s1_path)

    return s2, s1


def _build_sar_and_fusion_samples(
    sample_id: str,
    opt_png: Path,
    opt_rgb: np.ndarray,
    opt_response: str,
    s1: np.ndarray,
    img_dir: Path,
    native_bucket: str,
) -> list[dict[str, Any]]:
    """Build the SAR-VQA and cross-modal fusion samples for a patch with S1 data.

    Satisfies Mandate 4 (cross-modal optical+SAR): BigEarthNet's S1/S2
    patches are already co-registered, so this is the cheapest correct
    source for `pair_type: cross-modal` training rows — nothing else in
    the pipeline currently emits any.
    """
    vv, vh = s1[0].astype(np.float32), s1[1].astype(np.float32)
    sar_rgb = sar_pseudo_rgb(vv, vh)

    sar_png = img_dir / f"{sample_id}_sar.png"
    write_png(sar_rgb, sar_png)

    sar_sample = {
        "id": f"{sample_id}_sar",
        "dataset": "bigen",
        "task": "vqa",
        "image_path": [str(sar_png)],
        "pair_type": "single",
        "gsd_bucket": native_bucket,
        "split": "train",
        "instruction": "Describe the SAR backscatter characteristics of this scene.",
        "response": f"SAR backscatter over a scene classified optically as: {opt_response}.",
        "bbox": None,
        "modality": "sar",
    }

    concat_img = concat_horizontal(opt_rgb, sar_rgb)
    concat_png = img_dir / f"{sample_id}_concat.png"
    concat_img.save(str(concat_png), format="PNG", compress_level=0)

    fusion_sample = {
        "id": f"{sample_id}_fusion",
        "dataset": "bigen",
        "task": "fusion_vqa",
        "image_path": [str(opt_png), str(sar_png)],
        "pair_type": "cross-modal",
        "gsd_bucket": native_bucket,
        "split": "train",
        "instruction": (
            "Using both the optical and SAR images, describe this scene's "
            "land cover and surface conditions."
        ),
        "response": (
            f"Optical imagery shows: {opt_response}. "
            "SAR backscatter is consistent with these cover types."
        ),
        "bbox": None,
        "modality": "optical+sar",
    }

    return [sar_sample, fusion_sample]


def run_tier0_bigen(
    input_dir: Path,
    output_dir: Path,
    *,
    max_samples: int = 60000,
    seed: int = 42,
) -> dict[str, int]:
    """Run BigEarthNet preprocessing with manifest-based resumability.

    Returns stats dict: {processed, skipped, failed}. "processed" counts
    patches (each patch may emit 1-3 JSONL rows: optical, +SAR, +fusion).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.parquet"
    jsonl_path = output_dir / "bigen.jsonl"
    manifest = Manifest(manifest_path)

    native_bucket = assign_gsd_bucket("bigen")

    metadata_path = input_dir / "metadata.parquet"
    if not metadata_path.exists():
        print(f"  [WARN] metadata.parquet not found in {input_dir}")
        return {"processed": 0, "skipped": 0, "failed": 0}

    import pandas as pd
    metadata = pd.read_parquet(metadata_path)
    total = min(len(metadata), max_samples)
    print(f"Loaded metadata: {len(metadata)} patches, processing {total}")

    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    stats = {"processed": 0, "skipped": 0, "failed": 0}
    start = time.time()

    for idx in range(total):
        row = metadata.iloc[idx].to_dict()
        sample_id = f"bigen_{idx:08d}"

        if manifest.is_processed(sample_id):
            stats["skipped"] += 1
            continue

        try:
            s2, s1 = load_bigen_arrays(input_dir, row)
        except (FileNotFoundError, ValueError) as e:
            print(f"  [FAIL] {sample_id}: {e}", file=sys.stderr)
            stats["failed"] += 1
            continue

        label_vec = _labels_to_multihot(row.get("labels"))
        opt_sample = parse_bigen_sample({"image": s2, "label": label_vec}, idx)
        if opt_sample is None:
            stats["failed"] += 1
            continue

        opt_rgb = s2_to_rgb_uint8(s2)
        opt_png = img_dir / f"{sample_id}_optical.png"
        write_png(opt_rgb, opt_png)
        opt_sample["image_path"] = [str(opt_png)]
        opt_sample["gsd_bucket"] = native_bucket

        emitted = [opt_sample]

        if s1 is not None and s1.ndim == 3 and s1.shape[0] >= 2:
            emitted.extend(_build_sar_and_fusion_samples(
                sample_id, opt_png, opt_rgb, opt_sample["response"],
                s1, img_dir, native_bucket,
            ))

        errors = []
        for s in emitted:
            ok, errs = validate_sample(s)
            if not ok:
                errors.append((s["id"], errs))
        if errors:
            for sid, errs in errors:
                print(f"  [VALID] {sid}: {errs}", file=sys.stderr)
            stats["failed"] += 1
            continue

        for s in emitted:
            append_jsonl(s, jsonl_path)

        manifest.mark_processed(
            sample_id=sample_id,
            dataset="bigen",
            split="train",
            output_shard="local",
        )
        stats["processed"] += 1

        if (idx + 1) % 5000 == 0:
            elapsed = time.time() - start
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            print(f"  Processed {idx + 1}/{total} ({rate:.1f} patches/s)")

    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s: {stats['processed']} processed, "
          f"{stats['skipped']} skipped, {stats['failed']} failed")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess BigEarthNet")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_tier0_bigen(
        args.input_dir,
        args.output_dir,
        max_samples=args.max_samples,
        seed=args.seed,
    )
