"""Tier 1 — RSVQA-HR preprocessing.

RSVQA-HR provides high-resolution aerial imagery (USGS, 0.15m) with
question-answer pairs for visual question answering.

Spec treatment:
    - R1: RGB channels (already RGB in source)
    - R2: percentile normalization (deferred to stats.py)
    - R4: dual-resolution branching (native + proxy)
    - gsd_bucket: [GSD:0.15m] (native), CARTOSAT-proxy (proxy)
    - dataset enum: "rsvqa_hr"
    - pair_type: single
    - task: vqa

RSVQA-HR structure:
    <input_dir>/
        images/
            <image_id>.png (or .jpg)
        questions.json / train.json / val.json / test.json
            Each entry: {"question_id": ..., "question": ..., "answer": ...,
                         "image_id": ..., "type": ...}

Usage:
    python -m preprocess.tier1_rsvqa_hr --input-dir <path> --output-dir <path>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image

from preprocess.common.gsd import assign_gsd_bucket, create_proxy_sample
from preprocess.common.io import Manifest, append_jsonl, write_png
from preprocess.validator import validate_sample

_NATIVE_GSD = 0.15  # USGS 15cm
_DATASET = "rsvqa_hr"


def _ensure_png(src_path: Path, png_dir: Path) -> str:
    """R7: all outputs are 8-bit 3-channel PNG. RSVQA-HR images can ship
    as .jpg/.tif — convert those; a source .png is reused as-is."""
    if src_path.suffix.lower() == ".png":
        return str(src_path)

    png_path = png_dir / f"{src_path.stem}.png"
    if not png_path.exists():
        img = Image.open(str(src_path))
        if img.mode != "RGB":
            img = img.convert("RGB")
        png_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(png_path), format="PNG", compress_level=0)
    return str(png_path)


def load_annotations(input_dir: Path) -> list[dict]:
    """Load RSVQA-HR annotations from JSON files.

    Tries train.json first, falls back to questions.json.
    """
    for name in ["train.json", "questions.json"]:
        path = input_dir / name
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                # Some formats wrap in {"questions": [...]}
                return data.get("questions", data.get("data", []))
    return []


def parse_rsvqa_sample(
    entry: dict[str, Any],
    images_dir: Path | None,
    sample_id: str,
    *,
    png_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Parse a single RSVQA-HR annotation into our JSONL format."""
    question = entry.get("question", "")
    answer = entry.get("answer", "")
    image_id = entry.get("image_id", "")

    if not question or not answer:
        return None

    # Handle image path — R7 requires PNG; convert non-PNG sources.
    image_path_str = ""
    if images_dir and image_id:
        for ext in [".png", ".jpg", ".jpeg", ".tif"]:
            img_path = images_dir / f"{image_id}{ext}"
            if img_path.exists():
                image_path_str = _ensure_png(img_path, png_dir or images_dir)
                break

    return {
        "id": sample_id,
        "dataset": _DATASET,
        "task": "vqa",
        "image_path": [image_path_str] if image_path_str else [],
        "pair_type": "single",
        "gsd_bucket": "",  # Filled by caller
        "split": "train",
        "instruction": question,
        "response": str(answer),
        "bbox": None,
        "modality": "optical",
    }


def run_tier1_rsvqa_hr(
    input_dir: Path,
    output_dir: Path,
    *,
    max_samples: int | None = None,
) -> dict[str, int]:
    """Run RSVQA-HR preprocessing with manifest-based resumability."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.parquet"
    jsonl_path = output_dir / "rsvqa_hr.jsonl"
    manifest = Manifest(manifest_path)

    native_bucket = assign_gsd_bucket("rsvqa_hr")

    annotations = load_annotations(input_dir)
    if not annotations:
        print(f"  [WARN] No annotations found in {input_dir}")
        return {"processed": 0, "skipped": 0, "failed": 0}

    images_dir = None
    for name in ["images", "Images", "img"]:
        candidate = input_dir / name
        if candidate.exists():
            images_dir = candidate
            break

    png_dir = output_dir / "images" / "converted"

    total = len(annotations)
    if max_samples:
        total = min(total, max_samples)
    print(f"Loaded {len(annotations)} annotations, processing {total}")

    stats = {"processed": 0, "skipped": 0, "failed": 0}
    start = time.time()

    for i, entry in enumerate(annotations[:total]):
        qid = entry.get("question_id", entry.get("id", i))
        sample_id = f"rsvqa_{qid}"

        if manifest.is_processed(sample_id):
            stats["skipped"] += 1
            continue

        sample = parse_rsvqa_sample(entry, images_dir, sample_id, png_dir=png_dir)
        if sample is None:
            stats["failed"] += 1
            continue

        sample["gsd_bucket"] = native_bucket

        # R4: dual-resolution branching — emit native + CARTOSAT-proxy rows.
        rows = [sample]
        if sample["image_path"]:
            rows.append(create_proxy_sample(sample))

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

        if (i + 1) % 10000 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  Processed {i + 1}/{total} ({rate:.1f} entries/s)")

    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s: {stats['processed']} processed, "
          f"{stats['skipped']} skipped, {stats['failed']} failed")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess RSVQA-HR")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    run_tier1_rsvqa_hr(args.input_dir, args.output_dir, max_samples=args.max_samples)
