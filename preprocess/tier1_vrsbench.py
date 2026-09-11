"""VRSBench preprocessing — caption + grounding + VQA with R4 dual-resolution.

Parses VRSBench_train.json (LLaVA-format consolidated file, 142k samples).

Task prefixes in conversation:
    [caption] — image captioning (20,264 samples)
    [refer]   — visual grounding / referring (36,313 samples)
    [vqa]     — visual question answering (85,813 samples)

Referring answer format: {<x_left><y_top><x_right><y_bottom>}
    Coordinates are in GeoChat [0,100] grid (resized to 100×100).
    Converted to Qwen [0,1000] by multiplying by 10.

Each source sample produces TWO JSONL rows (R4 dual-resolution):
    Native branch  — gsd_bucket="VHR-native"
    Proxy branch   — gsd_bucket="CARTOSAT-proxy[GSD:2.0m]"

PNGs are saved when images are available; JSONL is written regardless.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from preprocess.common.bbox import unit01_to_qwen, validate_bbox_ordering
from preprocess.common.gsd import assign_gsd_bucket, create_proxy_sample
from preprocess.common.io import Manifest, append_jsonl, write_png

_DATASET = "vrsbench"

# Regex to extract task prefix: [caption], [refer], [vqa]
_TASK_RE = re.compile(r"\[(caption|refer|vqa)\]")

# Regex to parse referring answer: {<45><45><59><59>}
_REFERRING_RE = re.compile(r"\{<(\d+)><(\d+)><(\d+)><(\d+)>\}")


def parse_conversation(
    entry: dict[str, Any],
) -> tuple[str, str, str | None]:
    """Parse a VRSBench_train.json entry.

    Returns (task_type, instruction, response_or_coords).
    For referring, response_or_coords is the raw "{<x><y><x><y>}" string.
    """
    convs = entry["conversations"]
    if len(convs) < 2:
        return "unknown", "", None

    human_msg = convs[0]["value"]
    gpt_msg = convs[1]["value"]

    # Extract task prefix from human message
    m = _TASK_RE.search(human_msg)
    if not m:
        return "unknown", human_msg, gpt_msg

    task = m.group(1)

    # Extract instruction (everything after the prefix tag)
    instruction = human_msg[m.end():].strip()
    # Remove leading newline if present
    instruction = instruction.lstrip("\n").strip()

    return task, instruction, gpt_msg


def referring_to_qwen(answer: str) -> list[list[float]] | None:
    """Parse GeoChat-format referring answer to Qwen [0,1000] bbox.

    Input: "{<45><45><59><59>}"
    Output: [[450.0, 450.0, 590.0, 590.0]] or None if parse fails.
    """
    m = _REFERRING_RE.match(answer.strip())
    if not m:
        return None

    x1, y1, x2, y2 = (int(g) for g in m.groups())

    # Convert from GeoChat [0,100] to Qwen [0,1000]
    bbox = [[x1 * 10.0, y1 * 10.0, x2 * 10.0, y2 * 10.0]]

    if not validate_bbox_ordering(bbox):
        return None

    # Reject zero-area boxes
    bx1, by1, bx2, by2 = bbox[0]
    if bx1 >= bx2 or by1 >= by2:
        return None

    return bbox


def process_entry(
    entry: dict[str, Any],
    images_dir: Path | None,
    sample_id: str = "",
) -> list[dict[str, Any]]:
    """Process one VRSBench_train.json entry → list of sample dicts.

    Returns 0-1 samples (caller handles dual-resolution duplication).
    image_path points to the extracted source image (no copy).
    """
    task, instruction, response = parse_conversation(entry)
    image_name = entry.get("image", "")

    if not sample_id:
        sample_id = f"vrsbench_{image_name.replace('/', '_').replace('.', '_')}"

    if task == "unknown" or not instruction:
        return []

    # Reference extracted image directly — no copy
    image_path_str = ""
    if images_dir and image_name:
        img_path = images_dir / image_name
        if img_path.exists():
            image_path_str = str(img_path)

    if task == "caption":
        if not response:
            return []
        return [{
            "id": f"vrsbench_{sample_id}_caption",
            "dataset": _DATASET,
            "split": "train",
            "image_path": [image_path_str] if image_path_str else [],
            "bbox": None,
            "task": "caption",
            "pair_type": "single",
            "modality": "optical",
            "gsd_bucket": "",  # filled by caller
            "instruction": instruction,
            "response": response,
        }]

    elif task == "refer":
        if not response:
            return []
        bbox = referring_to_qwen(response)
        if bbox is None:
            return []
        return [{
            "id": f"vrsbench_{sample_id}_grounding",
            "dataset": _DATASET,
            "split": "train",
            "image_path": [image_path_str] if image_path_str else [],
            "bbox": bbox,
            "task": "grounding",
            "pair_type": "single",
            "modality": "optical",
            "gsd_bucket": "",  # filled by caller
            "instruction": instruction,
            "response": response,
        }]

    elif task == "vqa":
        if not response:
            return []
        return [{
            "id": f"vrsbench_{sample_id}_vqa",
            "dataset": _DATASET,
            "split": "train",
            "image_path": [image_path_str] if image_path_str else [],
            "bbox": None,
            "task": "vqa",
            "pair_type": "single",
            "modality": "optical",
            "gsd_bucket": "",  # filled by caller
            "instruction": instruction,
            "response": response,
        }]

    return []


def generate_proxy_image(
    src_path: Path,
    proxy_dir: Path,
    image_name: str,
    scale_factor: int = 4,
) -> str | None:
    """Generate a CARTOSAT-proxy (~2m) downsampled image via bicubic resize.

    Issue 7 fix: uses dynamic scale factor based on actual image dimensions
    rather than a hardcoded (128, 128) target, so any source size gets the
    intended 4× GSD reduction.

    Parameters
    ----------
    src_path : source image path.
    proxy_dir : directory to write the proxy PNG into.
    image_name : filename for the proxy (preserved from source).
    scale_factor : integer divisor applied to both width and height (default 4).
                   Resulting size is clamped to a minimum of 32px per side.

    Returns the proxy image path, or None on failure.
    """
    proxy_path = proxy_dir / image_name
    if proxy_path.exists():
        return str(proxy_path)

    try:
        img = Image.open(str(src_path))
        if img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size
        # Dynamic 4× reduction — fixes Issue 7 (hardcoded 512→128 assumption)
        proxy_w = max(32, w // scale_factor)
        proxy_h = max(32, h // scale_factor)
        proxy = img.resize((proxy_w, proxy_h), Image.BICUBIC)
        proxy_path.parent.mkdir(parents=True, exist_ok=True)
        # Issue 9 fix: compress_level=6 (was 0)
        proxy.save(str(proxy_path), format="PNG", compress_level=6)
        return str(proxy_path)
    except Exception as e:
        print(f"  [WARN] Failed to generate proxy for {image_name}: {e}", file=sys.stderr)
        return None


def run_tier1_vrsbench(
    input_dir: Path,
    output_dir: Path,
    *,
    max_samples: int | None = None,
) -> dict[str, int]:
    """Run VRSBench preprocessing with manifest/resumability.

    Parses VRSBench_train.json directly. Each sample → native + proxy rows.
    Native images referenced from extracted source; proxy generated once per unique image.

    Returns stats dict: {processed, skipped, failed}.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.parquet"
    jsonl_path = output_dir / "vrsbench.jsonl"
    manifest = Manifest(manifest_path)

    # Load consolidated JSON
    json_path = input_dir / "VRSBench_train.json"
    if not json_path.exists():
        raise FileNotFoundError(f"VRSBench_train.json not found in {input_dir}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Check for images directory
    images_dir = None
    for name in ["Images_train", "images", "Images"]:
        candidate = input_dir / name
        if candidate.exists():
            images_dir = candidate
            break

    proxy_dir = output_dir / "images" / "proxy"
    proxy_dir.mkdir(parents=True, exist_ok=True)

    total = len(data)
    if max_samples:
        total = min(total, max_samples)
    print(f"Loaded {len(data)} entries, processing {total}")

    stats = {"processed": 0, "skipped": 0, "failed": 0}
    start = time.time()
    native_bucket = assign_gsd_bucket("vrsbench", None)

    # Track unique images to generate proxy only once
    generated_proxies: set[str] = set()

    for i, entry in enumerate(data[:total]):
        sample_id = f"{i:07d}"
        manifest_id = sample_id

        if manifest.is_processed(manifest_id):
            stats["skipped"] += 1
            continue

        try:
            samples = process_entry(entry, images_dir, sample_id=sample_id)
        except Exception as e:
            print(f"  [FAIL] {sample_id}: {e}", file=sys.stderr)
            stats["failed"] += 1
            continue

        if not samples:
            stats["failed"] += 1
            continue

        # Emit native + proxy rows for each sample (R4 dual-resolution)
        for sample in samples:
            sample["gsd_bucket"] = native_bucket
            append_jsonl(sample, jsonl_path)

            # Generate proxy image once per unique source image
            proxy_path = None
            if sample["image_path"] and images_dir:
                image_name = Path(sample["image_path"][0]).name
                if image_name not in generated_proxies:
                    generate_proxy_image(
                        Path(sample["image_path"][0]),
                        proxy_dir,
                        image_name,
                    )
                    generated_proxies.add(image_name)
                proxy_path = str(proxy_dir / image_name)

            proxy_sample = create_proxy_sample(sample)
            # Point proxy sample to the downsampled image
            if proxy_path:
                proxy_sample["image_path"] = [proxy_path]
            append_jsonl(proxy_sample, jsonl_path)

        manifest.mark_processed(
            sample_id=manifest_id,
            dataset=_DATASET, split="train", output_shard="local",
        )

        stats["processed"] += 1
        if (i + 1) % 10000 == 0 or (i + 1) == total:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  Processed {i + 1}/{total} ({rate:.1f} entries/s)")

    print(f"Done: {stats['processed']} processed, "
          f"{stats['skipped']} skipped, {stats['failed']} failed")
    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess VRSBench → JSONL + PNGs (caption + grounding + VQA)",
    )
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Root VRSBench directory")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for JSONL + images")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Process at most N entries (for testing)")
    args = parser.parse_args()

    run_tier1_vrsbench(args.input_dir, args.output_dir, max_samples=args.max_samples)
