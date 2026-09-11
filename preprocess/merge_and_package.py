"""Merge and package processed JSONLs into final dataset.

Validates every sample against frozen schema, merges tier outputs into a
single dataset.jsonl, and optionally tars image shards for Kaggle upload.

Usage:
    python -m preprocess.merge_and_package --tier-dirs data/oscd_output data/vrsbench_output --output-dir data/final
"""

from __future__ import annotations

import json
import tarfile
import time
from collections import Counter
from pathlib import Path

from preprocess.validator import validate_sample


def load_jsonl(path: Path) -> list[dict]:
    """Load all samples from a JSONL file."""
    samples = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def validate_and_filter(
    samples: list[dict],
    tier_name: str,
) -> tuple[list[dict], list[dict]]:
    """Validate samples, returning (valid, errors)."""
    valid = []
    errors = []
    for s in samples:
        ok, errs = validate_sample(s)
        if ok:
            valid.append(s)
        else:
            errors.append({"id": s.get("id", "?"), "tier": tier_name, "errors": errs})
    return valid, errors


def merge_tiers(
    tier_dirs: list[Path],
) -> tuple[list[dict], list[dict]]:
    """Merge all tier outputs, validating each sample.

    Returns (all_valid, all_errors).
    """
    all_valid = []
    all_errors = []

    for tier_dir in tier_dirs:
        tier_name = tier_dir.name
        jsonl_files = list(tier_dir.glob("*.jsonl"))
        if not jsonl_files:
            print(f"  [WARN] No JSONL files in {tier_dir}")
            continue

        for jsonl_path in jsonl_files:
            if "_train" in jsonl_path.stem or "_val_internal" in jsonl_path.stem:
                continue  # Skip split files, use originals
            samples = load_jsonl(jsonl_path)
            valid, errors = validate_and_filter(samples, tier_name)
            all_valid.extend(valid)
            all_errors.extend(errors)
            print(f"  {tier_name}/{jsonl_path.name}: {len(valid)} valid, {len(errors)} errors")

    return all_valid, all_errors


def check_id_uniqueness(samples: list[dict]) -> list[str]:
    """Return list of duplicate IDs."""
    seen: dict[str, int] = {}
    duplicates = []
    for s in samples:
        sid = s.get("id", "")
        seen[sid] = seen.get(sid, 0) + 1
    for sid, count in seen.items():
        if count > 1:
            duplicates.append(f"{sid} (×{count})")
    return duplicates


def write_dataset(
    samples: list[dict],
    output_path: Path,
) -> None:
    """Write merged samples to a single JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")


def tar_directory(
    src_dir: Path,
    tar_path: Path,
    *,
    include_jsonl: bool = False,
) -> int:
    """Tar a directory of images. Returns file count."""
    count = 0
    with tarfile.open(tar_path, "w:gz") as tar:
        for path in sorted(src_dir.rglob("*")):
            if path.is_file():
                if not include_jsonl and path.suffix == ".jsonl":
                    continue
                tar.add(path, arcname=path.relative_to(src_dir.parent))
                count += 1
    return count


def compute_stats(samples: list[dict]) -> dict:
    """Compute dataset statistics."""
    stats: dict = {}
    stats["total_samples"] = len(samples)

    # By dataset
    by_dataset = Counter(s.get("dataset", "?") for s in samples)
    stats["by_dataset"] = dict(by_dataset)

    # By task
    by_task = Counter(s.get("task", "?") for s in samples)
    stats["by_task"] = dict(by_task)

    # By pair_type
    by_pair = Counter(s.get("pair_type", "?") for s in samples)
    stats["by_pair_type"] = dict(by_pair)

    # By modality
    by_mod = Counter(s.get("modality", "?") for s in samples)
    stats["by_modality"] = dict(by_mod)

    # By split
    by_split = Counter(s.get("split", "?") for s in samples)
    stats["by_split"] = dict(by_split)

    # Bbox coverage
    has_bbox = sum(1 for s in samples if s.get("bbox") is not None)
    stats["bbox_coverage"] = f"{has_bbox}/{len(samples)} ({100*has_bbox/max(1,len(samples)):.1f}%)"

    # Image path stats
    single = sum(1 for s in samples if len(s.get("image_path", [])) == 1)
    paired = sum(1 for s in samples if len(s.get("image_path", [])) == 2)
    stats["image_paths"] = {"single": single, "paired": paired}

    return stats


def merge_and_package(
    tier_dirs: list[Path],
    output_dir: Path,
    *,
    tar_shards: bool = False,
) -> dict:
    """Full merge pipeline: validate → deduplicate → merge → package.

    Returns stats dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    print("Merging tiers...")
    all_valid, all_errors = merge_tiers(tier_dirs)

    if all_errors:
        print(f"\n  {len(all_errors)} validation errors:")
        for e in all_errors[:10]:
            print(f"    {e['id']}: {e['errors']}")
        if len(all_errors) > 10:
            print(f"    ... and {len(all_errors) - 10} more")

    # Check ID uniqueness
    dupes = check_id_uniqueness(all_valid)
    if dupes:
        print(f"\n  {len(dupes)} duplicate IDs:")
        for d in dupes[:10]:
            print(f"    {d}")

    # Write merged dataset
    dataset_path = output_dir / "dataset.jsonl"
    write_dataset(all_valid, dataset_path)
    print(f"\n  Wrote {len(all_valid)} samples to {dataset_path}")

    # Compute stats
    stats = compute_stats(all_valid)
    stats["errors"] = len(all_errors)
    stats["duplicate_ids"] = len(dupes)

    # Save stats
    stats_path = output_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # Tar image shards if requested
    if tar_shards:
        for tier_dir in tier_dirs:
            images_dir = tier_dir / "images"
            if images_dir.exists():
                tar_path = output_dir / f"{tier_dir.name}_images.tar.gz"
                count = tar_directory(images_dir, tar_path)
                print(f"  Tared {count} files → {tar_path}")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s")

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Merge tier outputs into final dataset",
    )
    parser.add_argument(
        "--tier-dirs", type=Path, nargs="+", required=True,
        help="Tier output directories to merge",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output directory for merged dataset",
    )
    parser.add_argument(
        "--tar-shards", action="store_true",
        help="Tar image directories for Kaggle upload",
    )
    args = parser.parse_args()

    stats = merge_and_package(
        args.tier_dirs,
        args.output_dir,
        tar_shards=args.tar_shards,
    )

    print(f"\nStats:")
    print(json.dumps(stats, indent=2))
