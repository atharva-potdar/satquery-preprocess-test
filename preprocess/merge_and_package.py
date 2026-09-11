"""Merge and package processed JSONLs into final dataset.

Validates every sample against frozen schema, merges tier outputs into a
single dataset.jsonl, and optionally tars image shards for Kaggle upload.

Usage:
    python -m preprocess.merge_and_package --tier-dirs data/oscd_output data/vrsbench_output --output-dir data/final
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

from preprocess.common.io import ShardWriter
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


def _select_source_files(jsonl_files: list[Path]) -> list[Path]:
    """Pick which JSONL files in a tier dir actually feed the merge.

    split_internal_val.py (when it has run) replaces `<stem>.jsonl` with
    `<stem>_train.jsonl` + `<stem>_val_internal.jsonl` — those carry the
    real split tags and must be preferred. Falling back to the original
    `<stem>.jsonl` silently drops every val_internal row (it only exists
    with everything tagged "train"), which is what happened here before:
    merge always skipped `_train`/`_val_internal` files and re-read
    the un-split original.
    """
    by_stem = {p.stem: p for p in jsonl_files}
    train_suffix = "_train"
    val_suffix = "_val_internal"

    split_bases = {
        stem[: -len(train_suffix)]
        for stem in by_stem
        if stem.endswith(train_suffix)
    }

    selected: list[Path] = []
    for stem, path in by_stem.items():
        if stem.endswith(train_suffix) or stem.endswith(val_suffix):
            selected.append(path)
            continue
        if stem in split_bases:
            continue  # superseded by <stem>_train.jsonl / _val_internal.jsonl
        selected.append(path)  # no split output for this file — use as-is

    return selected


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

        for jsonl_path in _select_source_files(jsonl_files):
            samples = load_jsonl(jsonl_path)
            valid, errors = validate_and_filter(samples, tier_name)
            all_valid.extend(valid)
            all_errors.extend(errors)
            print(f"  {tier_name}/{jsonl_path.name}: {len(valid)} valid, {len(errors)} errors")

    return all_valid, all_errors


def check_id_uniqueness(samples: list[dict]) -> list[str]:
    """Return list of duplicate IDs, formatted as '<id> (×N)'."""
    seen: dict[str, int] = {}
    for s in samples:
        sid = s.get("id", "")
        seen[sid] = seen.get(sid, 0) + 1
    return [f"{sid} (×{count})" for sid, count in seen.items() if count > 1]


def dedupe_by_id(samples: list[dict]) -> tuple[list[dict], int]:
    """Keep the first occurrence of each id. Returns (deduped, n_dropped).

    "id" is documented as globally unique (Section 4) — the schema can't
    enforce that across files, so this is where cross-tier collisions
    (mostly a same-id `id` scheme reused between two tier scripts) get
    caught before they reach the training set as silent duplicates.
    """
    seen: set[str] = set()
    out: list[dict] = []
    dropped = 0
    for s in samples:
        sid = s.get("id", "")
        if sid in seen:
            dropped += 1
            continue
        seen.add(sid)
        out.append(s)
    return out, dropped


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
    tar_dir: Path,
    *,
    prefix: str,
    include_jsonl: bool = False,
    max_files_per_shard: int = 2000,
) -> int:
    """Shard a directory of images into <=max_files_per_shard tar.gz archives.

    Kaggle Dataset uploads cap raw file counts (Section 8), and a single
    giant tar risks losing an entire tier's images to one failed
    upload/session. ShardWriter (already tested) produces
    `<prefix>_shard_0000.tar.gz`, `_0001.tar.gz`, ... in tar_dir.

    Returns total file count across all shards.
    """
    writer = ShardWriter(tar_dir, prefix=prefix, max_files=max_files_per_shard)
    count = 0
    for path in sorted(src_dir.rglob("*")):
        if path.is_file():
            if not include_jsonl and path.suffix == ".jsonl":
                continue
            writer.add(path)
            count += 1
    writer.close()
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

    # Check + drop ID collisions — a duplicate id must not reach the
    # training set silently (it did before: this was report-only).
    dupes = check_id_uniqueness(all_valid)
    if dupes:
        print(f"\n  {len(dupes)} duplicate IDs (keeping first occurrence):")
        for d in dupes[:10]:
            print(f"    {d}")
    all_valid, n_dropped = dedupe_by_id(all_valid)

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

    # Shard-tar images if requested (Kaggle Dataset file-count cap: Section 8)
    if tar_shards:
        for tier_dir in tier_dirs:
            images_dir = tier_dir / "images"
            if images_dir.exists():
                count = tar_directory(
                    images_dir, output_dir, prefix=f"{tier_dir.name}_images",
                )
                print(f"  Tared {count} files → {output_dir}/{tier_dir.name}_images_shard_*.tar.gz")

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
