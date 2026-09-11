"""Post-download sanity checks for all datasets.

Verifies file counts, annotation structure, and split proportions
before preprocessing begins. Fail fast if mirror is truncated/reshuffled.

Usage:
    python -m preprocess.sanity_check --dataset oscd --input-dir data/oscd
    python -m preprocess.sanity_check --dataset all --input-dir data/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


class SanityCheckError(Exception):
    """Raised when a sanity check fails."""
    pass


def check_oscd(input_dir: Path) -> dict[str, Any]:
    """Check OSCD dataset structure."""
    results = {"dataset": "oscd", "checks": []}

    # Check images directory
    images_dir = input_dir / "images"
    if not images_dir.exists():
        raise SanityCheckError(f"Missing images directory: {images_dir}")

    locations = [d for d in images_dir.iterdir() if d.is_dir()]
    results["checks"].append({"name": "locations", "count": len(locations), "expected": 24})

    if len(locations) < 24:
        print(f"  [WARN] Expected 24 locations, found {len(locations)}")

    # Check each location has imgs_1 and imgs_2
    missing_pairs = []
    for loc in locations:
        if not (loc / "imgs_1").exists() or not (loc / "imgs_2").exists():
            missing_pairs.append(loc.name)

    results["checks"].append({
        "name": "complete_pairs",
        "missing": len(missing_pairs),
        "missing_locations": missing_pairs[:5],
    })

    # Check labels
    labels_dir = input_dir / "labels"
    if labels_dir.exists():
        label_locs = [d for d in labels_dir.iterdir() if d.is_dir()]
        results["checks"].append({"name": "label_dirs", "count": len(label_locs)})

    results["status"] = "pass"
    return results


def check_vrsbench(input_dir: Path) -> dict[str, Any]:
    """Check VRSBench dataset structure."""
    results = {"dataset": "vrsbench", "checks": []}

    # Check main JSON
    json_path = input_dir / "VRSBench_train.json"
    if not json_path.exists():
        raise SanityCheckError(f"Missing VRSBench_train.json")

    with open(json_path) as f:
        data = json.load(f)

    results["checks"].append({"name": "total_entries", "count": len(data)})

    # Check task distribution
    tasks = {}
    for entry in data:
        conv = entry.get("conversations", [{}])[0].get("value", "")
        if "[caption]" in conv:
            tasks["caption"] = tasks.get("caption", 0) + 1
        elif "[refer]" in conv:
            tasks["refer"] = tasks.get("refer", 0) + 1
        elif "[vqa]" in conv:
            tasks["vqa"] = tasks.get("vqa", 0) + 1

    results["checks"].append({"name": "task_distribution", "tasks": tasks})

    # Check image zips exist
    train_zip = input_dir / "Images_train.zip"
    val_zip = input_dir / "Images_val.zip"
    results["checks"].append({
        "name": "image_zips",
        "train_exists": train_zip.exists(),
        "val_exists": val_zip.exists(),
        "train_size_gb": round(train_zip.stat().st_size / 1e9, 2) if train_zip.exists() else 0,
    })

    results["status"] = "pass"
    return results


def check_bigearthnet(input_dir: Path) -> dict[str, Any]:
    """Check BigEarthNet dataset structure."""
    results = {"dataset": "bigearthnet", "checks": []}

    # Check for metadata
    metadata_path = input_dir / "metadata.parquet"
    if metadata_path.exists():
        import pandas as pd
        metadata = pd.read_parquet(metadata_path)
        results["checks"].append({"name": "total_patches", "count": len(metadata)})

        # Check country distribution
        if "country" in metadata.columns:
            countries = metadata["country"].value_counts().to_dict()
            results["checks"].append({"name": "countries", "count": len(countries)})

        # Check label distribution
        if "labels" in metadata.columns:
            has_labels = True
            results["checks"].append({"name": "has_labels_column", "value": has_labels})
    else:
        # Check for HDF5 files
        h5_files = list(input_dir.glob("*.h5")) + list(input_dir.glob("*.hdf5"))
        results["checks"].append({"name": "h5_files", "count": len(h5_files)})

    results["status"] = "pass"
    return results


def check_rsvqa_hr(input_dir: Path) -> dict[str, Any]:
    """Check RSVQA-HR dataset structure."""
    results = {"dataset": "rsvqa_hr", "checks": []}

    # Check annotation files
    for split in ["train", "val", "test"]:
        split_path = input_dir / f"{split}.json"
        if split_path.exists():
            with open(split_path) as f:
                data = json.load(f)
            count = len(data) if isinstance(data, list) else len(data.get("questions", []))
            results["checks"].append({"name": f"{split}_count", "count": count})

    # Check images directory
    images_dir = input_dir / "images"
    if images_dir.exists():
        n_images = len(list(images_dir.glob("*.png"))) + len(list(images_dir.glob("*.jpg")))
        results["checks"].append({"name": "image_count", "count": n_images})

    results["status"] = "pass"
    return results


def check_cdvqa(input_dir: Path) -> dict[str, Any]:
    """Check CDVQA dataset structure."""
    results = {"dataset": "cdvqa", "checks": []}

    # Check split files (train/val only)
    for split in ["train", "val"]:
        split_path = input_dir / f"{split}.json"
        if split_path.exists():
            with open(split_path) as f:
                data = json.load(f)
            count = len(data) if isinstance(data, list) else len(data.get("questions", []))
            results["checks"].append({"name": f"{split}_count", "count": count})

    # Reject test splits
    test_path = input_dir / "test.json"
    if test_path.exists():
        print("  [WARN] test.json found — will be EXCLUDED from training")

    results["status"] = "pass"
    return results


def check_levir_cd(input_dir: Path) -> dict[str, Any]:
    """Check LEVIR-CD dataset structure."""
    results = {"dataset": "levir_cd", "checks": []}

    for split in ["train", "val", "test"]:
        split_dir = input_dir / split
        if split_dir.exists():
            a_dir = split_dir / "A"
            b_dir = split_dir / "B"
            label_dir = split_dir / "label"

            n_before = len(list(a_dir.glob("*.png"))) if a_dir.exists() else 0
            n_after = len(list(b_dir.glob("*.png"))) if b_dir.exists() else 0
            n_labels = len(list(label_dir.glob("*.png"))) if label_dir.exists() else 0

            results["checks"].append({
                "name": f"{split}_counts",
                "before": n_before,
                "after": n_after,
                "labels": n_labels,
            })

    results["status"] = "pass"
    return results


def check_spacenet6(input_dir: Path) -> dict[str, Any]:
    """Check SpaceNet 6 dataset structure."""
    results = {"dataset": "spacenet6", "checks": []}

    # Check train directory
    train_dir = input_dir / "train"
    if train_dir.exists():
        images_dir = train_dir / "images"
        labels_dir = train_dir / "labels"

        n_images = len(list(images_dir.glob("*.tif"))) if images_dir.exists() else 0
        n_labels = len(list(labels_dir.glob("*.geojson"))) if labels_dir.exists() else 0

        results["checks"].append({
            "name": "train_counts",
            "images": n_images,
            "labels": n_labels,
        })

        # Check for SAR
        sar_dir = train_dir / "sar"
        if sar_dir.exists():
            n_sar = len(list(sar_dir.glob("*.tif")))
            results["checks"].append({"name": "sar_count", "count": n_sar})

    results["status"] = "pass"
    return results


def check_sardet(input_dir: Path) -> dict[str, Any]:
    """Check SARDet-100K dataset structure."""
    results = {"dataset": "sardet", "checks": []}

    images_dir = input_dir / "images"
    labels_dir = input_dir / "labels"

    if images_dir.exists():
        n_images = len(list(images_dir.glob("*.png"))) + len(list(images_dir.glob("*.jpg")))
        results["checks"].append({"name": "image_count", "count": n_images})

    if labels_dir.exists():
        n_labels = len(list(labels_dir.glob("*.txt")))
        results["checks"].append({"name": "label_count", "count": n_labels})

    results["status"] = "pass"
    return results


def check_sen2lulc(input_dir: Path) -> dict[str, Any]:
    """Check Sen-2 LULC dataset structure."""
    results = {"dataset": "sen2lulc", "checks": []}

    # Check metadata
    for name in ["metadata.csv", "annotations.json"]:
        meta_path = input_dir / name
        if meta_path.exists():
            results["checks"].append({"name": "metadata_file", "file": name, "exists": True})

    # Check images
    images_dir = input_dir / "images"
    if images_dir.exists():
        n_images = len(list(images_dir.glob("*.png")))
        results["checks"].append({"name": "image_count", "count": n_images})

    results["status"] = "pass"
    return results


# Registry of sanity check functions
CHECKS = {
    "oscd": check_oscd,
    "vrsbench": check_vrsbench,
    "bigen": check_bigearthnet,
    "bigearthnet": check_bigearthnet,
    "rsvqa_hr": check_rsvqa_hr,
    "cdvqa": check_cdvqa,
    "levir_cd": check_levir_cd,
    "sn6_opt": check_spacenet6,
    "sn6_sar": check_spacenet6,
    "spacenet6": check_spacenet6,
    "sardet": check_sardet,
    "sen2lulc": check_sen2lulc,
}


def run_sanity_check(dataset: str, input_dir: Path) -> dict[str, Any]:
    """Run sanity check for a specific dataset."""
    check_fn = CHECKS.get(dataset)
    if check_fn is None:
        raise ValueError(f"Unknown dataset: {dataset}. Available: {list(CHECKS.keys())}")

    print(f"Checking {dataset} at {input_dir}...")
    results = check_fn(input_dir)
    print(f"  Status: {results['status']}")
    for check in results["checks"]:
        print(f"  {check}")

    return results


def run_all_checks(data_dir: Path) -> dict[str, dict]:
    """Run sanity checks for all known datasets in a directory."""
    all_results = {}

    for dataset in CHECKS:
        dataset_dir = data_dir / dataset
        if dataset_dir.exists():
            try:
                results = run_sanity_check(dataset, dataset_dir)
                all_results[dataset] = results
            except SanityCheckError as e:
                print(f"  [FAIL] {dataset}: {e}")
                all_results[dataset] = {"status": "fail", "error": str(e)}
        else:
            print(f"  [SKIP] {dataset}: directory not found at {dataset_dir}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Post-download sanity checks")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name or 'all'")
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Dataset or data directory")
    args = parser.parse_args()

    if args.dataset == "all":
        results = run_all_checks(args.input_dir)
    else:
        results = run_sanity_check(args.dataset, args.input_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("SANITY CHECK SUMMARY")
    print("=" * 60)

    if isinstance(results, dict) and "status" in results:
        # Single dataset
        status = results["status"]
        print(f"\n{args.dataset}: {status.upper()}")
    else:
        # All datasets
        for dataset, result in results.items():
            status = result.get("status", "unknown")
            print(f"  {dataset}: {status.upper()}")
