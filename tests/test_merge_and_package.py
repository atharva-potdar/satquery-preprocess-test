"""Tests for preprocess.merge_and_package — merge & validation pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from preprocess.merge_and_package import (
    check_id_uniqueness,
    compute_stats,
    load_jsonl,
    merge_tiers,
    tar_directory,
    validate_and_filter,
    write_dataset,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sample(
    dataset: str = "vrsbench",
    task: str = "vqa",
    sample_id: str = "",
    bbox: list | None = None,
    image_path: list | None = None,
) -> dict:
    return {
        "id": sample_id or f"{dataset}_{task}_000001",
        "dataset": dataset,
        "task": task,
        "image_path": image_path or ["dummy.png"],
        "pair_type": "single",
        "gsd_bucket": "VHR-native",
        "split": "train",
        "instruction": "test instruction",
        "response": "test response",
        "bbox": bbox,
        "modality": "optical",
    }


def _write_jsonl(samples: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")


# ---------------------------------------------------------------------------
# load_jsonl
# ---------------------------------------------------------------------------

class TestLoadJsonl:
    def test_loads_all(self, tmp_path):
        samples = [_make_sample(sample_id=f"s{i}") for i in range(5)]
        _write_jsonl(samples, tmp_path / "test.jsonl")
        loaded = load_jsonl(tmp_path / "test.jsonl")
        assert len(loaded) == 5

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.touch()
        assert load_jsonl(path) == []


# ---------------------------------------------------------------------------
# validate_and_filter
# ---------------------------------------------------------------------------

class TestValidateAndFilter:
    def test_valid_sample(self):
        sample = _make_sample()
        valid, errors = validate_and_filter([sample], "test")
        assert len(valid) == 1
        assert errors == []

    def test_invalid_sample(self):
        sample = _make_sample()
        del sample["dataset"]  # Missing required field
        valid, errors = validate_and_filter([sample], "test")
        assert len(valid) == 0
        assert len(errors) == 1
        assert errors[0]["tier"] == "test"


# ---------------------------------------------------------------------------
# check_id_uniqueness
# ---------------------------------------------------------------------------

class TestCheckIdUniqueness:
    def test_no_duplicates(self):
        samples = [_make_sample(sample_id=f"s{i}") for i in range(5)]
        dupes = check_id_uniqueness(samples)
        assert dupes == []

    def test_finds_duplicates(self):
        samples = [
            _make_sample(sample_id="dup"),
            _make_sample(sample_id="dup"),
            _make_sample(sample_id="unique"),
        ]
        dupes = check_id_uniqueness(samples)
        assert len(dupes) == 1
        assert "dup" in dupes[0]


# ---------------------------------------------------------------------------
# write_dataset
# ---------------------------------------------------------------------------

class TestWriteDataset:
    def test_writes_jsonl(self, tmp_path):
        samples = [_make_sample(sample_id=f"s{i}") for i in range(3)]
        out = tmp_path / "out" / "dataset.jsonl"
        write_dataset(samples, out)
        assert out.exists()
        loaded = load_jsonl(out)
        assert len(loaded) == 3

    def test_creates_parent_dirs(self, tmp_path):
        samples = [_make_sample()]
        out = tmp_path / "a" / "b" / "c" / "dataset.jsonl"
        write_dataset(samples, out)
        assert out.exists()


# ---------------------------------------------------------------------------
# merge_tiers
# ---------------------------------------------------------------------------

class TestMergeTiers:
    def test_merges_multiple_tiers(self, tmp_path):
        # Create tier1 with 2 samples
        tier1 = tmp_path / "tier1"
        tier1.mkdir()
        _write_jsonl(
            [_make_sample(dataset="vrsbench", sample_id="v1"),
             _make_sample(dataset="vrsbench", sample_id="v2")],
            tier1 / "vrsbench.jsonl",
        )

        # Create tier2 with 3 samples
        tier2 = tmp_path / "tier2"
        tier2.mkdir()
        _write_jsonl(
            [_make_sample(dataset="oscd", sample_id="o1"),
             _make_sample(dataset="oscd", sample_id="o2"),
             _make_sample(dataset="oscd", sample_id="o3")],
            tier2 / "oscd.jsonl",
        )

        valid, errors = merge_tiers([tier1, tier2])
        assert len(valid) == 5
        assert errors == []

    def test_skips_split_files(self, tmp_path):
        tier = tmp_path / "tier1"
        tier.mkdir()
        _write_jsonl(
            [_make_sample(sample_id="main")],
            tier / "data.jsonl",
        )
        _write_jsonl(
            [_make_sample(sample_id="train")],
            tier / "data_train.jsonl",
        )
        _write_jsonl(
            [_make_sample(sample_id="val")],
            tier / "data_val_internal.jsonl",
        )

        valid, errors = merge_tiers([tier])
        assert len(valid) == 1  # Only main.jsonl
        assert valid[0]["id"] == "main"

    def test_empty_tier(self, tmp_path):
        tier = tmp_path / "empty_tier"
        tier.mkdir()
        valid, errors = merge_tiers([tier])
        assert valid == []


# ---------------------------------------------------------------------------
# tar_directory
# ---------------------------------------------------------------------------

class TestTarDirectory:
    def test_creates_tarball(self, tmp_path):
        src = tmp_path / "images"
        src.mkdir()
        (src / "a.png").write_bytes(b"\x89PNG")
        (src / "b.png").write_bytes(b"\x89PNG")

        tar_path = tmp_path / "out.tar.gz"
        count = tar_directory(src, tar_path)
        assert count == 2
        assert tar_path.exists()

    def test_skips_jsonl(self, tmp_path):
        src = tmp_path / "images"
        src.mkdir()
        (src / "a.png").write_bytes(b"\x89PNG")
        (src / "data.jsonl").write_text("{}")

        tar_path = tmp_path / "out.tar.gz"
        count = tar_directory(src, tar_path, include_jsonl=False)
        assert count == 1  # Only PNG


# ---------------------------------------------------------------------------
# compute_stats
# ---------------------------------------------------------------------------

class TestComputeStats:
    def test_basic_stats(self):
        samples = [
            _make_sample(dataset="vrsbench", task="vqa", bbox=None),
            _make_sample(dataset="vrsbench", task="caption", bbox=None),
            _make_sample(dataset="oscd", task="change_grounding",
                        bbox=[[100, 200, 300, 400]]),
        ]
        stats = compute_stats(samples)
        assert stats["total_samples"] == 3
        assert stats["by_dataset"]["vrsbench"] == 2
        assert stats["by_dataset"]["oscd"] == 1
        assert stats["by_task"]["vqa"] == 1
        assert stats["by_task"]["caption"] == 1
        assert stats["by_task"]["change_grounding"] == 1

    def test_bbox_coverage(self):
        samples = [
            _make_sample(bbox=[[100, 200, 300, 400]]),
            _make_sample(bbox=None),
        ]
        stats = compute_stats(samples)
        assert "1/2" in stats["bbox_coverage"]

    def test_empty_samples(self):
        stats = compute_stats([])
        assert stats["total_samples"] == 0
