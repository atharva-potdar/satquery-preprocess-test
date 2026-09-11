"""Tests for preprocess.split_internal_val — R4 internal validation split."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from preprocess.split_internal_val import (
    load_samples,
    split_and_write,
    split_all_tiers,
    stratified_split,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sample(
    dataset: str = "vrsbench",
    task: str = "vqa",
    sample_id: str = "",
) -> dict:
    return {
        "id": sample_id or f"{dataset}_{task}_{random.randint(0, 999999):06d}",
        "dataset": dataset,
        "task": task,
        "image_path": ["dummy.png"],
        "pair_type": "single",
        "gsd_bucket": "VHR-native",
        "split": "train",
        "instruction": "test instruction",
        "response": "test response",
        "bbox": None,
        "modality": "optical",
    }


def _write_jsonl(samples: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")


# ---------------------------------------------------------------------------
# load_samples
# ---------------------------------------------------------------------------

class TestLoadSamples:
    def test_loads_all_samples(self, tmp_path):
        samples = [_make_sample(sample_id=f"s{i}") for i in range(10)]
        path = tmp_path / "test.jsonl"
        _write_jsonl(samples, path)
        loaded = load_samples(path)
        assert len(loaded) == 10

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.touch()
        loaded = load_samples(path)
        assert loaded == []


# ---------------------------------------------------------------------------
# stratified_split
# ---------------------------------------------------------------------------

class TestStratifiedSplit:
    def test_total_preserved(self):
        samples = [_make_sample(task="vqa") for _ in range(100)]
        train, val = stratified_split(samples, val_fraction=0.03, seed=42)
        assert len(train) + len(val) == 100

    def test_val_fraction_approximate(self):
        samples = [_make_sample(task="vqa") for _ in range(1000)]
        train, val = stratified_split(samples, val_fraction=0.03, seed=42)
        ratio = len(val) / len(samples)
        assert 0.02 <= ratio <= 0.05, f"Val ratio {ratio:.3f} outside [0.02, 0.05]"

    def test_stratified_by_task(self):
        samples = (
            [_make_sample(task="vqa") for _ in range(100)] +
            [_make_sample(task="caption") for _ in range(100)] +
            [_make_sample(task="grounding") for _ in range(100)]
        )
        train, val = stratified_split(samples, val_fraction=0.03, seed=42)

        # Check each task has some val samples
        val_tasks = [s["task"] for s in val]
        for task in ["vqa", "caption", "grounding"]:
            assert task in val_tasks, f"Task {task} missing from val split"

    def test_stratified_by_dataset(self):
        samples = (
            [_make_sample(dataset="vrsbench") for _ in range(50)] +
            [_make_sample(dataset="oscd") for _ in range(50)]
        )
        train, val = stratified_split(samples, val_fraction=0.03, seed=42)

        val_datasets = [s["dataset"] for s in val]
        assert "vrsbench" in val_datasets
        assert "oscd" in val_datasets

    def test_deterministic_with_seed(self):
        samples = [_make_sample() for _ in range(100)]
        t1, v1 = stratified_split(samples, val_fraction=0.03, seed=42)
        t2, v2 = stratified_split(samples, val_fraction=0.03, seed=42)
        assert [s["id"] for s in v1] == [s["id"] for s in v2]

    def test_small_group_gets_minimum(self):
        # Group with only 2 samples → val gets 1, train gets 1
        samples = [_make_sample(task="rare_task", sample_id=f"rare_{i}") for i in range(2)]
        train, val = stratified_split(samples, val_fraction=0.03, seed=42)
        assert len(train) + len(val) == 2
        # With val_fraction=0.03, n_val = max(1, int(2*0.03))=1
        # Cap at len-1=1, so val=1, train=1
        assert len(val) == 1
        assert len(train) == 1

    def test_single_sample_group_skipped(self):
        # Group with 1 sample → train=0, val=0 (can't split)
        samples = [_make_sample(task="single_task", sample_id="only_one")]
        train, val = stratified_split(samples, val_fraction=0.03, seed=42)
        # n_val = min(1, 0) = 0, so everything goes to train
        assert len(train) == 1
        assert len(val) == 0


# ---------------------------------------------------------------------------
# split_and_write
# ---------------------------------------------------------------------------

class TestSplitAndWrite:
    def test_creates_two_files(self, tmp_path):
        samples = [_make_sample(sample_id=f"s{i}") for i in range(50)]
        input_path = tmp_path / "input.jsonl"
        _write_jsonl(samples, input_path)

        stats = split_and_write(input_path, tmp_path / "output", val_fraction=0.03)

        assert stats["total"] == 50
        assert stats["train"] + stats["val"] == 50
        assert (tmp_path / "output" / "input_train.jsonl").exists()
        assert (tmp_path / "output" / "input_val_internal.jsonl").exists()

    def test_val_samples_tagged(self, tmp_path):
        samples = [_make_sample(sample_id=f"s{i}") for i in range(50)]
        input_path = tmp_path / "input.jsonl"
        _write_jsonl(samples, input_path)

        split_and_write(input_path, tmp_path / "output", val_fraction=0.03)

        val_path = tmp_path / "output" / "input_val_internal.jsonl"
        val_samples = load_samples(val_path)
        for s in val_samples:
            assert s["split"] == "val_internal"

    def test_train_samples_keep_split(self, tmp_path):
        samples = [_make_sample(sample_id=f"s{i}") for i in range(50)]
        input_path = tmp_path / "input.jsonl"
        _write_jsonl(samples, input_path)

        split_and_write(input_path, tmp_path / "output", val_fraction=0.03)

        train_path = tmp_path / "output" / "input_train.jsonl"
        train_samples = load_samples(train_path)
        for s in train_samples:
            assert s["split"] == "train"

    def test_empty_input(self, tmp_path):
        input_path = tmp_path / "empty.jsonl"
        input_path.touch()
        stats = split_and_write(input_path, tmp_path / "output")
        assert stats == {"total": 0, "train": 0, "val": 0}


# ---------------------------------------------------------------------------
# split_all_tiers
# ---------------------------------------------------------------------------

class TestSplitAllTiers:
    def test_processes_multiple_jsonls(self, tmp_path):
        for name in ["tier1_a.jsonl", "tier1_b.jsonl"]:
            samples = [_make_sample(sample_id=f"s{i}") for i in range(30)]
            _write_jsonl(samples, tmp_path / name)

        results = split_all_tiers(tmp_path)
        assert len(results) == 2
        for name, stats in results.items():
            assert stats["total"] == 30
            assert stats["train"] + stats["val"] == 30

    def test_skips_already_split(self, tmp_path):
        samples = [_make_sample(sample_id=f"s{i}") for i in range(30)]
        _write_jsonl(samples, tmp_path / "data.jsonl")
        _write_jsonl(samples, tmp_path / "data_train.jsonl")
        _write_jsonl(samples, tmp_path / "data_val_internal.jsonl")

        results = split_all_tiers(tmp_path)
        assert len(results) == 1  # Only data.jsonl processed

    def test_no_jsonl_files(self, tmp_path):
        results = split_all_tiers(tmp_path)
        assert results == {}


# ---------------------------------------------------------------------------
# Stratification correctness
# ---------------------------------------------------------------------------

class TestStratificationCorrectness:
    def test_no_leakage(self):
        """No sample appears in both train and val."""
        samples = [_make_sample(sample_id=f"s{i}") for i in range(100)]
        train, val = stratified_split(samples, val_fraction=0.03, seed=42)
        train_ids = {s["id"] for s in train}
        val_ids = {s["id"] for s in val}
        assert train_ids.isdisjoint(val_ids)

    def test_all_samples_accounted_for(self):
        """Every sample ends up in exactly one split."""
        samples = [_make_sample(sample_id=f"s{i}") for i in range(100)]
        train, val = stratified_split(samples, val_fraction=0.03, seed=42)
        all_ids = {s["id"] for s in samples}
        result_ids = {s["id"] for s in train} | {s["id"] for s in val}
        assert all_ids == result_ids

    def test_no_split_field_on_train(self):
        """Train samples don't have split modified."""
        samples = [_make_sample(sample_id=f"s{i}") for i in range(100)]
        train, val = stratified_split(samples, val_fraction=0.03, seed=42)
        for s in train:
            assert s["split"] == "train"
