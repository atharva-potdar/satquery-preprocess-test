"""Tests for preprocess/tier1_oscd.py — manifest/resumability pattern.

Creates dummy OSCD data (individual band TIFs + masks) and verifies:
    1. First run processes all pairs and writes JSONL + PNGs
    2. Second run skips already-processed pairs (idempotent)
    3. Interrupted run can resume from manifest
    4. Output PNGs are byte-identical across runs (deterministic)
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import tifffile

from preprocess.common.io import Manifest
from preprocess.tier1_oscd import (
    find_oscd_pairs,
    load_bands_as_rgb,
    load_mask,
    make_sample_id,
    mask_to_bboxes,
    process_pair,
    run_tier1_oscd,
)

# Bands needed for RGB
_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]


# ---------------------------------------------------------------------------
# Fixtures: create synthetic OSCD directory structure (real layout)
# ---------------------------------------------------------------------------


def _create_dummy_bands(bands_dir: Path) -> None:
    """Create individual band TIFs in a directory."""
    bands_dir.mkdir(parents=True, exist_ok=True)
    for band in _BANDS:
        arr = np.random.randint(100, 8000, size=(64, 64), dtype=np.uint16)
        tifffile.imwrite(str(bands_dir / f"dummy_{band}.tif"), arr)


def _create_dummy_mask(path: Path, change_fraction: float = 0.1) -> None:
    """Create a synthetic binary change mask."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((64, 64), dtype=np.uint8)
    if change_fraction > 0:
        mask[16:48, 16:48] = 255  # Change region
    tifffile.imwrite(str(path), mask)


@pytest.fixture
def oscd_dir():
    """Create a temporary OSCD directory with 2 dummy pairs (real layout)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # location_a: has mask
        _create_dummy_bands(root / "images" / "location_a" / "imgs_1")
        _create_dummy_bands(root / "images" / "location_a" / "imgs_2")
        _create_dummy_mask(root / "labels" / "location_a" / "cm" / "location_a-cm.tif")

        # location_b: has mask
        _create_dummy_bands(root / "images" / "location_b" / "imgs_1")
        _create_dummy_bands(root / "images" / "location_b" / "imgs_2")
        _create_dummy_mask(root / "labels" / "location_b" / "cm" / "location_b-cm.tif")

        yield root


# ---------------------------------------------------------------------------
# Unit tests: helper functions
# ---------------------------------------------------------------------------


class TestFindOscdPairs:
    def test_discovers_all_pairs(self, oscd_dir):
        pairs = find_oscd_pairs(oscd_dir)
        assert len(pairs) == 2
        locations = {p["location"] for p in pairs}
        assert locations == {"location_a", "location_b"}

    def test_pair_has_required_keys(self, oscd_dir):
        pairs = find_oscd_pairs(oscd_dir)
        for pair in pairs:
            assert "location" in pair
            assert "before_bands_dir" in pair
            assert "after_bands_dir" in pair
            assert "mask_path" in pair
            assert pair["before_bands_dir"].exists()
            assert pair["after_bands_dir"].exists()

    def test_no_pairs_if_no_images(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "images").mkdir()
            (root / "labels").mkdir()
            pairs = find_oscd_pairs(root)
            assert len(pairs) == 0


class TestLoadBandsAsRgb:
    def test_extracts_rgb_from_bands(self, oscd_dir):
        bands_dir = oscd_dir / "images" / "location_a" / "imgs_1"
        rgb = load_bands_as_rgb(bands_dir)
        assert rgb.ndim == 3
        assert rgb.shape[2] == 3
        assert rgb.dtype == np.uint8
        assert rgb.max() <= 255

    def test_raises_on_missing_bands(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bands_dir = Path(tmpdir)
            # Only B01 — missing B04/B03/B02 which are in _BAND_PRIORITY
            tifffile.imwrite(str(bands_dir / "dummy_B01.tif"), np.zeros((16, 16), dtype=np.uint16))
            with pytest.raises(ValueError, match="Missing required band"):
                load_bands_as_rgb(bands_dir)


class TestLoadMask:
    def test_loads_binary_mask(self, oscd_dir):
        mask_path = oscd_dir / "labels" / "location_a" / "cm" / "location_a-cm.tif"
        mask = load_mask(mask_path)
        assert mask is not None
        assert mask.ndim == 2
        assert set(np.unique(mask)).issubset({0, 1})

    def test_returns_none_if_no_mask(self):
        result = load_mask(None)
        assert result is None

    def test_binarizes_255_to_1(self, oscd_dir):
        mask_path = oscd_dir / "labels" / "location_a" / "cm" / "location_a-cm.tif"
        mask = load_mask(mask_path)
        assert mask.max() <= 1


class TestMaskToBboxes:
    def test_finds_change_region(self, oscd_dir):
        mask_path = oscd_dir / "labels" / "location_a" / "cm" / "location_a-cm.tif"
        mask = load_mask(mask_path)
        bboxes = mask_to_bboxes(mask)
        assert len(bboxes) >= 1
        x1, y1, x2, y2 = bboxes[0]
        assert x1 < x2
        assert y1 < y2

    def test_empty_mask_returns_empty(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        bboxes = mask_to_bboxes(mask)
        assert bboxes == []


class TestMakeSampleId:
    def test_lowercase_underscored(self):
        sid = make_sample_id("Location-A")
        assert sid == "oscd_location_a"

    def test_max_length(self):
        sid = make_sample_id("a" * 100)
        assert len(sid) <= 64


# ---------------------------------------------------------------------------
# Integration tests: manifest/resumability pattern
# ---------------------------------------------------------------------------


class TestProcessPair:
    def test_returns_valid_sample(self, oscd_dir):
        pairs = find_oscd_pairs(oscd_dir)
        sample = process_pair(pairs[0], oscd_dir / "output")
        assert sample is not None
        assert sample["dataset"] == "oscd"
        assert sample["pair_type"] == "bitemporal"
        assert sample["modality"] == "optical"
        assert len(sample["image_path"]) == 2
        assert sample["bbox"] is not None

    def test_empty_mask_produces_null_bbox_change_vqa(self, oscd_dir):
        """Empty mask → bbox=null, task=change_vqa."""
        # Create a location with empty mask
        _create_dummy_bands(oscd_dir / "images" / "location_empty" / "imgs_1")
        _create_dummy_bands(oscd_dir / "images" / "location_empty" / "imgs_2")
        _create_dummy_mask(oscd_dir / "labels" / "location_empty" / "cm" / "location_empty-cm.tif",
                           change_fraction=0.0)

        pairs = find_oscd_pairs(oscd_dir)
        empty_pair = [p for p in pairs if p["location"] == "location_empty"][0]
        sample = process_pair(empty_pair, oscd_dir / "output")

        assert sample is not None
        assert sample["bbox"] is None, f"Expected null bbox for empty mask, got {sample['bbox']}"
        assert sample["task"] == "change_vqa", f"Expected change_vqa task, got {sample['task']}"

    def test_writes_pngs(self, oscd_dir):
        pairs = find_oscd_pairs(oscd_dir)
        output_dir = oscd_dir / "output"
        process_pair(pairs[0], output_dir)
        assert (output_dir / "images" / f"{pairs[0]['location']}_before.png").exists()
        assert (output_dir / "images" / f"{pairs[0]['location']}_after.png").exists()


class TestManifestResumability:
    """Core test: verify the manifest pattern works as designed."""

    def test_first_run_processes_all(self, oscd_dir):
        """First run should process all pairs and write manifest entries."""
        output_dir = oscd_dir / "output"
        stats = run_tier1_oscd(oscd_dir, output_dir)

        assert stats["processed"] == 2
        assert stats["skipped"] == 0
        assert stats["failed"] == 0

        jsonl_path = output_dir / "oscd.jsonl"
        assert jsonl_path.exists()
        with open(jsonl_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 2

    def test_second_run_skips_all(self, oscd_dir):
        """Second run should skip all already-processed pairs."""
        output_dir = oscd_dir / "output"

        stats1 = run_tier1_oscd(oscd_dir, output_dir)
        assert stats1["processed"] == 2

        stats2 = run_tier1_oscd(oscd_dir, output_dir)
        assert stats2["processed"] == 0
        assert stats2["skipped"] == 2

    def test_idempotent_output(self, oscd_dir):
        """Re-running produces byte-identical PNGs."""
        output_dir = oscd_dir / "output"

        run_tier1_oscd(oscd_dir, output_dir)

        png1 = output_dir / "images" / "location_a_before.png"
        hash1 = hashlib.md5(png1.read_bytes()).hexdigest()

        run_tier1_oscd(oscd_dir, output_dir)
        hash2 = hashlib.md5(png1.read_bytes()).hexdigest()

        assert hash1 == hash2, "PNG changed across runs — not idempotent"

    def test_manifest_counts_accurate(self, oscd_dir):
        """Manifest processed count matches actual output."""
        output_dir = oscd_dir / "output"
        manifest_path = output_dir / "manifest.parquet"

        run_tier1_oscd(oscd_dir, output_dir)

        manifest = Manifest(manifest_path)
        assert manifest.count_processed() == 2

    def test_partial_run_resumes(self, oscd_dir):
        """Simulate interrupted run: process 1 pair, then resume."""
        output_dir = oscd_dir / "output"
        manifest_path = output_dir / "manifest.parquet"
        jsonl_path = output_dir / "oscd.jsonl"

        # Manually process only the first pair
        pairs = find_oscd_pairs(oscd_dir)
        manifest = Manifest(manifest_path)
        sample = process_pair(pairs[0], output_dir)
        assert sample is not None

        from preprocess.common.io import append_jsonl
        append_jsonl(sample, jsonl_path)
        manifest.mark_processed(
            sample_id=make_sample_id(pairs[0]["location"]),
            dataset="oscd", split="train", output_shard="local",
        )

        # Now run the full pipeline — should process only the second pair
        stats = run_tier1_oscd(oscd_dir, output_dir)
        assert stats["processed"] == 1
        assert stats["skipped"] == 1

        with open(jsonl_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 2


class TestValidation:
    """Verify produced samples pass the frozen schema."""

    def test_all_samples_validate(self, oscd_dir):
        pairs = find_oscd_pairs(oscd_dir)
        output_dir = oscd_dir / "output"
        for pair in pairs:
            sample = process_pair(pair, output_dir)
            assert sample is not None
            from preprocess.validator import validate_sample
            ok, errs = validate_sample(sample)
            assert ok, f"Validation failed for {pair['location']}: {errs}"
