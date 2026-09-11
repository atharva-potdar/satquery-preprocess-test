"""Tests for preprocess.tier1_levir_cd — LEVIR-CD preprocessing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from preprocess.tier1_levir_cd import (
    change_fraction,
    find_levir_pairs,
    mask_to_bbox,
    process_pair,
    run_tier1_levir_cd,
)


class TestFindLevirPairs:
    def test_discovers_pairs(self, tmp_path):
        # Create train/A, train/B, train/label structure
        split_dir = tmp_path / "train"
        (split_dir / "A").mkdir(parents=True)
        (split_dir / "B").mkdir(parents=True)
        (split_dir / "label").mkdir(parents=True)

        (split_dir / "A" / "img001_1.png").write_bytes(b"\x89PNG")
        (split_dir / "B" / "img001_2.png").write_bytes(b"\x89PNG")
        (split_dir / "label" / "img001.png").write_bytes(b"\x89PNG")

        pairs = find_levir_pairs(split_dir)
        assert len(pairs) == 1
        assert pairs[0]["base_name"] == "img001"

    def test_missing_after_skipped(self, tmp_path):
        split_dir = tmp_path / "train"
        (split_dir / "A").mkdir(parents=True)
        (split_dir / "B").mkdir(parents=True)

        (split_dir / "A" / "img001_1.png").write_bytes(b"\x89PNG")
        # No corresponding _2.png in B

        pairs = find_levir_pairs(split_dir)
        assert len(pairs) == 0

    def test_empty_dir(self, tmp_path):
        split_dir = tmp_path / "train"
        split_dir.mkdir()
        pairs = find_levir_pairs(split_dir)
        assert pairs == []


class TestMaskToBbox:
    def test_finds_change_region(self, tmp_path):
        import numpy as np
        from PIL import Image

        # Create mask with change in center
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:40, 20:40] = 255

        mask_path = tmp_path / "mask.png"
        Image.fromarray(mask).save(str(mask_path))

        bbox = mask_to_bbox(mask_path, (64, 64))
        assert bbox is not None
        assert len(bbox) == 1
        assert len(bbox[0]) == 4

    def test_empty_mask_returns_none(self, tmp_path):
        import numpy as np
        from PIL import Image

        mask = np.zeros((64, 64), dtype=np.uint8)
        mask_path = tmp_path / "mask.png"
        Image.fromarray(mask).save(str(mask_path))

        bbox = mask_to_bbox(mask_path, (64, 64))
        assert bbox is None


class TestProcessPair:
    def test_returns_valid_sample(self, tmp_path):
        from PIL import Image
        import numpy as np

        # Create images
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img.save(str(tmp_path / "before.png"))
        img.save(str(tmp_path / "after.png"))

        # Create mask
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:40, 20:40] = 255
        Image.fromarray(mask).save(str(tmp_path / "mask.png"))

        pair = {
            "base_name": "test001",
            "before": tmp_path / "before.png",
            "after": tmp_path / "after.png",
            "mask": tmp_path / "mask.png",
        }
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        sample = process_pair(pair, output_dir)
        assert sample is not None
        assert sample["id"] == "levir_test001"
        assert sample["dataset"] == "levir_cd"
        assert sample["pair_type"] == "bitemporal"
        assert sample["bbox"] is not None


class TestChangeFraction:
    def test_computes_fraction(self, tmp_path):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[0:32, :] = 255  # half changed
        mask_path = tmp_path / "mask.png"
        Image.fromarray(mask).save(str(mask_path))
        assert abs(change_fraction(mask_path) - 0.5) < 0.01

    def test_none_path_returns_zero(self):
        assert change_fraction(None) == 0.0


def _make_levir_input(root: Path, n_low: int, n_high: int) -> None:
    """n_low pairs with ~1% change, n_high pairs with ~80% change."""
    train_dir = root / "train"
    (train_dir / "A").mkdir(parents=True)
    (train_dir / "B").mkdir(parents=True)
    (train_dir / "label").mkdir(parents=True)

    idx = 0
    for _ in range(n_low):
        name = f"low{idx:03d}"
        Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)).save(
            str(train_dir / "A" / f"{name}_1.png"))
        Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)).save(
            str(train_dir / "B" / f"{name}_2.png"))
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[0:2, 0:2] = 255  # ~1.5% change
        Image.fromarray(mask).save(str(train_dir / "label" / f"{name}.png"))
        idx += 1

    for _ in range(n_high):
        name = f"high{idx:03d}"
        Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)).save(
            str(train_dir / "A" / f"{name}_1.png"))
        Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)).save(
            str(train_dir / "B" / f"{name}_2.png"))
        mask = np.full((32, 32), 255, dtype=np.uint8)  # ~100% change
        Image.fromarray(mask).save(str(train_dir / "label" / f"{name}.png"))
        idx += 1


class TestRunTier1LevirCd:
    def test_r4_dual_resolution_doubles_rows(self, tmp_path):
        input_dir = tmp_path / "input"
        _make_levir_input(input_dir, n_low=5, n_high=5)

        output_dir = tmp_path / "output"
        stats = run_tier1_levir_cd(input_dir, output_dir, sample_fraction=1.0)
        assert stats["processed"] == 10

        with open(output_dir / "levir_cd.jsonl") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) == 20  # native + proxy
        assert any("CARTOSAT-proxy" in l["gsd_bucket"] for l in lines)

    def test_stratified_sampling_keeps_both_magnitudes(self, tmp_path):
        """A low sample_fraction must not wipe out the low-change stratum —
        that's exactly what plain random.sample over a high-change-majority
        pool would risk."""
        input_dir = tmp_path / "input"
        _make_levir_input(input_dir, n_low=20, n_high=2)

        output_dir = tmp_path / "output"
        run_tier1_levir_cd(input_dir, output_dir, sample_fraction=0.2, seed=1)

        with open(output_dir / "levir_cd.jsonl") as f:
            lines = [json.loads(l) for l in f if l.strip()]

        names = {Path(l["image_path"][0]).stem.replace("levir_", "").replace("_before", "")
                 for l in lines}
        assert any(n.startswith("high") for n in names), \
            "High-change stratum (2 items) was dropped by subsampling"

    def test_all_rows_validate(self, tmp_path):
        input_dir = tmp_path / "input"
        _make_levir_input(input_dir, n_low=3, n_high=2)
        output_dir = tmp_path / "output"
        run_tier1_levir_cd(input_dir, output_dir, sample_fraction=1.0)

        from preprocess.validator import validate_sample
        with open(output_dir / "levir_cd.jsonl") as f:
            for line in f:
                if not line.strip():
                    continue
                ok, errs = validate_sample(json.loads(line))
                assert ok, errs


class TestLevirSchema:
    def test_has_all_fields(self, tmp_path):
        from PIL import Image
        import numpy as np

        img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        img.save(str(tmp_path / "before.png"))
        img.save(str(tmp_path / "after.png"))

        pair = {
            "base_name": "test",
            "before": tmp_path / "before.png",
            "after": tmp_path / "after.png",
            "mask": None,
        }
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        sample = process_pair(pair, output_dir)
        required = ["id", "dataset", "task", "image_path", "pair_type",
                     "gsd_bucket", "split", "instruction", "response",
                     "bbox", "modality"]
        for field in required:
            assert field in sample, f"Missing field: {field}"
