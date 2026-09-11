"""Tests for preprocess.tier1_levir_cd — LEVIR-CD preprocessing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from preprocess.tier1_levir_cd import find_levir_pairs, mask_to_bbox, process_pair


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
