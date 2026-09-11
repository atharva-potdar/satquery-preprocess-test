"""Tests for preprocess.tier2_sn6_sar — SpaceNet 6 SAR preprocessing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from preprocess.tier2_sn6_sar import process_sar_tile


class TestProcessSarTile:
    def test_returns_valid_sample(self, tmp_path):
        import numpy as np
        import tifffile

        # Create a 2D SAR tile
        img = np.random.randint(0, 1000, (64, 64), dtype=np.uint32)
        tile_path = tmp_path / "tile001.tif"
        tifffile.imwrite(str(tile_path), img)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        sample = process_sar_tile(tile_path, output_dir, "tile001")
        assert sample is not None
        assert sample["id"] == "sn6_sar_tile001"
        assert sample["dataset"] == "sn6_sar"
        assert sample["modality"] == "sar"

    def test_3d_sar_tile(self, tmp_path):
        import numpy as np
        import tifffile

        # Create a 3D SAR tile (multi-pol)
        img = np.random.randint(0, 1000, (3, 64, 64), dtype=np.uint32)
        tile_path = tmp_path / "tile002.tif"
        tifffile.imwrite(str(tile_path), img)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        sample = process_sar_tile(tile_path, output_dir, "tile002")
        assert sample is not None
        assert sample["modality"] == "sar"


class TestSn6SarSchema:
    def test_has_all_fields(self, tmp_path):
        import numpy as np
        import tifffile

        img = np.zeros((64, 64), dtype=np.uint32)
        tile_path = tmp_path / "tile.tif"
        tifffile.imwrite(str(tile_path), img)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        sample = process_sar_tile(tile_path, output_dir, "tile")
        required = ["id", "dataset", "task", "image_path", "pair_type",
                     "gsd_bucket", "split", "instruction", "response",
                     "bbox", "modality"]
        for field in required:
            assert field in sample, f"Missing field: {field}"
