"""Tests for preprocess.tier0_bigen — BigEarthNet preprocessing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from preprocess.tier0_bigen import parse_bigen_sample


class TestParseBigENetSample:
    def test_valid_sample(self):
        import numpy as np
        sample = {
            "image": np.random.randint(0, 1000, (12, 64, 64), dtype=np.uint16),
            "label": np.zeros(19, dtype=np.int32),
        }
        sample["label"][0] = 1  # Continuous urban fabric
        sample["label"][11] = 1  # Non-irrigated arable land

        result = parse_bigen_sample(sample, idx=0)
        assert result is not None
        assert result["id"] == "bigen_00000000"
        assert result["dataset"] == "bigen"
        assert result["task"] == "vqa"
        assert result["modality"] == "optical"
        assert "Continuous urban fabric" in result["response"]
        assert "Non-irrigated arable land" in result["response"]

    def test_missing_image(self):
        import numpy as np
        sample = {"label": np.ones(19, dtype=np.int32)}
        result = parse_bigen_sample(sample, idx=0)
        assert result is None

    def test_missing_label(self):
        import numpy as np
        sample = {"image": np.random.randint(0, 1000, (12, 64, 64), dtype=np.uint16)}
        result = parse_bigen_sample(sample, idx=0)
        assert result is None

    def test_empty_labels_skipped(self):
        import numpy as np
        sample = {
            "image": np.random.randint(0, 1000, (12, 64, 64), dtype=np.uint16),
            "label": np.zeros(19, dtype=np.int32),
        }
        result = parse_bigen_sample(sample, idx=0)
        assert result is None

    def test_rgb_extraction(self):
        import numpy as np
        # Create image with known values
        image = np.zeros((12, 32, 32), dtype=np.uint16)
        image[3] = 100  # B04 (R)
        image[2] = 200  # B03 (G)
        image[1] = 300  # B02 (B)

        sample = {
            "image": image,
            "label": np.array([1] + [0]*18, dtype=np.int32),
        }
        result = parse_bigen_sample(sample, idx=5)
        assert result is not None
        assert result["id"] == "bigen_00000005"

    def test_sample_id_format(self):
        import numpy as np
        sample = {
            "image": np.random.randint(0, 1000, (12, 64, 64), dtype=np.uint16),
            "label": np.array([1] + [0]*18, dtype=np.int32),
        }
        result = parse_bigen_sample(sample, idx=12345)
        assert result["id"] == "bigen_00012345"


class TestBigENetSchema:
    def test_sample_has_all_fields(self):
        import numpy as np
        sample = {
            "image": np.random.randint(0, 1000, (12, 64, 64), dtype=np.uint16),
            "label": np.array([1] + [0]*18, dtype=np.int32),
        }
        result = parse_bigen_sample(sample, idx=0)
        required = ["id", "dataset", "task", "image_path", "pair_type",
                     "gsd_bucket", "split", "instruction", "response",
                     "bbox", "modality"]
        for field in required:
            assert field in result, f"Missing field: {field}"

    def test_dataset_enum(self):
        import numpy as np
        sample = {
            "image": np.random.randint(0, 1000, (12, 64, 64), dtype=np.uint16),
            "label": np.array([1] + [0]*18, dtype=np.int32),
        }
        result = parse_bigen_sample(sample, idx=0)
        assert result["dataset"] == "bigen"
