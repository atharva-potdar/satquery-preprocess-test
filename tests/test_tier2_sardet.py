"""Tests for preprocess.tier2_sardet — SARDet-100K preprocessing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from preprocess.tier2_sardet import load_yolo_labels, process_sardet_sample


class TestLoadYoloLabels:
    def test_loads_labels(self, tmp_path):
        # YOLO format: class_id cx cy w h (normalized)
        label_path = tmp_path / "img001.txt"
        label_path.write_text("0 0.5 0.5 0.1 0.2\n1 0.3 0.3 0.05 0.1\n")

        annotations = load_yolo_labels(label_path, 100, 100)
        assert len(annotations) == 2
        assert annotations[0]["class_name"] == "ship"
        assert annotations[1]["class_name"] == "aircraft"

    def test_empty_file(self, tmp_path):
        label_path = tmp_path / "empty.txt"
        label_path.write_text("")
        annotations = load_yolo_labels(label_path, 100, 100)
        assert annotations == []

    def test_missing_file(self):
        annotations = load_yolo_labels(Path("/nonexistent.txt"), 100, 100)
        assert annotations == []

    def test_invalid_class_id_skipped(self, tmp_path):
        label_path = tmp_path / "img.txt"
        label_path.write_text("99 0.5 0.5 0.1 0.1\n")
        annotations = load_yolo_labels(label_path, 100, 100)
        assert annotations == []


class TestProcessSardetSample:
    def test_returns_valid_sample(self, tmp_path):
        from PIL import Image
        import numpy as np

        # Create SAR-like image
        img = Image.fromarray(np.random.randint(0, 255, (64, 64), dtype=np.uint8))
        img.save(str(tmp_path / "img001.png"))

        # Create labels
        (tmp_path / "img001.txt").write_text("0 0.5 0.5 0.2 0.2\n")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        sample = process_sardet_sample(
            tmp_path / "img001.png",
            tmp_path / "img001.txt",
            output_dir,
            "sardet_001",
        )
        assert sample is not None
        assert sample["id"] == "sardet_001"
        assert sample["dataset"] == "sardet"
        assert sample["modality"] == "sar"

    def test_no_labels_returns_none(self, tmp_path):
        from PIL import Image
        import numpy as np

        img = Image.fromarray(np.zeros((64, 64), dtype=np.uint8))
        img.save(str(tmp_path / "img001.png"))

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        sample = process_sardet_sample(
            tmp_path / "img001.png",
            tmp_path / "nonexistent.txt",
            output_dir,
            "sardet_001",
        )
        assert sample is None


class TestSardetSchema:
    def test_has_all_fields(self, tmp_path):
        from PIL import Image
        import numpy as np

        img = Image.fromarray(np.zeros((64, 64), dtype=np.uint8))
        img.save(str(tmp_path / "img.png"))
        (tmp_path / "img.txt").write_text("0 0.5 0.5 0.2 0.2\n")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        sample = process_sardet_sample(
            tmp_path / "img.png",
            tmp_path / "img.txt",
            output_dir,
            "sardet_001",
        )
        required = ["id", "dataset", "task", "image_path", "pair_type",
                     "gsd_bucket", "split", "instruction", "response",
                     "bbox", "modality"]
        for field in required:
            assert field in sample, f"Missing field: {field}"
