"""Tests for preprocess.tier3_sen2lulc — Sen-2 LULC preprocessing."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from preprocess.tier3_sen2lulc import load_metadata, mask_to_bbox, parse_sen2lulc_sample


class TestLoadMetadata:
    def test_loads_csv(self, tmp_path):
        csv_path = tmp_path / "metadata.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image_id", "label"])
            writer.writeheader()
            writer.writerow({"image_id": "img001", "label": "0"})
            writer.writerow({"image_id": "img002", "label": "3"})

        entries = load_metadata(tmp_path)
        assert len(entries) == 2

    def test_loads_json(self, tmp_path):
        data = [{"image_id": "img001", "label": 1}]
        (tmp_path / "annotations.json").write_text(json.dumps(data))
        entries = load_metadata(tmp_path)
        assert len(entries) == 1

    def test_empty_dir(self, tmp_path):
        entries = load_metadata(tmp_path)
        assert entries == []


class TestMaskToBbox:
    def test_finds_class_regions(self, tmp_path):
        import numpy as np
        from PIL import Image

        # Create mask with class 0 (Built-up) in top-left, class 2 (Water) in bottom-right
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[0:32, 0:32] = 0  # Built-up
        mask[32:64, 32:64] = 2  # Water

        mask_path = tmp_path / "mask.png"
        Image.fromarray(mask).save(str(mask_path))

        bboxes = mask_to_bbox(mask_path)
        assert bboxes is not None
        assert len(bboxes) == 2

    def test_empty_mask_returns_none(self, tmp_path):
        import numpy as np
        from PIL import Image

        mask = np.full((64, 64), 255, dtype=np.uint8)  # Class 255 doesn't exist
        mask_path = tmp_path / "mask.png"
        Image.fromarray(mask).save(str(mask_path))

        bboxes = mask_to_bbox(mask_path)
        assert bboxes is None


class TestParseSen2LulcSample:
    def test_valid_sample(self):
        entry = {"image_id": "img001", "label": 0}
        result = parse_sen2lulc_sample(entry, None, None, "sen2lulc_img001")
        assert result is not None
        assert result["id"] == "sen2lulc_img001"
        assert result["dataset"] == "sen2lulc"
        assert result["response"] == "Built-up"

    def test_string_label(self):
        entry = {"image_id": "img001", "label": "Water"}
        result = parse_sen2lulc_sample(entry, None, None, "sen2lulc_img001")
        assert result["response"] == "Water"

    def test_missing_image_id(self):
        entry = {"label": 0}
        result = parse_sen2lulc_sample(entry, None, None, "sen2lulc_001")
        assert result is None

    def test_has_all_fields(self):
        entry = {"image_id": "img001", "label": 1}
        result = parse_sen2lulc_sample(entry, None, None, "sen2lulc_img001")
        required = ["id", "dataset", "task", "image_path", "pair_type",
                     "gsd_bucket", "split", "instruction", "response",
                     "bbox", "modality"]
        for field in required:
            assert field in result, f"Missing field: {field}"
