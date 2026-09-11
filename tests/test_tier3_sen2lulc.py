"""Tests for preprocess.tier3_sen2lulc — Sen-2 LULC preprocessing."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from preprocess.tier3_sen2lulc import (
    load_metadata,
    mask_to_bbox,
    parse_sen2lulc_sample,
    run_tier3_sen2lulc,
)


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


def _make_sen2lulc_input(root: Path, class_counts: dict[int, int]) -> None:
    images_dir = root / "images"
    images_dir.mkdir(parents=True)
    rows = []
    idx = 0
    for label, n in class_counts.items():
        for _ in range(n):
            image_id = f"img{idx:04d}"
            Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(
                str(images_dir / f"{image_id}.png"))
            rows.append({"image_id": image_id, "label": label})
            idx += 1

    import csv as csv_mod
    with open(root / "metadata.csv", "w", newline="") as f:
        writer = csv_mod.DictWriter(f, fieldnames=["image_id", "label"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


class TestRunTier3Sen2lulc:
    def test_sample_fraction_actually_subsamples(self, tmp_path):
        """Regression: sample_fraction used to be a dead parameter — the
        full annotation set got processed regardless, risking a resource
        blowout against Kaggle's 20GB /kaggle/working cap."""
        input_dir = tmp_path / "input"
        _make_sen2lulc_input(input_dir, {0: 100})

        output_dir = tmp_path / "output"
        stats = run_tier3_sen2lulc(input_dir, output_dir, sample_fraction=0.1, seed=1)
        assert 5 <= stats["processed"] <= 20  # ~10% of 100, not all 100

    def test_stratified_sampling_keeps_rare_class(self, tmp_path):
        input_dir = tmp_path / "input"
        _make_sen2lulc_input(input_dir, {0: 50, 2: 2})  # class 2 = Water, rare

        output_dir = tmp_path / "output"
        run_tier3_sen2lulc(input_dir, output_dir, sample_fraction=0.2, seed=1)

        with open(output_dir / "sen2lulc.jsonl") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert any(l["response"] == "Water" for l in lines), \
            "Rare class (Water, n=2) was dropped by subsampling"

    def test_all_rows_validate(self, tmp_path):
        input_dir = tmp_path / "input"
        _make_sen2lulc_input(input_dir, {0: 5, 1: 5})
        output_dir = tmp_path / "output"
        run_tier3_sen2lulc(input_dir, output_dir, sample_fraction=1.0)

        from preprocess.validator import validate_sample
        with open(output_dir / "sen2lulc.jsonl") as f:
            for line in f:
                if not line.strip():
                    continue
                ok, errs = validate_sample(json.loads(line))
                assert ok, errs
