"""Tests for preprocess.tier2_sardet — SARDet-100K preprocessing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from preprocess.tier2_sardet import (
    load_yolo_labels,
    primary_class,
    process_sardet_sample,
    run_tier2_sardet,
)


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


class TestPrimaryClass:
    def test_reads_first_class(self, tmp_path):
        label_path = tmp_path / "img.txt"
        label_path.write_text("2 0.5 0.5 0.1 0.1\n0 0.2 0.2 0.1 0.1\n")
        assert primary_class(label_path) == "bridge"

    def test_missing_file_returns_unknown(self, tmp_path):
        assert primary_class(tmp_path / "nope.txt") == "unknown"


def _make_sardet_input(root: Path, class_counts: dict[str, int]) -> None:
    """class_counts: {category_name: n_images_with_that_primary_class}."""
    images_dir = root / "images"
    labels_dir = root / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    categories = ["ship", "aircraft", "bridge", "tank", "car", "harbor"]
    idx = 0
    for cat, n in class_counts.items():
        class_id = categories.index(cat)
        for _ in range(n):
            name = f"img{idx:04d}"
            Image.fromarray(np.random.randint(0, 255, (32, 32), dtype=np.uint8)).save(
                str(images_dir / f"{name}.png"))
            (labels_dir / f"{name}.txt").write_text(f"{class_id} 0.5 0.5 0.2 0.2\n")
            idx += 1


class TestRunTier2Sardet:
    def test_stratified_sampling_keeps_rare_category(self, tmp_path):
        input_dir = tmp_path / "input"
        _make_sardet_input(input_dir, {"ship": 20, "harbor": 2})

        output_dir = tmp_path / "output"
        run_tier2_sardet(input_dir, output_dir, sample_fraction=0.2, seed=1)

        with open(output_dir / "sardet.jsonl") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        responses = [l["response"] for l in lines]
        assert any("harbor" in r for r in responses), \
            "Rare category (harbor, n=2) was dropped by subsampling"

    def test_all_rows_validate(self, tmp_path):
        input_dir = tmp_path / "input"
        _make_sardet_input(input_dir, {"ship": 3, "car": 2})
        output_dir = tmp_path / "output"
        run_tier2_sardet(input_dir, output_dir, sample_fraction=1.0)

        from preprocess.validator import validate_sample
        with open(output_dir / "sardet.jsonl") as f:
            for line in f:
                if not line.strip():
                    continue
                ok, errs = validate_sample(json.loads(line))
                assert ok, errs


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
