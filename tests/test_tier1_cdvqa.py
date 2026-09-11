"""Tests for preprocess.tier1_cdvqa — CDVQA preprocessing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from preprocess.tier1_cdvqa import load_split, parse_cdvqa_sample


class TestLoadSplit:
    def test_loads_train(self, tmp_path):
        data = [{"question": "What changed?", "answer": "building"}]
        (tmp_path / "train.json").write_text(json.dumps(data))
        result = load_split(tmp_path, "train")
        assert len(result) == 1

    def test_loads_val(self, tmp_path):
        data = [{"question": "What changed?", "answer": "road"}]
        (tmp_path / "val.json").write_text(json.dumps(data))
        result = load_split(tmp_path, "val")
        assert len(result) == 1

    def test_rejects_test_split(self, tmp_path):
        with pytest.raises(ValueError, match="Invalid split"):
            load_split(tmp_path, "test")

    def test_empty_when_missing(self, tmp_path):
        result = load_split(tmp_path, "train")
        assert result == []

    def test_wrapped_format(self, tmp_path):
        data = {"questions": [{"question": "Q", "answer": "A"}]}
        (tmp_path / "train.json").write_text(json.dumps(data))
        result = load_split(tmp_path, "train")
        assert len(result) == 1


class TestParseCdvqaSample:
    def test_valid_sample(self, tmp_path):
        entry = {
            "question": "What changed?",
            "answer": "new building",
            "before": "before.png",
            "after": "after.png",
        }
        result = parse_cdvqa_sample(entry, tmp_path, "cdvqa_0")
        assert result is not None
        assert result["id"] == "cdvqa_0"
        assert result["dataset"] == "cdvqa"
        assert result["task"] == "change_vqa"
        assert result["pair_type"] == "bitemporal"

    def test_empty_question_skipped(self):
        entry = {"question": "", "answer": "A", "before": "b.png", "after": "a.png"}
        result = parse_cdvqa_sample(entry, None, "cdvqa_0")
        assert result is None

    def test_missing_before_skipped(self):
        entry = {"question": "Q", "answer": "A", "before": "", "after": "a.png"}
        result = parse_cdvqa_sample(entry, None, "cdvqa_0")
        assert result is None

    def test_image_paths_resolved(self, tmp_path):
        (tmp_path / "before.png").write_bytes(b"\x89PNG")
        (tmp_path / "after.png").write_bytes(b"\x89PNG")
        entry = {
            "question": "Q",
            "answer": "A",
            "before": "before.png",
            "after": "after.png",
        }
        result = parse_cdvqa_sample(entry, tmp_path, "cdvqa_0")
        assert len(result["image_path"]) == 2
        assert "before.png" in result["image_path"][0]
        assert "after.png" in result["image_path"][1]

    def test_has_all_fields(self):
        entry = {"question": "Q", "answer": "A", "before": "b.png", "after": "a.png"}
        result = parse_cdvqa_sample(entry, None, "cdvqa_0")
        required = ["id", "dataset", "task", "image_path", "pair_type",
                     "gsd_bucket", "split", "instruction", "response",
                     "bbox", "modality"]
        for field in required:
            assert field in result, f"Missing field: {field}"

    def test_modality_optical(self):
        entry = {"question": "Q", "answer": "A", "before": "b.png", "after": "a.png"}
        result = parse_cdvqa_sample(entry, None, "cdvqa_0")
        assert result["modality"] == "optical"
