"""Tests for preprocess.tier1_rsvqa_hr — RSVQA-HR preprocessing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from preprocess.tier1_rsvqa_hr import load_annotations, parse_rsvqa_sample


class TestLoadAnnotations:
    def test_loads_train_json(self, tmp_path):
        data = [{"question_id": 1, "question": "What is this?", "answer": "building"}]
        (tmp_path / "train.json").write_text(json.dumps(data))
        result = load_annotations(tmp_path)
        assert len(result) == 1

    def test_loads_questions_json(self, tmp_path):
        data = [{"question_id": 2, "question": "How many?", "answer": "5"}]
        (tmp_path / "questions.json").write_text(json.dumps(data))
        result = load_annotations(tmp_path)
        assert len(result) == 1

    def test_train_takes_priority(self, tmp_path):
        train = [{"question_id": 1}]
        questions = [{"question_id": 2}]
        (tmp_path / "train.json").write_text(json.dumps(train))
        (tmp_path / "questions.json").write_text(json.dumps(questions))
        result = load_annotations(tmp_path)
        assert result[0]["question_id"] == 1

    def test_wrapped_format(self, tmp_path):
        data = {"questions": [{"question_id": 1}]}
        (tmp_path / "train.json").write_text(json.dumps(data))
        result = load_annotations(tmp_path)
        assert len(result) == 1

    def test_empty_dir(self, tmp_path):
        result = load_annotations(tmp_path)
        assert result == []


class TestParseRsvqaSample:
    def test_valid_sample(self, tmp_path):
        entry = {"question_id": 1, "question": "What is this?", "answer": "building"}
        result = parse_rsvqa_sample(entry, None, "rsvqa_1")
        assert result is not None
        assert result["id"] == "rsvqa_1"
        assert result["dataset"] == "rsvqa_hr"
        assert result["task"] == "vqa"
        assert result["instruction"] == "What is this?"
        assert result["response"] == "building"

    def test_empty_question_skipped(self):
        entry = {"question_id": 1, "question": "", "answer": "building"}
        result = parse_rsvqa_sample(entry, None, "rsvqa_1")
        assert result is None

    def test_empty_answer_skipped(self):
        entry = {"question_id": 1, "question": "What?", "answer": ""}
        result = parse_rsvqa_sample(entry, None, "rsvqa_1")
        assert result is None

    def test_image_path_set_when_available(self, tmp_path):
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        (images_dir / "img001.png").write_bytes(b"\x89PNG")

        entry = {"question_id": 1, "question": "What?", "answer": "road", "image_id": "img001"}
        result = parse_rsvqa_sample(entry, images_dir, "rsvqa_1")
        assert result is not None
        assert len(result["image_path"]) == 1
        assert "img001.png" in result["image_path"][0]

    def test_image_path_empty_when_missing(self):
        entry = {"question_id": 1, "question": "What?", "answer": "road", "image_id": "nonexistent"}
        result = parse_rsvqa_sample(entry, Path("/nonexistent"), "rsvqa_1")
        assert result is not None
        assert result["image_path"] == []


class TestRsvqaHrSchema:
    def test_has_all_fields(self):
        entry = {"question_id": 1, "question": "Q", "answer": "A"}
        result = parse_rsvqa_sample(entry, None, "rsvqa_1")
        required = ["id", "dataset", "task", "image_path", "pair_type",
                     "gsd_bucket", "split", "instruction", "response",
                     "bbox", "modality"]
        for field in required:
            assert field in result, f"Missing field: {field}"

    def test_dataset_enum(self):
        entry = {"question_id": 1, "question": "Q", "answer": "A"}
        result = parse_rsvqa_sample(entry, None, "rsvqa_1")
        assert result["dataset"] == "rsvqa_hr"
        assert result["pair_type"] == "single"
        assert result["modality"] == "optical"
