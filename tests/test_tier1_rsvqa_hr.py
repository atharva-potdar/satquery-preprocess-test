"""Tests for preprocess.tier1_rsvqa_hr — RSVQA-HR preprocessing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from preprocess.tier1_rsvqa_hr import load_annotations, parse_rsvqa_sample, run_tier1_rsvqa_hr


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

    def test_jpg_source_converted_to_png(self, tmp_path):
        """R7: all outputs must be PNG — a .jpg source must be converted,
        not referenced directly."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        png_dir = tmp_path / "converted"
        img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
        img.save(str(images_dir / "img002.jpg"), format="JPEG")

        entry = {"question_id": 2, "question": "What?", "answer": "field", "image_id": "img002"}
        result = parse_rsvqa_sample(entry, images_dir, "rsvqa_2", png_dir=png_dir)
        assert result is not None
        assert result["image_path"][0].endswith(".png")
        assert Path(result["image_path"][0]).exists()


class TestRunTier1RsvqaHr:
    def _make_input(self, root: Path, n: int = 4) -> None:
        images_dir = root / "images"
        images_dir.mkdir(parents=True)
        entries = []
        for i in range(n):
            Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(
                str(images_dir / f"img{i:03d}.png")
            )
            entries.append({
                "question_id": i,
                "question": "What is this?",
                "answer": "building",
                "image_id": f"img{i:03d}",
            })
        (root / "train.json").write_text(json.dumps(entries))

    def test_r4_dual_resolution_doubles_rows(self, tmp_path):
        """R4: RSVQA-HR is in DUAL_RESOLUTION_DATASETS — every sample
        with an image must produce a native + CARTOSAT-proxy row."""
        input_dir = tmp_path / "input"
        self._make_input(input_dir)

        output_dir = tmp_path / "output"
        stats = run_tier1_rsvqa_hr(input_dir, output_dir)
        assert stats["processed"] == 4

        with open(output_dir / "rsvqa_hr.jsonl") as f:
            lines = [json.loads(l) for l in f if l.strip()]

        assert len(lines) == 8  # 4 samples x 2 (native + proxy)
        buckets = [l["gsd_bucket"] for l in lines]
        assert "[GSD:0.15m]" in buckets
        assert any("CARTOSAT-proxy" in b for b in buckets)

    def test_all_rows_validate(self, tmp_path):
        input_dir = tmp_path / "input"
        self._make_input(input_dir)
        output_dir = tmp_path / "output"
        run_tier1_rsvqa_hr(input_dir, output_dir)

        from preprocess.validator import validate_sample
        with open(output_dir / "rsvqa_hr.jsonl") as f:
            for line in f:
                if not line.strip():
                    continue
                ok, errs = validate_sample(json.loads(line))
                assert ok, errs


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
