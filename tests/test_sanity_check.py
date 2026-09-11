"""Tests for preprocess.sanity_check — post-download verification."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from preprocess.sanity_check import (
    SanityCheckError,
    check_cdvqa,
    check_levir_cd,
    check_oscd,
    check_rsvqa_hr,
    check_sardet,
    check_vrsbench,
    run_sanity_check,
)


class TestCheckOscd:
    def test_valid_structure(self, tmp_path):
        # Create OSCD structure
        images_dir = tmp_path / "images"
        for i in range(24):
            loc = images_dir / f"location_{i:02d}"
            (loc / "imgs_1").mkdir(parents=True)
            (loc / "imgs_2").mkdir(parents=True)

        labels_dir = tmp_path / "labels"
        for i in range(24):
            (labels_dir / f"location_{i:02d}" / "cm").mkdir(parents=True)

        results = check_oscd(tmp_path)
        assert results["status"] == "pass"

    def test_missing_images_dir(self, tmp_path):
        with pytest.raises(SanityCheckError):
            check_oscd(tmp_path)


class TestCheckVrsbench:
    def test_valid_structure(self, tmp_path):
        # Create VRSBench structure
        data = [
            {"conversations": [{"value": "<image>\n[caption] Describe this"}]},
            {"conversations": [{"value": "<image>\n[refer] Where is <b>?</b>"}]},
            {"conversations": [{"value": "<image>\n[vqa] What is this?"}]},
        ]
        (tmp_path / "VRSBench_train.json").write_text(json.dumps(data))
        (tmp_path / "Images_train.zip").touch()

        results = check_vrsbench(tmp_path)
        assert results["status"] == "pass"

    def test_missing_json(self, tmp_path):
        with pytest.raises(SanityCheckError):
            check_vrsbench(tmp_path)


class TestCheckCdvqa:
    def test_valid_structure(self, tmp_path):
        train_data = [{"question": "Q", "answer": "A"}] * 1600
        val_data = [{"question": "Q", "answer": "A"}] * 400

        (tmp_path / "train.json").write_text(json.dumps(train_data))
        (tmp_path / "val.json").write_text(json.dumps(val_data))

        results = check_cdvqa(tmp_path)
        assert results["status"] == "pass"

    def test_rejects_test_split(self, tmp_path):
        # Should warn but not fail
        (tmp_path / "train.json").write_text(json.dumps([]))
        (tmp_path / "test.json").write_text(json.dumps([]))

        results = check_cdvqa(tmp_path)
        assert results["status"] == "pass"


class TestCheckLevirCd:
    def test_valid_structure(self, tmp_path):
        for split in ["train", "val", "test"]:
            (tmp_path / split / "A").mkdir(parents=True)
            (tmp_path / split / "B").mkdir(parents=True)
            (tmp_path / split / "label").mkdir(parents=True)

            # Add some files
            for i in range(10):
                (tmp_path / split / "A" / f"img{i:03d}_1.png").touch()
                (tmp_path / split / "B" / f"img{i:03d}_2.png").touch()
                (tmp_path / split / "label" / f"img{i:03d}.png").touch()

        results = check_levir_cd(tmp_path)
        assert results["status"] == "pass"


class TestCheckSardet:
    def test_valid_structure(self, tmp_path):
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()

        for i in range(50):
            (images_dir / f"img{i:04d}.png").touch()
            (labels_dir / f"img{i:04d}.txt").touch()

        results = check_sardet(tmp_path)
        assert results["status"] == "pass"


class TestCheckRsvqaHr:
    def test_valid_structure(self, tmp_path):
        train_data = [{"question_id": i, "question": "Q", "answer": "A"} for i in range(100)]
        (tmp_path / "train.json").write_text(json.dumps(train_data))

        images_dir = tmp_path / "images"
        images_dir.mkdir()
        for i in range(100):
            (images_dir / f"img{i:04d}.png").touch()

        results = check_rsvqa_hr(tmp_path)
        assert results["status"] == "pass"


class TestRunSanityCheck:
    def test_valid_dataset(self, tmp_path):
        # Create minimal OSCD structure
        images_dir = tmp_path / "images"
        for i in range(24):
            loc = images_dir / f"loc_{i:02d}"
            (loc / "imgs_1").mkdir(parents=True)
            (loc / "imgs_2").mkdir(parents=True)

        results = run_sanity_check("oscd", tmp_path)
        assert results["status"] == "pass"

    def test_unknown_dataset(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            run_sanity_check("nonexistent", Path("/tmp"))
