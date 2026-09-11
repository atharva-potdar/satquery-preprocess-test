"""Unit tests for preprocess/validator.py.

Tests all 7 hand-crafted fixtures from tests/fixtures/samples.json:
    #1 — Valid single-image VQA (should pass)
    #2 — Valid single-image grounding (should pass)
    #3 — Valid bitemporal change_vqa (should pass)
    #4 — Valid cross-modal fusion_grounding (should pass)
    #5 — INVALID: missing modality field (schema rejection)
    #6 — INVALID: bbox value 1500 exceeds [0,1000] (schema rejection)
    #7 — INVALID: pair_type='single' but image_path has 2 entries (cross-field)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from preprocess.validator import cross_field_checks, validate_jsonl, validate_sample

FIXTURES_DIR = Path(__file__).parent / "fixtures"
FIXTURES_JSON = FIXTURES_DIR / "samples.json"


def _load_fixtures() -> list[dict]:
    with open(FIXTURES_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def samples() -> list[dict]:
    return _load_fixtures()


# ---- Schema-level validation ----


class TestValidSamples:
    """Samples #1–#4 must pass schema + cross-field validation."""

    @pytest.fixture(autouse=True)
    def _load(self, samples):
        self.valid_vqa = samples[0]           # #1
        self.valid_grounding = samples[1]     # #2
        self.valid_change_vqa = samples[2]    # #3
        self.valid_fusion = samples[3]        # #4

    def test_valid_single_vqa(self):
        ok, errs = validate_sample(self.valid_vqa)
        assert ok, f"Expected valid, got errors: {errs}"

    def test_valid_single_grounding(self):
        ok, errs = validate_sample(self.valid_grounding)
        assert ok, f"Expected valid, got errors: {errs}"

    def test_valid_bitemporal_change_vqa(self):
        ok, errs = validate_sample(self.valid_change_vqa)
        assert ok, f"Expected valid, got errors: {errs}"

    def test_valid_cross_modal_fusion_grounding(self):
        ok, errs = validate_sample(self.valid_fusion)
        assert ok, f"Expected valid, got errors: {errs}"

    def test_all_four_valid_no_errors(self):
        for i, sample in enumerate([self.valid_vqa, self.valid_grounding,
                                     self.valid_change_vqa, self.valid_fusion]):
            ok, errs = validate_sample(sample)
            assert ok, f"Sample #{i+1} unexpected errors: {errs}"


class TestInvalidSamples:
    """Samples #5–#7 must fail with specific errors."""

    @pytest.fixture(autouse=True)
    def _load(self, samples):
        self.missing_modality = samples[4]   # #5
        self.bbox_overflow = samples[5]      # #6
        self.pair_image_mismatch = samples[6]  # #7

    def test_missing_modality(self):
        ok, errs = validate_sample(self.missing_modality)
        assert not ok, "Expected invalid (missing modality)"
        assert any("'modality'" in e or "modality" in e.lower() for e in errs), \
            f"Expected modality-related error, got: {errs}"

    def test_bbox_overflow(self):
        ok, errs = validate_sample(self.bbox_overflow)
        assert not ok, "Expected invalid (bbox value 1500)"
        assert any("1500" in e or "1000" in e for e in errs), \
            f"Expected bbox range error, got: {errs}"

    def test_pair_image_mismatch(self):
        ok, errs = validate_sample(self.pair_image_mismatch)
        assert not ok, "Expected invalid (pair_type != len(image_path))"
        assert any("image" in e.lower() and ("pair_type" in e or "single" in e)
                    for e in errs), \
            f"Expected pair/image mismatch error, got: {errs}"


# ---- Cross-field checks (isolated) ----


class TestCrossFieldChecks:
    """Direct tests of cross_field_checks beyond schema validation."""

    def test_change_grounding_valid_bitemporal_two_images(self):
        """Positive: change_grounding + bitemporal + 2 images should pass."""
        sample = {
            "id": "x", "dataset": "cdvqa", "task": "change_grounding",
            "image_path": ["a.png", "b.png"], "pair_type": "bitemporal",
            "gsd_bucket": "[GSD:2m]", "split": "train",
            "instruction": "Where?", "response": "Here.",
            "bbox": [[0, 0, 500, 500]], "modality": "optical",
        }
        errs = cross_field_checks(sample)
        assert errs == [], f"Expected no errors, got: {errs}"

    def test_change_grounding_requires_bitemporal(self):
        sample = {
            "id": "x", "dataset": "cdvqa", "task": "change_grounding",
            "image_path": ["a.png", "b.png"], "pair_type": "cross-modal",
            "gsd_bucket": "[GSD:2m]", "split": "train",
            "instruction": "Where?", "response": "Here.",
            "bbox": [[0, 0, 500, 500]], "modality": "optical+sar",
        }
        errs = cross_field_checks(sample)
        assert any("bitemporal" in e for e in errs), \
            f"Expected bitemporal requirement error, got: {errs}"

    def test_change_grounding_requires_two_images(self):
        sample = {
            "id": "x", "dataset": "cdvqa", "task": "change_grounding",
            "image_path": ["a.png"], "pair_type": "bitemporal",
            "gsd_bucket": "[GSD:2m]", "split": "train",
            "instruction": "Where?", "response": "Here.",
            "bbox": [[0, 0, 500, 500]], "modality": "optical",
        }
        errs = cross_field_checks(sample)
        assert any("2 images" in e for e in errs), \
            f"Expected 2-images requirement error, got: {errs}"

    def test_fusion_grounding_requires_cross_modal(self):
        sample = {
            "id": "x", "dataset": "sn6_sar", "task": "fusion_grounding",
            "image_path": ["a.tif", "b.tif"], "pair_type": "bitemporal",
            "gsd_bucket": "[GSD:0.5m]", "split": "train",
            "instruction": "Where?", "response": "Here.",
            "bbox": [[0, 0, 500, 500]], "modality": "optical+sar",
        }
        errs = cross_field_checks(sample)
        assert any("cross-modal" in e for e in errs), \
            f"Expected cross-modal requirement error, got: {errs}"

    def test_cross_modal_requires_optical_sar(self):
        sample = {
            "id": "x", "dataset": "sn6_sar", "task": "fusion_vqa",
            "image_path": ["a.tif", "b.tif"], "pair_type": "cross-modal",
            "gsd_bucket": "[GSD:0.5m]", "split": "train",
            "instruction": "What?", "response": "This.",
            "bbox": None, "modality": "optical",
        }
        errs = cross_field_checks(sample)
        assert any("optical+sar" in e for e in errs), \
            f"Expected optical+sar requirement error, got: {errs}"

    def test_grounding_with_empty_bbox_list_rejected(self):
        """Regression: bbox=[] (empty, not null) used to slip past the
        None-only check on a grounding task."""
        sample = {
            "id": "x", "dataset": "vrsbench", "task": "grounding",
            "image_path": ["a.png"], "pair_type": "single",
            "gsd_bucket": "VHR-native", "split": "train",
            "instruction": "Where?", "response": "Here.",
            "bbox": [], "modality": "optical",
        }
        errs = cross_field_checks(sample)
        assert any("non-empty bbox" in e for e in errs), \
            f"Expected empty-bbox rejection, got: {errs}"

    def test_single_with_two_images(self):
        sample = {
            "id": "x", "dataset": "cdvqa", "task": "vqa",
            "image_path": ["a.png", "b.png"], "pair_type": "single",
            "gsd_bucket": "[GSD:2m]", "split": "train",
            "instruction": "What?", "response": "This.",
            "bbox": None, "modality": "optical",
        }
        errs = cross_field_checks(sample)
        assert any("single" in e and "1 image" in e for e in errs), \
            f"Expected single/1-image error, got: {errs}"


# ---- validate_jsonl (file-level) ----


class TestValidateJsonl:
    def test_valid_jsonl(self):
        """Write 4 valid samples to a temp file and validate."""
        samples = _load_fixtures()[:4]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
            tmp_path = f.name

        count, errs = validate_jsonl(tmp_path)
        assert count == 4
        assert errs == [], f"Unexpected errors: {errs}"

        Path(tmp_path).unlink()

    def test_mixed_valid_invalid(self):
        """Write 2 valid + 1 invalid, expect errors only for invalid."""
        samples = _load_fixtures()
        rows = [samples[0], samples[4], samples[1]]  # valid, invalid, valid
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            for s in rows:
                f.write(json.dumps(s) + "\n")
            tmp_path = f.name

        count, errs = validate_jsonl(tmp_path)
        assert count == 3
        assert len(errs) > 0, "Expected errors for invalid sample"
        assert any("line 2" in e for e in errs), \
            f"Expected error on line 2, got: {errs}"

        Path(tmp_path).unlink()

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            tmp_path = f.name

        count, errs = validate_jsonl(tmp_path)
        assert count == 0
        assert errs == []

        Path(tmp_path).unlink()


# ---- Edge cases ----


class TestEdgeCases:
    def test_empty_string_instruction_rejected(self):
        sample = {
            "id": "x", "dataset": "bigen", "task": "vqa",
            "image_path": ["a.png"], "pair_type": "single",
            "gsd_bucket": "[GSD:10m]", "split": "train",
            "instruction": "", "response": "Answer.",
            "bbox": None, "modality": "optical",
        }
        ok, errs = validate_sample(sample)
        assert not ok
        assert any("instruction" in e for e in errs)

    def test_bbox_at_boundaries_valid(self):
        sample = {
            "id": "x", "dataset": "vrsbench", "task": "grounding",
            "image_path": ["a.png"], "pair_type": "single",
            "gsd_bucket": "VHR-native", "split": "train",
            "instruction": "Locate.", "response": "Here.",
            "bbox": [[0, 0, 1000, 1000]], "modality": "optical",
        }
        ok, errs = validate_sample(sample)
        assert ok, f"Boundary bbox [0,0,1000,1000] should be valid, got: {errs}"

    def test_negative_bbox_rejected(self):
        sample = {
            "id": "x", "dataset": "vrsbench", "task": "grounding",
            "image_path": ["a.png"], "pair_type": "single",
            "gsd_bucket": "VHR-native", "split": "train",
            "instruction": "Locate.", "response": "Here.",
            "bbox": [[-1, 0, 500, 500]], "modality": "optical",
        }
        ok, errs = validate_sample(sample)
        assert not ok
        assert any("-1" in e or "minimum" in e.lower() for e in errs)

    def test_unknown_dataset_rejected(self):
        sample = {
            "id": "x", "dataset": "unknown_ds", "task": "vqa",
            "image_path": ["a.png"], "pair_type": "single",
            "gsd_bucket": "[GSD:10m]", "split": "train",
            "instruction": "Q?", "response": "A.",
            "bbox": None, "modality": "optical",
        }
        ok, errs = validate_sample(sample)
        assert not ok
        assert any("unknown_ds" in e or "dataset" in e for e in errs)

    def test_additional_properties_rejected(self):
        sample = {
            "id": "x", "dataset": "bigen", "task": "vqa",
            "image_path": ["a.png"], "pair_type": "single",
            "gsd_bucket": "[GSD:10m]", "split": "train",
            "instruction": "Q?", "response": "A.",
            "bbox": None, "modality": "optical",
            "extra_field": "should not be here",
        }
        ok, errs = validate_sample(sample)
        assert not ok
        assert any("extra_field" in e or "Additional" in e or "additional" in e
                    for e in errs)
