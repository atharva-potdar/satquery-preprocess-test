"""Tests for preprocess.tier0_bigen — BigEarthNet preprocessing."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from preprocess.tier0_bigen import (
    load_bigen_arrays,
    parse_bigen_sample,
    run_tier0_bigen,
    s2_to_rgb_uint8,
)


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


# ---------------------------------------------------------------------------
# load_bigen_arrays / run_tier0_bigen — the actual wired pipeline
# ---------------------------------------------------------------------------


def _make_bigen_input(root: Path, n_patches: int = 3, with_sar: bool = False) -> None:
    """Build a synthetic Kaggle-style BigEarthNet input dir."""
    import pandas as pd

    (root / "s2_npy").mkdir(parents=True, exist_ok=True)
    if with_sar:
        (root / "s1_npy").mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(n_patches):
        patch_id = f"patch_{i:04d}"
        s2 = np.random.randint(100, 8000, size=(12, 32, 32), dtype=np.uint16)
        np.save(root / "s2_npy" / f"{patch_id}.npy", s2)

        if with_sar:
            s1 = (np.random.rand(2, 32, 32).astype(np.float32) * 0.1)
            np.save(root / "s1_npy" / f"{patch_id}.npy", s1)

        rows.append({"patch_id": patch_id, "labels": [0, 11]})

    pd.DataFrame(rows).to_parquet(root / "metadata.parquet")


class TestLoadBigenArrays:
    def test_loads_s2_only(self, tmp_path):
        _make_bigen_input(tmp_path, n_patches=1, with_sar=False)
        s2, s1 = load_bigen_arrays(tmp_path, {"patch_id": "patch_0000"})
        assert s2.shape == (12, 32, 32)
        assert s1 is None

    def test_loads_s1_when_present(self, tmp_path):
        _make_bigen_input(tmp_path, n_patches=1, with_sar=True)
        s2, s1 = load_bigen_arrays(tmp_path, {"patch_id": "patch_0000"})
        assert s1 is not None
        assert s1.shape == (2, 32, 32)

    def test_missing_patch_raises(self, tmp_path):
        _make_bigen_input(tmp_path, n_patches=1, with_sar=False)
        with pytest.raises(FileNotFoundError):
            load_bigen_arrays(tmp_path, {"patch_id": "does_not_exist"})

    def test_missing_patch_id_raises(self, tmp_path):
        with pytest.raises(ValueError):
            load_bigen_arrays(tmp_path, {})


class TestS2ToRgbUint8:
    def test_shape_and_dtype(self):
        image = np.random.randint(0, 8000, size=(12, 16, 16), dtype=np.uint16)
        rgb = s2_to_rgb_uint8(image)
        assert rgb.shape == (16, 16, 3)
        assert rgb.dtype == np.uint8


class TestRunTier0Bigen:
    def test_optical_only_pipeline_produces_valid_samples(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _make_bigen_input(input_dir, n_patches=3, with_sar=False)

        output_dir = tmp_path / "output"
        stats = run_tier0_bigen(input_dir, output_dir, max_samples=10)

        assert stats["processed"] == 3
        assert stats["failed"] == 0

        jsonl_path = output_dir / "bigen.jsonl"
        assert jsonl_path.exists()
        with open(jsonl_path) as f:
            lines = [json.loads(l) for l in f if l.strip()]

        assert len(lines) == 3  # 1 row/patch when no SAR
        for row in lines:
            assert row["modality"] == "optical"
            assert Path(row["image_path"][0]).exists()

            from preprocess.validator import validate_sample
            ok, errs = validate_sample(row)
            assert ok, f"Invalid sample {row['id']}: {errs}"

    def test_sar_and_fusion_products_emitted(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _make_bigen_input(input_dir, n_patches=2, with_sar=True)

        output_dir = tmp_path / "output"
        stats = run_tier0_bigen(input_dir, output_dir, max_samples=10)

        assert stats["processed"] == 2

        with open(output_dir / "bigen.jsonl") as f:
            lines = [json.loads(l) for l in f if l.strip()]

        # 3 rows/patch when SAR is present: optical, sar, fusion
        assert len(lines) == 6

        modalities = {l["modality"] for l in lines}
        assert modalities == {"optical", "sar", "optical+sar"}

        pair_types = {l["pair_type"] for l in lines}
        assert "cross-modal" in pair_types

        fusion_rows = [l for l in lines if l["pair_type"] == "cross-modal"]
        for row in fusion_rows:
            assert row["task"] == "fusion_vqa"
            assert len(row["image_path"]) == 2
            for p in row["image_path"]:
                assert Path(p).exists()

            from preprocess.validator import validate_sample
            ok, errs = validate_sample(row)
            assert ok, f"Invalid fusion sample {row['id']}: {errs}"

    def test_second_run_skips_all(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _make_bigen_input(input_dir, n_patches=2, with_sar=False)

        output_dir = tmp_path / "output"
        stats1 = run_tier0_bigen(input_dir, output_dir, max_samples=10)
        assert stats1["processed"] == 2

        stats2 = run_tier0_bigen(input_dir, output_dir, max_samples=10)
        assert stats2["processed"] == 0
        assert stats2["skipped"] == 2

    def test_missing_metadata_returns_zero(self, tmp_path):
        input_dir = tmp_path / "empty_input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        stats = run_tier0_bigen(input_dir, output_dir, max_samples=10)
        assert stats == {"processed": 0, "skipped": 0, "failed": 0}

    def test_missing_npy_patch_counted_as_failed(self, tmp_path):
        import pandas as pd

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "s2_npy").mkdir()
        # metadata references a patch with no corresponding .npy file
        pd.DataFrame([{"patch_id": "ghost", "labels": [0]}]).to_parquet(
            input_dir / "metadata.parquet"
        )

        output_dir = tmp_path / "output"
        stats = run_tier0_bigen(input_dir, output_dir, max_samples=10)
        assert stats["failed"] == 1
        assert stats["processed"] == 0
