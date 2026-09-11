"""Tests for preprocess/common/ — bbox, stats, sar, gsd modules."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from preprocess.common.bbox import (
    normalized_to_latlon,
    normalized_to_pixel,
    pixel_to_normalized,
    validate_bbox_ordering,
)
from preprocess.common.gsd import (
    CARTOSAT_PROXY,
    DUAL_RESOLUTION_DATASETS,
    RISAT_PROXY,
    assign_gsd_bucket,
    create_proxy_sample,
    curriculum_mix_ratio,
)
from preprocess.common.sar import (
    clip_vh_db,
    clip_vv_db,
    downsample_sar,
    linear_to_db,
    sar_pseudo_rgb,
)
from preprocess.common.stats import GlobalPercentileStats, clip_percentile


# =========================================================================
# bbox.py tests
# =========================================================================


class TestPixelToNormalized:
    def test_origin(self):
        result = pixel_to_normalized((0, 0, 100, 100), 1000, 1000)
        assert result == [[0.0, 0.0, 100.0, 100.0]]

    def test_full_image(self):
        result = pixel_to_normalized((0, 0, 640, 480), 640, 480)
        assert result == [[0.0, 0.0, 1000.0, 1000.0]]

    def test_half_image(self):
        result = pixel_to_normalized((160, 120, 480, 360), 640, 480)
        expected = [[250.0, 250.0, 750.0, 750.0]]
        assert result == expected

    def test_roundtrip(self):
        original = (100, 50, 400, 300)
        norm = pixel_to_normalized(original, 800, 600)
        back = normalized_to_pixel(norm, 800, 600)
        assert back == original


class TestNormalizedToPixel:
    def test_full_extent(self):
        result = normalized_to_pixel([[0, 0, 1000, 1000]], 640, 480)
        assert result == (0, 0, 640, 480)

    def test_center_box(self):
        result = normalized_to_pixel([[250, 250, 750, 750]], 400, 400)
        assert result == (100, 100, 300, 300)


class TestValidateBboxOrdering:
    def test_valid_topleft_first(self):
        assert validate_bbox_ordering([[100, 200, 500, 600]]) is True

    def test_equal_coordinates(self):
        assert validate_bbox_ordering([[100, 100, 100, 100]]) is True

    def test_reversed_x(self):
        assert validate_bbox_ordering([[500, 100, 100, 500]]) is False

    def test_reversed_y(self):
        assert validate_bbox_ordering([[100, 500, 500, 100]]) is False


class TestNormalizedToLatlon:
    def test_identity_transform(self):
        """With identity affine, latlon should equal pixel coords."""
        from rasterio.transform import Affine

        transform = Affine(1, 0, 0, 0, -1, 100)
        bbox = [[0.0, 0.0, 1000.0, 1000.0]]
        result = normalized_to_latlon(bbox, transform, None)
        assert len(result) == 2
        assert len(result[0]) == 2  # [lon, lat]


# =========================================================================
# stats.py tests
# =========================================================================


class TestGlobalPercentileStats:
    def test_single_image(self):
        stats = GlobalPercentileStats(max_samples=1000)
        arr = np.random.rand(64, 64, 3).astype(np.float32) * 100
        stats.accumulate(arr, bands=["R", "G", "B"])

        pcts = stats.compute_percentiles(low=2, high=98)
        assert set(pcts.keys()) == {"R", "G", "B"}
        for band in ["R", "G", "B"]:
            low, high = pcts[band]
            assert low < high
            assert 0 <= low <= 100
            assert 0 <= high <= 100

    def test_multiple_images(self):
        stats = GlobalPercentileStats(max_samples=10000)
        for _ in range(10):
            arr = np.random.rand(32, 32).astype(np.float32) * 50 + 25
            stats.accumulate(arr, bands=["VV"])

        pcts = stats.compute_percentiles(low=10, high=90)
        low, high = pcts["VV"]
        # With uniform-ish data in [25, 75], 10th percentile ~29, 90th ~71
        assert 20 < low < 40
        assert 60 < high < 80

    def test_apply_clip(self):
        stats = GlobalPercentileStats(max_samples=1000)
        arr = np.arange(100, dtype=np.float64).reshape(10, 10)
        stats.accumulate(arr, bands=["B4"])
        pcts = stats.compute_percentiles(low=10, high=90)

        clipped = stats.apply_clip(arr, pcts, bands=["B4"])
        low, high = pcts["B4"]
        assert clipped.min() >= low - 1e-10
        assert clipped.max() <= high + 1e-10

    def test_save_load_roundtrip(self):
        stats = GlobalPercentileStats(max_samples=500)
        arr = np.random.rand(16, 16).astype(np.float32)
        stats.accumulate(arr, bands=["test_band"])

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            tmp_path = f.name
        stats.save(tmp_path)

        loaded = GlobalPercentileStats.load(tmp_path)
        pcts_orig = stats.compute_percentiles(low=5, high=95)
        pcts_loaded = loaded.compute_percentiles(low=5, high=95)

        assert pcts_orig.keys() == pcts_loaded.keys()
        for band in pcts_orig:
            assert abs(pcts_orig[band][0] - pcts_loaded[band][0]) < 1e-6
            assert abs(pcts_orig[band][1] - pcts_loaded[band][1]) < 1e-6

        Path(tmp_path).unlink()


class TestClipPercentile:
    def test_basic_clip(self):
        arr = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        clipped, low, high = clip_percentile(arr, 10, 90)
        assert low < high
        assert clipped.min() >= low - 1e-10
        assert clipped.max() <= high + 1e-10


# =========================================================================
# sar.py tests
# =========================================================================


class TestLinearToDb:
    def test_basic_conversion(self):
        linear = np.array([1.0, 10.0, 100.0])
        db = linear_to_db(linear)
        np.testing.assert_allclose(db, [0.0, 10.0, 20.0], atol=1e-6)

    def test_zero_handled(self):
        linear = np.array([0.0])
        db = linear_to_db(linear, epsilon=1e-10)
        assert db[0] < -90  # ~-100 dB

    def test_epsilon_prevents_log10_error(self):
        linear = np.array([0.0, 0.0, 0.0])
        db = linear_to_db(linear, epsilon=1e-10)
        assert np.all(np.isfinite(db))

    def test_negative_input_clamped_to_epsilon(self):
        """Negative linear values (from float noise) should become ~-100 dB, not NaN."""
        linear = np.array([-0.5, -1.0, -100.0])
        db = linear_to_db(linear, epsilon=1e-10)
        assert np.all(np.isfinite(db)), f"Got NaN/Inf: {db}"
        # 10*log10(1e-10) = -100 dB — this is the de facto "no signal" floor
        expected_floor = 10.0 * np.log10(1e-10)
        np.testing.assert_allclose(db, expected_floor, atol=1e-6,
            err_msg=f"Expected clamped value ~{expected_floor} dB, got {db}")


class TestSarClipping:
    def test_vv_clip_range(self):
        arr = np.array([-30.0, -25.0, -12.5, 0.0, 5.0])
        clipped = clip_vv_db(arr)
        np.testing.assert_array_equal(clipped, [-25.0, -25.0, -12.5, 0.0, 0.0])

    def test_vh_clip_range(self):
        arr = np.array([-35.0, -30.0, -17.5, -5.0, 0.0])
        clipped = clip_vh_db(arr)
        np.testing.assert_array_equal(clipped, [-30.0, -30.0, -17.5, -5.0, -5.0])


class TestSarPseudoRgb:
    def test_dual_pol_shape(self):
        vv = np.random.rand(64, 64).astype(np.float32) * 0.1
        vh = np.random.rand(64, 64).astype(np.float32) * 0.01
        rgb = sar_pseudo_rgb(vv, vh)
        assert rgb.shape == (64, 64, 3)
        assert rgb.dtype == np.uint8

    def test_dual_pol_range(self):
        vv = np.ones((16, 16), dtype=np.float32) * 0.01  # ~-20 dB
        vh = np.ones((16, 16), dtype=np.float32) * 0.001  # ~-30 dB
        rgb = sar_pseudo_rgb(vv, vh)
        assert rgb.min() >= 0
        assert rgb.max() <= 255

    def test_quad_pol_shape(self):
        hh = np.random.rand(64, 64).astype(np.float32) * 0.1
        vv = np.random.rand(64, 64).astype(np.float32) * 0.1
        vh = np.random.rand(64, 64).astype(np.float32) * 0.01
        rgb = sar_pseudo_rgb(vv, vh, hh=hh)
        assert rgb.shape == (64, 64, 3)
        assert rgb.dtype == np.uint8

    def test_identical_inputs_deterministic(self):
        vv = np.ones((8, 8), dtype=np.float32) * 0.05
        vh = np.ones((8, 8), dtype=np.float32) * 0.005
        rgb1 = sar_pseudo_rgb(vv, vh)
        rgb2 = sar_pseudo_rgb(vv, vh)
        np.testing.assert_array_equal(rgb1, rgb2)


class TestDownsampleSar:
    def test_no_downsample_when_already_coarse(self):
        arr = np.random.rand(64, 64).astype(np.float32)
        result = downsample_sar(arr, target_gsd=2.0, native_gsd=5.0)
        assert result.shape == arr.shape

    def test_downsample_2x(self):
        arr = np.random.rand(128, 128).astype(np.float32)
        result = downsample_sar(arr, target_gsd=4.0, native_gsd=2.0)
        # Scale = 2/4 = 0.5, so 128*0.5 = 64
        assert result.shape[0] == 64
        assert result.shape[1] == 64

    def test_downsample_preserves_dtype(self):
        arr = np.random.rand(64, 64).astype(np.uint8) * 255
        result = downsample_sar(arr, target_gsd=4.0, native_gsd=2.0)
        assert result.dtype == np.uint8

    def test_downsample_3d(self):
        arr = np.random.rand(64, 64, 3).astype(np.float32)
        result = downsample_sar(arr, target_gsd=4.0, native_gsd=2.0)
        assert result.ndim == 3
        assert result.shape[2] == 3


# =========================================================================
# gsd.py tests
# =========================================================================


class TestAssignGsdBucket:
    def test_rsvqa_hr_literal(self):
        assert assign_gsd_bucket("rsvqa_hr") == "[GSD:0.15m]"

    def test_sn6_opt_literal(self):
        assert assign_gsd_bucket("sn6_opt") == "[GSD:0.5m]"

    def test_bigen_literal(self):
        assert assign_gsd_bucket("bigen") == "[GSD:10m]"

    def test_sardet_literal(self):
        assert assign_gsd_bucket("sardet") == "[GSD:5m]"

    def test_vrsbench_native_no_gsd(self):
        assert assign_gsd_bucket("vrsbench") == "VHR-native"

    def test_cdvqa_native_no_gsd(self):
        assert assign_gsd_bucket("cdvqa") == "VHR-native"

    def test_unknown_dataset_with_gsd(self):
        result = assign_gsd_bucket("unknown_ds", native_gsd=3.5)
        assert result == "[GSD:3.50m]"

    def test_unknown_dataset_no_gsd(self):
        assert assign_gsd_bucket("unknown_ds") == "VHR-native"

    def test_integer_gsd_format(self):
        result = assign_gsd_bucket("unknown_ds", native_gsd=10.0)
        assert result == "[GSD:10m]"


class TestCreateProxySample:
    def test_optical_proxy(self):
        sample = {
            "id": "vrsbench_001",
            "dataset": "vrsbench",
            "task": "grounding",
            "image_path": ["img.png"],
            "pair_type": "single",
            "gsd_bucket": "VHR-native",
            "split": "train",
            "instruction": "Locate.",
            "response": "Here.",
            "bbox": [[0, 0, 500, 500]],
            "modality": "optical",
        }
        proxy = create_proxy_sample(sample)
        assert proxy["id"] == "vrsbench_001_proxy"
        assert proxy["gsd_bucket"] == "CARTOSAT-proxy[GSD:2m]"
        # Original unchanged
        assert sample["id"] == "vrsbench_001"
        assert sample["gsd_bucket"] == "VHR-native"

    def test_sar_proxy(self):
        sample = {
            "id": "cdvqa_001",
            "dataset": "cdvqa",
            "task": "change_vqa",
            "image_path": ["a.png", "b.png"],
            "pair_type": "bitemporal",
            "gsd_bucket": "VHR-native",
            "split": "train",
            "instruction": "What changed?",
            "response": "New building.",
            "bbox": None,
            "modality": "sar",
        }
        proxy = create_proxy_sample(sample)
        assert "RISAT-proxy" in proxy["gsd_bucket"]

    def test_explicit_proxy_gsd(self):
        sample = {
            "id": "test_001",
            "modality": "optical",
            "gsd_bucket": "VHR-native",
        }
        proxy = create_proxy_sample(sample, proxy_gsd=1.5)
        assert "1.50m" in proxy["gsd_bucket"]


class TestCurriculumMixRatio:
    def test_stage_1_all_native(self):
        assert curriculum_mix_ratio(1) == (1.0, 0.0)

    def test_stage_2_80_20(self):
        assert curriculum_mix_ratio(2) == (0.8, 0.2)

    def test_stage_3_50_50(self):
        assert curriculum_mix_ratio(3) == (0.5, 0.5)

    def test_stage_4_30_70(self):
        assert curriculum_mix_ratio(4) == (0.3, 0.7)

    def test_stage_5_caps_at_4(self):
        assert curriculum_mix_ratio(5) == (0.3, 0.7)

    def test_stage_0_falls_back_to_1(self):
        assert curriculum_mix_ratio(0) == (1.0, 0.0)

    def test_ratios_sum_to_one(self):
        for stage in range(1, 10):
            native, proxy = curriculum_mix_ratio(stage)
            assert abs(native + proxy - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# R6: unit01_to_qwen
# ---------------------------------------------------------------------------


class TestUnit01ToQwen:
    def test_zero_maps_to_zero(self):
        from preprocess.common.bbox import unit01_to_qwen
        result = unit01_to_qwen([0.0, 0.0, 1.0, 1.0])
        assert result == [[0.0, 0.0, 1000.0, 1000.0]]

    def test_half_maps_to_500(self):
        from preprocess.common.bbox import unit01_to_qwen
        result = unit01_to_qwen([0.25, 0.25, 0.75, 0.75])
        assert result == [[250.0, 250.0, 750.0, 750.0]]

    def test_preserves_ordering(self):
        from preprocess.common.bbox import unit01_to_qwen
        result = unit01_to_qwen([0.1, 0.2, 0.3, 0.4])
        assert result[0][0] < result[0][2]
        assert result[0][1] < result[0][3]


# ---------------------------------------------------------------------------
# R7: concat
# ---------------------------------------------------------------------------


class TestConcatHorizontal:
    def test_same_size_images(self):
        from preprocess.common.concat import concat_horizontal
        left = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        right = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = concat_horizontal(left, right)
        assert result.size == (64 + 2 + 64, 64)  # (width, height)
        assert result.mode == "RGB"

    def test_different_sizes_resizes_right(self):
        from preprocess.common.concat import concat_horizontal
        left = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
        right = np.random.randint(0, 255, (50, 40, 3), dtype=np.uint8)
        result = concat_horizontal(left, right)
        assert result.height == 100
        # right (50,40) resized to height 100 → width = 40 * (100/50) = 80
        assert result.width == 80 + 2 + 80

    def test_pil_input(self):
        from preprocess.common.concat import concat_horizontal
        left = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        right = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        result = concat_horizontal(left, right)
        assert result.size == (64 + 2 + 64, 64)

    def test_custom_separator(self):
        from preprocess.common.concat import concat_horizontal
        left = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        right = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = concat_horizontal(left, right, separator_width=10)
        assert result.width == 64 + 10 + 64

    def test_target_height(self):
        from preprocess.common.concat import concat_horizontal
        left = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
        right = np.random.randint(0, 255, (200, 160, 3), dtype=np.uint8)
        result = concat_horizontal(left, right, target_height=50)
        assert result.height == 50

    def test_rejects_2d_input(self):
        from preprocess.common.concat import concat_horizontal
        left = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        right = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        with pytest.raises(ValueError, match="3-channel"):
            concat_horizontal(left, right)

    def test_deterministic(self):
        from preprocess.common.concat import concat_horizontal
        arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        r1 = concat_horizontal(arr, arr)
        r2 = concat_horizontal(arr, arr)
        assert list(r1.getdata()) == list(r2.getdata())


class TestSaveConcat:
    def test_saves_file(self):
        from preprocess.common.concat import save_concat
        with tempfile.TemporaryDirectory() as tmpdir:
            left = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
            right = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
            left_path = Path(tmpdir) / "left.png"
            right_path = Path(tmpdir) / "right.png"
            out_path = Path(tmpdir) / "out" / "concat.png"
            left.save(str(left_path))
            right.save(str(right_path))
            save_concat(left_path, right_path, out_path)
            assert out_path.exists()
            result = Image.open(str(out_path))
            assert result.size == (64 + 2 + 64, 64)
