"""Tests for preprocess.tier2_sn6_sar — SpaceNet 6 SAR preprocessing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image

from preprocess.tier2_sn6_sar import (
    load_optical_selection,
    process_sar_tile,
    run_tier2_sn6_sar,
)


class TestProcessSarTile:
    def test_returns_valid_sample(self, tmp_path):
        # Create a 2D SAR tile
        img = np.random.randint(0, 1000, (64, 64), dtype=np.uint32)
        tile_path = tmp_path / "tile001.tif"
        tifffile.imwrite(str(tile_path), img)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        samples = process_sar_tile(tile_path, output_dir, "tile001")
        assert len(samples) == 1
        sample = samples[0]
        assert sample["id"] == "sn6_sar_tile001"
        assert sample["dataset"] == "sn6_sar"
        assert sample["modality"] == "sar"

    def test_3d_sar_tile(self, tmp_path):
        # Create a 3D SAR tile (multi-pol)
        img = np.random.randint(0, 1000, (3, 64, 64), dtype=np.uint32)
        tile_path = tmp_path / "tile002.tif"
        tifffile.imwrite(str(tile_path), img)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        samples = process_sar_tile(tile_path, output_dir, "tile002")
        assert len(samples) == 1
        assert samples[0]["modality"] == "sar"

    def test_fusion_sample_emitted_when_optical_given(self, tmp_path):
        img = np.random.randint(0, 1000, (64, 64), dtype=np.uint32)
        tile_path = tmp_path / "tile003.tif"
        tifffile.imwrite(str(tile_path), img)

        optical_png = tmp_path / "optical.png"
        Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(str(optical_png))

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        samples = process_sar_tile(tile_path, output_dir, "tile003", optical_png=optical_png)
        assert len(samples) == 2
        fusion = [s for s in samples if s["pair_type"] == "cross-modal"][0]
        assert fusion["task"] == "fusion_vqa"
        assert fusion["modality"] == "optical+sar"
        assert len(fusion["image_path"]) == 2


class TestLoadOpticalSelection:
    def test_reads_selection_and_paths(self, tmp_path):
        opt_dir = tmp_path / "opt_output"
        opt_dir.mkdir()
        (opt_dir / "selected_tile_ids.json").write_text(json.dumps(["tile001", "tile002"]))

        rows = [
            {"id": "sn6_tile001", "image_path": ["/x/tile001.png"]},
            {"id": "sn6_tile001_proxy", "image_path": ["/x/tile001_proxy.png"]},
            {"id": "sn6_tile002", "image_path": ["/x/tile002.png"]},
        ]
        with open(opt_dir / "sn6_opt.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        ids, by_tile = load_optical_selection(opt_dir)
        assert ids == {"tile001", "tile002"}
        assert by_tile == {"tile001": "/x/tile001.png", "tile002": "/x/tile002.png"}

    def test_missing_files_returns_empty(self, tmp_path):
        ids, by_tile = load_optical_selection(tmp_path / "nonexistent")
        assert ids == set()
        assert by_tile == {}


class TestRunTier2Sn6Sar:
    def _make_sar_input(self, root: Path, tile_ids: list[str]) -> None:
        sar_dir = root / "train" / "sar"
        sar_dir.mkdir(parents=True)
        for tid in tile_ids:
            img = np.random.randint(0, 1000, (32, 32), dtype=np.uint32)
            tifffile.imwrite(str(sar_dir / f"{tid}.tif"), img)

    def test_paired_selection_restricts_to_optical_tiles(self, tmp_path):
        sar_input = tmp_path / "sar_input"
        self._make_sar_input(sar_input, ["t1", "t2", "t3", "t4"])

        opt_dir = tmp_path / "opt_output"
        opt_dir.mkdir()
        (opt_dir / "selected_tile_ids.json").write_text(json.dumps(["t1", "t3"]))
        Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(str(opt_dir / "t1.png"))
        rows = [{"id": "sn6_t1", "image_path": [str(opt_dir / "t1.png")]}]
        with open(opt_dir / "sn6_opt.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        output_dir = tmp_path / "output"
        stats = run_tier2_sn6_sar(sar_input, output_dir, optical_dir=opt_dir)
        assert stats["processed"] == 2  # only t1, t3 — not t2/t4

    def test_fusion_row_emitted_for_paired_tile(self, tmp_path):
        sar_input = tmp_path / "sar_input"
        self._make_sar_input(sar_input, ["t1"])

        opt_dir = tmp_path / "opt_output"
        opt_dir.mkdir()
        (opt_dir / "selected_tile_ids.json").write_text(json.dumps(["t1"]))
        Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(str(opt_dir / "t1.png"))
        with open(opt_dir / "sn6_opt.jsonl", "w") as f:
            f.write(json.dumps({"id": "sn6_t1", "image_path": [str(opt_dir / "t1.png")]}) + "\n")

        output_dir = tmp_path / "output"
        run_tier2_sn6_sar(sar_input, output_dir, optical_dir=opt_dir)

        with open(output_dir / "sn6_sar.jsonl") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert any(l["pair_type"] == "cross-modal" for l in lines)

    def test_all_rows_validate(self, tmp_path):
        sar_input = tmp_path / "sar_input"
        self._make_sar_input(sar_input, ["t1", "t2"])
        output_dir = tmp_path / "output"
        run_tier2_sn6_sar(sar_input, output_dir, sample_fraction=1.0)

        from preprocess.validator import validate_sample
        with open(output_dir / "sn6_sar.jsonl") as f:
            for line in f:
                if not line.strip():
                    continue
                ok, errs = validate_sample(json.loads(line))
                assert ok, errs


class TestSn6SarSchema:
    def test_has_all_fields(self, tmp_path):
        img = np.zeros((64, 64), dtype=np.uint32)
        tile_path = tmp_path / "tile.tif"
        tifffile.imwrite(str(tile_path), img)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        samples = process_sar_tile(tile_path, output_dir, "tile")
        sample = samples[0]
        required = ["id", "dataset", "task", "image_path", "pair_type",
                     "gsd_bucket", "split", "instruction", "response",
                     "bbox", "modality"]
        for field in required:
            assert field in sample, f"Missing field: {field}"
