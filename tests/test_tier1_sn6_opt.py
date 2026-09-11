"""Tests for preprocess.tier1_sn6_opt — SpaceNet 6 Optical preprocessing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from preprocess.tier1_sn6_opt import (
    load_geojson_labels,
    polygons_to_bboxes,
    process_tile,
    run_tier1_sn6_opt,
)


class TestLoadGeojsonLabels:
    def test_loads_polygons(self, tmp_path):
        geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [100, 0], [100, 100], [0, 100], [0, 0]]],
                },
            }],
        }
        path = tmp_path / "labels.geojson"
        path.write_text(json.dumps(geojson))
        polygons = load_geojson_labels(path)
        assert len(polygons) == 1
        assert len(polygons[0]) == 5

    def test_loads_multipolygon(self, tmp_path):
        geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {
                    "type": "MultiPolygon",
                    "coordinates": [
                        [[[0, 0], [50, 0], [50, 50], [0, 50], [0, 0]]],
                        [[[60, 60], [100, 60], [100, 100], [60, 100], [60, 60]]],
                    ],
                },
            }],
        }
        path = tmp_path / "labels.geojson"
        path.write_text(json.dumps(geojson))
        polygons = load_geojson_labels(path)
        assert len(polygons) == 2

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.geojson"
        path.write_text("{}")
        polygons = load_geojson_labels(path)
        assert polygons == []

    def test_missing_file(self):
        polygons = load_geojson_labels(Path("/nonexistent.geojson"))
        assert polygons == []


class TestPolygonsToBboxes:
    def test_converts_polygon(self):
        polygon = [[0, 0], [100, 0], [100, 100], [0, 100], [0, 0]]
        bboxes = polygons_to_bboxes([polygon], 200, 200)
        assert len(bboxes) == 1
        assert len(bboxes[0]) == 4
        # Should be normalized to [0, 1000]
        assert bboxes[0][0] >= 0
        assert bboxes[0][2] <= 1000

    def test_small_polygon_included(self):
        polygon = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]
        bboxes = polygons_to_bboxes([polygon], 1000, 1000)
        assert len(bboxes) == 1


class TestProcessTile:
    def test_returns_valid_sample(self, tmp_path):
        import numpy as np
        import tifffile

        # Create a test TIF
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        tif_path = tmp_path / "tile001.tif"
        tifffile.imwrite(str(tif_path), img)

        # Create empty geojson
        geojson = {"type": "FeatureCollection", "features": []}
        geojson_path = tmp_path / "tile001.geojson"
        geojson_path.write_text(json.dumps(geojson))

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        sample = process_tile(tif_path, geojson_path, output_dir, "tile001")
        assert sample is not None
        assert sample["id"] == "sn6_tile001"
        assert sample["dataset"] == "sn6_opt"


    def test_bbox_count_matches_response_count(self, tmp_path):
        """Response text must describe exactly the boxes returned, not a
        total that only the first box represents."""
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        tif_path = tmp_path / "tile002.tif"
        tifffile.imwrite(str(tif_path), img)

        def _poly(x0, y0, x1, y1):
            return [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]

        geojson = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [_poly(0, 0, 10, 10)]}},
                {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [_poly(20, 20, 30, 30)]}},
                {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [_poly(40, 40, 50, 50)]}},
            ],
        }
        geojson_path = tmp_path / "tile002.geojson"
        geojson_path.write_text(json.dumps(geojson))

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        sample = process_tile(tif_path, geojson_path, output_dir, "tile002")
        assert sample is not None
        assert sample["bbox"] is not None
        assert len(sample["bbox"]) == 3
        assert "3 buildings detected" in sample["response"]


def _make_sn6_input(root: Path, tiles: list[int]) -> None:
    """tiles: list of building counts, one tile per entry."""
    images_dir = root / "train" / "images"
    labels_dir = root / "train" / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    for i, n_buildings in enumerate(tiles):
        tile_id = f"tile{i:04d}"
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        tifffile.imwrite(str(images_dir / f"{tile_id}.tif"), img)

        features = []
        for b in range(n_buildings):
            x0 = (b * 2) % 30
            y0 = (b * 3) % 30
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[x0, y0], [x0 + 1, y0], [x0 + 1, y0 + 1], [x0, y0 + 1], [x0, y0]]],
                },
            })
        geojson = {"type": "FeatureCollection", "features": features}
        (labels_dir / f"{tile_id}.geojson").write_text(json.dumps(geojson))


class TestRunTier1Sn6Opt:
    def test_r4_dual_resolution_doubles_rows(self, tmp_path):
        input_dir = tmp_path / "input"
        _make_sn6_input(input_dir, tiles=[2, 5, 0])

        output_dir = tmp_path / "output"
        stats = run_tier1_sn6_opt(input_dir, output_dir, sample_fraction=1.0)
        assert stats["processed"] == 3

        with open(output_dir / "sn6_opt.jsonl") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) == 6
        assert any("CARTOSAT-proxy" in l["gsd_bucket"] for l in lines)

    def test_selected_tile_ids_written(self, tmp_path):
        input_dir = tmp_path / "input"
        _make_sn6_input(input_dir, tiles=[1, 2, 3])
        output_dir = tmp_path / "output"
        run_tier1_sn6_opt(input_dir, output_dir, sample_fraction=1.0)

        selected_path = output_dir / "selected_tile_ids.json"
        assert selected_path.exists()
        ids = json.loads(selected_path.read_text())
        assert set(ids) == {"tile0000", "tile0001", "tile0002"}

    def test_stratified_sampling_keeps_sparse_and_dense(self, tmp_path):
        input_dir = tmp_path / "input"
        # Mostly dense tiles, a couple of empty (sparse) ones.
        _make_sn6_input(input_dir, tiles=[10] * 10 + [0, 0])

        output_dir = tmp_path / "output"
        run_tier1_sn6_opt(input_dir, output_dir, sample_fraction=0.3, seed=1)

        selected = json.loads((output_dir / "selected_tile_ids.json").read_text())
        counts = {"tile0010": 0, "tile0011": 0}  # the 2 sparse (0-building) tiles
        assert any(t in selected for t in counts), \
            "Sparse-density stratum was dropped by subsampling"


class TestSn6OptSchema:
    def test_has_all_fields(self, tmp_path):
        import numpy as np
        import tifffile

        img = np.zeros((32, 32, 3), dtype=np.uint8)
        tif_path = tmp_path / "tile.tif"
        tifffile.imwrite(str(tif_path), img)

        geojson = {"type": "FeatureCollection", "features": []}
        geojson_path = tmp_path / "tile.geojson"
        geojson_path.write_text(json.dumps(geojson))

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        sample = process_tile(tif_path, geojson_path, output_dir, "tile")
        required = ["id", "dataset", "task", "image_path", "pair_type",
                     "gsd_bucket", "split", "instruction", "response",
                     "bbox", "modality"]
        for field in required:
            assert field in sample, f"Missing field: {field}"
