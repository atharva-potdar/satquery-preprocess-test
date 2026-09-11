"""Tests for preprocess.tier1_sn6_opt — SpaceNet 6 Optical preprocessing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from preprocess.tier1_sn6_opt import load_geojson_labels, polygons_to_bboxes, process_tile


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
