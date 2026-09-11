"""R6 — Bounding box format conversion.

Qwen convention: normalized [0,1000], ordered
(x_topleft, y_topleft), (x_bottomright, y_bottomright).  Not y-first.

Public API:
    pixel_to_normalized(bbox, img_w, img_h) -> [[x1,y1,x2,y2]]
    normalized_to_pixel(bbox, img_w, img_h) -> (x1, y1, x2, y2)
    normalized_to_latlon(bbox, transform, crs) -> [[lon1,lat1],[lon2,lat2]]
    validate_bbox_ordering(bbox) -> bool
"""

from __future__ import annotations

from typing import Any

import numpy as np


def unit01_to_qwen(bbox_01: list[float]) -> list[list[float]]:
    """Convert 0-1 normalized bbox to Qwen [0,1000] format.

    Parameters
    ----------
    bbox_01 : [x1, y1, x2, y2] in 0-1 normalized coordinates.

    Returns
    -------
    [[x1, y1, x2, y2]] in [0,1000] Qwen order.
    """
    x1, y1, x2, y2 = bbox_01
    return [[x1 * 1000, y1 * 1000, x2 * 1000, y2 * 1000]]


def pixel_to_normalized(
    bbox: tuple[int, int, int, int],
    img_width: int,
    img_height: int,
) -> list[list[float]]:
    """Convert pixel-space bbox (x1,y1,x2,y2) to [0,1000] normalized.

    Parameters
    ----------
    bbox : (x_topleft, y_topleft, x_bottomright, y_bottomright) in pixels.
    img_width, img_height : dimensions of the source image in pixels.

    Returns
    -------
    [[x1_norm, y1_norm, x2_norm, y2_norm]] in [0,1000] Qwen order.
    """
    x1, y1, x2, y2 = bbox
    return [[
        round(x1 / img_width * 1000, 2),
        round(y1 / img_height * 1000, 2),
        round(x2 / img_width * 1000, 2),
        round(y2 / img_height * 1000, 2),
    ]]


def normalized_to_pixel(
    bbox: list[list[float]],
    img_width: int,
    img_height: int,
) -> tuple[int, int, int, int]:
    """Convert [0,1000] normalized bbox back to pixel coordinates.

    Returns (x_topleft, y_topleft, x_bottomright, y_bottomright) as ints.
    """
    x1n, y1n, x2n, y2n = bbox[0]
    return (
        int(round(x1n / 1000 * img_width)),
        int(round(y1n / 1000 * img_height)),
        int(round(x2n / 1000 * img_width)),
        int(round(y2n / 1000 * img_height)),
    )


def normalized_to_latlon(
    bbox: list[list[float]],
    transform: Any,
    crs: Any,
) -> list[list[float]]:
    """Convert [0,1000] normalized bbox to (lon, lat) via rasterio transform.

    Parameters
    ----------
    bbox : [[x1n, y1n, x2n, y2n]] in [0,1000].
    transform : rasterio.Affine (pixel → CRS coordinates).
    crs : rasterio CRS object.

    Returns
    -------
    [[lon1, lat1], [lon2, lat2]] (lon/lat if CRS is geographic,
    projected coords otherwise).
    """
    try:
        from rasterio.transform import xy as rio_xy
    except ImportError:
        raise ImportError("rasterio is required for normalized_to_latlon()")

    x1n, y1n, x2n, y2n = bbox[0]

    # Convert [0,1000] → pixel coords (float, not rounded — we want precise projection)
    # The [0,1000] range maps to [0, img_width) and [0, img_height) conceptually.
    # Since we don't have img dimensions here, we treat 1000 as the full extent.
    px1 = x1n / 1000
    py1 = y1n / 1000
    px2 = x2n / 1000
    py2 = y2n / 1000

    # rasterio xy gives (x, y) in CRS units for (col, row)
    lon1, lat1 = rio_xy(transform, px1, py1)
    lon2, lat2 = rio_xy(transform, px2, py2)

    return [[lon1, lat1], [lon2, lat2]]


def validate_bbox_ordering(bbox: list[list[float]]) -> bool:
    """Verify topleft/bottomright ordering (R6 convention).

    Returns True if x1 <= x2 and y1 <= y2 (valid ordering).
    Returns False if the box is degenerate/reversed.
    """
    x1, y1, x2, y2 = bbox[0]
    return x1 <= x2 and y1 <= y2
