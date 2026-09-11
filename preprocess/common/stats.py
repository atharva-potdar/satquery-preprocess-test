"""R2 — Two-pass global percentile computation for radiometric normalization.

Percentile clipping (2nd/98th) is computed ONCE globally per sensor-and-band,
not per-image.  This module provides an accumulator that processes images in
two passes: first to gather statistics, then to apply clipping.

Public API:
    GlobalPercentileStats  — accumulator class
    clip_percentile(arr, low, high) -> np.ndarray
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


class GlobalPercentileStats:
    """Accumulate per-band statistics across many images for global percentiles.

    Usage (two-pass):
        stats = GlobalPercentileStats()

        # Pass 1: accumulate from each image
        for img_path in image_paths:
            arr = load_image(img_path)  # (H, W, C) or (H, W)
            stats.accumulate(arr, bands=["B4", "B3", "B2"])

        # Compute global percentiles
        percentiles = stats.compute_percentiles(low=2, high=98)

        # Pass 2: apply clipping
        for img_path in image_paths:
            arr = load_image(img_path)
            clipped = stats.apply_clip(arr, bands=["B4", "B3", "B2"])

    The accumulator stores per-band min, max, and a reservoir sample for
    percentile estimation.  For large datasets, the reservoir is capped at
    ``max_samples`` per band to bound memory usage.
    """

    def __init__(self, max_samples: int = 50_000):
        self.max_samples = max_samples
        self._samples: dict[str, list[np.ndarray]] = defaultdict(list)
        self._counts: dict[str, int] = defaultdict(int)

    def accumulate(
        self,
        arr: np.ndarray,
        bands: list[str] | None = None,
    ) -> None:
        """Accumulate statistics from a single image array.

        Parameters
        ----------
        arr : (H, W, C) or (H, W) numpy array.
        bands : list of band names, one per channel.  If arr is 2D,
                 pass a single-element list like ["VV"].
        """
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]
            if bands is None:
                bands = ["band_0"]
        if bands is None:
            bands = [f"band_{i}" for i in range(arr.shape[2])]

        for c, band_name in enumerate(bands):
            channel = arr[:, :, c].astype(np.float64).ravel()

            # Reservoir sampling: cap memory at max_samples
            n_new = len(channel)
            current_count = self._counts[band_name]

            if current_count >= self.max_samples:
                # Already full — skip
                continue

            available = self.max_samples - current_count
            if n_new <= available:
                self._samples[band_name].append(channel)
            else:
                # Take a random subsample to fill the reservoir
                indices = np.random.choice(n_new, size=available, replace=False)
                self._samples[band_name].append(channel[indices])

            self._counts[band_name] = current_count + n_new

    def compute_percentiles(
        self,
        low: float = 2.0,
        high: float = 98.0,
    ) -> dict[str, tuple[float, float]]:
        """Compute global (low, high) percentiles for each accumulated band.

        Returns dict mapping band_name → (low_val, high_val).
        """
        result: dict[str, tuple[float, float]] = {}
        for band_name, sample_list in self._samples.items():
            if not sample_list:
                raise ValueError(f"No data accumulated for band '{band_name}'")
            combined = np.concatenate(sample_list)
            low_val = float(np.percentile(combined, low))
            high_val = float(np.percentile(combined, high))
            result[band_name] = (low_val, high_val)
        return result

    def apply_clip(
        self,
        arr: np.ndarray,
        percentiles: dict[str, tuple[float, float]],
        bands: list[str] | None = None,
    ) -> np.ndarray:
        """Apply percentile clipping to an image array.

        Parameters
        ----------
        arr : (H, W, C) or (H, W) numpy array.
        percentiles : dict from compute_percentiles().  This MUST be the
                       output of compute_percentiles() — it is NOT
                       recomputed here.  R2 requires a single, stable,
                       global stat per sensor-band; recomputing would
                       reintroduce per-image inconsistency.
        bands : band names matching the channels in arr.

        Returns
        -------
        Clipped array (same shape, dtype preserved where possible).
        """
        out = arr.copy()
        if out.ndim == 2:
            out = out[:, :, np.newaxis]
            if bands is None:
                bands = ["band_0"]

        if bands is None:
            bands = [f"band_{i}" for i in range(out.shape[2])]

        for c, band_name in enumerate(bands):
            if band_name not in percentiles:
                continue
            low_val, high_val = percentiles[band_name]
            out[:, :, c] = np.clip(out[:, :, c], low_val, high_val)

        # Squeeze back if input was 2D
        if arr.ndim == 2:
            out = out[:, :, 0]
        return out

    def save(self, path: str | Path) -> None:
        """Persist accumulator state to disk (numpy compressed)."""
        np.savez_compressed(
            str(path),
            samples={k: np.concatenate(v) for k, v in self._samples.items()},
            counts=dict(self._counts),
            max_samples=self.max_samples,
        )

    @classmethod
    def load(cls, path: str | Path) -> GlobalPercentileStats:
        """Restore accumulator state from disk."""
        data = np.load(str(path), allow_pickle=True)
        obj = cls(max_samples=int(data["max_samples"]))
        obj._counts = defaultdict(int, data["counts"].item())
        for band_name, samples in data["samples"].item().items():
            obj._samples[band_name] = [samples]
        return obj


def clip_percentile(
    arr: np.ndarray,
    low: float,
    high: float,
) -> tuple[np.ndarray, float, float]:
    """Quick single-image percentile clip (for small datasets / testing).

    Returns (clipped_array, low_val, high_val).
    """
    low_val = float(np.percentile(arr, low))
    high_val = float(np.percentile(arr, high))
    return np.clip(arr, low_val, high_val), low_val, high_val
