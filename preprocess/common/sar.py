"""R3 — SAR pseudo-RGB generation and R5 — SAR GSD downsampling.

R3 pipeline:
    1. Linear → dB:  dB = 10·log10(x + ε)
    2. Clip:  VV ∈ [-25, 0] dB,  VH ∈ [-30, -5] dB
    3. Map (dual-pol):  R = VV(dB),  G = VH(dB),  B = (VV − VH) dB post-clip
    4. Map (quad-pol SpaceNet6):  R = HH,  G = VV,  B = VH

R5 SAR GSD:
    SpaceNet6 SAR (0.5m) and SARDet-100K finer patches are downsampled
    into the RISAT proxy band (~2–10 m).  BigEarthNet S1 (10 m) is the
    coarse anchor and is left as-is.

Public API:
    linear_to_db(arr, epsilon)
    clip_vv_db(arr) / clip_vh_db(arr)
    sar_pseudo_rgb(vv, vh, hh)
    downsample_sar(arr, target_gsd, native_gsd)
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# R3 — dB conversion and clipping
# ---------------------------------------------------------------------------

def linear_to_db(arr: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """Convert linear-scale SAR amplitude/power to decibels.

    Parameters
    ----------
    arr : numpy array (any shape), linear scale.
    epsilon : small constant to avoid log10(0).

    Returns
    -------
    numpy array in dB (same shape).

    Notes
    -----
    np.clip(arr, epsilon, None) clamps the lower bound to epsilon.
    This handles zero AND negative values (from floating-point noise
    in upstream denoising or VV-VH math) — negatives become ~-100 dB
    rather than producing NaN.  This is intentional: SAR linear power
    should never be negative, so any negative values are artifacts.
    """
    return 10.0 * np.log10(np.clip(arr, epsilon, None))


def clip_vv_db(arr: np.ndarray) -> np.ndarray:
    """Clip VV polarisation to [-25, 0] dB (R3 spec)."""
    return np.clip(arr, -25.0, 0.0)


def clip_vh_db(arr: np.ndarray) -> np.ndarray:
    """Clip VH polarisation to [-30, -5] dB (R3 spec)."""
    return np.clip(arr, -30.0, -5.0)


# ---------------------------------------------------------------------------
# R3 — Pseudo-RGB construction
# ---------------------------------------------------------------------------

def sar_pseudo_rgb(
    vv: np.ndarray,
    vh: np.ndarray,
    hh: np.ndarray | None = None,
) -> np.ndarray:
    """Build a 3-channel SAR pseudo-RGB image.

    Dual-pol (Sentinel-1, RISAT):
        R = VV(dB)  clipped to [-25, 0]
        G = VH(dB)  clipped to [-30, -5]
        B = (VV − VH) in dB, post-clipping

    Quad-pol (SpaceNet6 SAR):
        R = HH(dB)
        G = VV(dB)
        B = VH(dB)

    Parameters
    ----------
    vv, vh : 2-D numpy arrays in linear scale.
    hh : 2-D numpy array in linear scale (quad-pol only).  If None,
         dual-pol mapping is used.

    Returns
    -------
    (H, W, 3) uint8 array in [0, 255], suitable for PNG output.
    """
    vv_db = linear_to_db(vv)
    vh_db = linear_to_db(vh)

    if hh is not None:
        # Quad-pol: R=HH, G=VV, B=VH  (R3 rule 4)
        hh_db = linear_to_db(hh)
        r = _db_to_uint8(hh_db, vmin=-25, vmax=0)
        g = _db_to_uint8(vv_db, vmin=-25, vmax=0)
        b = _db_to_uint8(vh_db, vmin=-30, vmax=-5)
    else:
        # Dual-pol: R=VV, G=VH, B=VV−VH  (R3 rules 2–3)
        # CRITICAL: clip FIRST, then subtract.  R3 says "post-clipping"
        # meaning: clip VV to [-25,0] dB and VH to [-30,-5] dB FIRST,
        # then compute B = VV_clipped - VH_clipped.  This is NOT the same
        # as computing raw VV-VH and clipping that result.
        vv_clipped = clip_vv_db(vv_db)
        vh_clipped = clip_vh_db(vh_db)
        diff_db = vv_clipped - vh_clipped

        r = _db_to_uint8(vv_clipped, vmin=-25, vmax=0)
        g = _db_to_uint8(vh_clipped, vmin=-30, vmax=-5)
        b = _db_to_uint8(diff_db, vmin=0, vmax=25)

    return np.stack([r, g, b], axis=-1)


def sar_intensity_pseudo_gray(intensity: np.ndarray) -> np.ndarray:
    """Render a single-channel SAR intensity image as an honest grayscale-in-dB PNG.

    Some sources (SARDet-100K's shipped PNGs, single-band fallback tiles)
    give us amplitude/intensity only — no separate VV/VH. R3's dual-pol
    mapping needs two channels; fabricating a second one (e.g. VH = VV*0.8)
    invents a physically meaningless, constant ratio real backscatter
    never has. Converting the one real channel to dB and replicating it
    across R/G/B is the honest "pseudo-RGB" for single-pol data — visually
    a grayscale SAR render, same R3 dB pipeline, no invented signal.
    """
    db = linear_to_db(intensity)
    db_clipped = clip_vv_db(db)  # reuse VV's [-25, 0] range as the general SAR floor
    gray = _db_to_uint8(db_clipped, vmin=-25, vmax=0)
    return np.stack([gray, gray, gray], axis=-1)


def _db_to_uint8(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Linearly map a dB-range array to uint8 [0, 255]."""
    normalized = (arr - vmin) / (vmax - vmin)
    normalized = np.clip(normalized, 0.0, 1.0)
    return (normalized * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# R5 — SAR GSD downsampling
# ---------------------------------------------------------------------------

def downsample_sar(
    arr: np.ndarray,
    target_gsd: float,
    native_gsd: float,
) -> np.ndarray:
    """Downsample a SAR image from native GSD to target GSD.

    Uses area interpolation (averaging) to preserve radiometric integrity.
    If native_gsd <= target_gsd (already coarser or equal), returns as-is.

    Parameters
    ----------
    arr : (H, W) or (H, W, C) numpy array.
    target_gsd : desired ground sampling distance in metres.
    native_gsd : current GSD in metres.

    Returns
    -------
    Downsampled array (or original if no downsampling needed).
    """
    if native_gsd >= target_gsd:
        return arr  # Already at or coarser than target

    scale = native_gsd / target_gsd  # < 1.0
    new_h = max(1, int(round(arr.shape[0] * scale)))
    new_w = max(1, int(round(arr.shape[1] * scale)))

    if arr.ndim == 2:
        return _resize_2d(arr, new_h, new_w)
    else:
        channels = []
        for c in range(arr.shape[2]):
            channels.append(_resize_2d(arr[:, :, c], new_h, new_w))
        return np.stack(channels, axis=-1)


def _resize_2d(arr: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """Resize a 2D array using area averaging (no scipy dependency)."""
    from PIL import Image

    img = Image.fromarray(arr.astype(np.float32) if arr.dtype != np.float32 else arr)
    resized = img.resize((new_w, new_h), resample=Image.Resampling.BOX)
    return np.array(resized, dtype=arr.dtype)
