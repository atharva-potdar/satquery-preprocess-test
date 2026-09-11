"""R4/R5 — GSD bucket assignment and dual-resolution curriculum.

Section 5 rules (authoritative):
    Datasets with known, uniform native GSD → literal tags [GSD:Xm]
    Datasets with variable/unknown per-tile GSD → categorical buckets

    Known-uniform datasets:
        RSVQA-HR:           0.15 m   → [GSD:0.15m]
        SpaceNet6 optical:  0.5 m    → [GSD:0.5m]
        LEVIR-CD:           0.5 m    → [GSD:0.5m]
        SpaceNet6 SAR:      0.5 m    → [GSD:0.5m]
        SARDet-100K:        2–10 m   → [GSD:5m] (post-R5 midpoint)
        BigEarthNet:        10 m     → [GSD:10m]
        OSCD:               10 m Sentinel-2, same sensor as BigEarthNet → [GSD:10m]
        Sen-2 LULC:         10 m Sentinel-2 → [GSD:10m]

    Variable/unknown GSD datasets:
        VRSBench            → VHR-native  (native branch)
        VRSBench proxy      → CARTOSAT-proxy
        CDVQA                → SECOND imagery spans ~0.5-2m per scene, no single
                                 native value → VHR-native (no R4 doubling — see
                                 DUAL_RESOLUTION_DATASETS below)

R4 dual-resolution branching (for benchmark datasets only):
    Each training image → two independent samples:
        Native branch  → gsd_bucket per assignment rule
        Proxy branch   → CARTOSAT-proxy (optical ~2m) or RISAT-band (SAR ~2-10m)
    Curriculum mixing across Stage 2:
        80:20 → 50:50 → 30:70  (native:proxy)

R5 SAR GSD calibration:
    Optical VHR → Cartosat-2S (~2m MS)
    SAR → RISAT proxy band (~2–10m)
    SpaceNet6 SAR (0.5m) & SARDet-100K finer → downsampled to RISAT band
    BigEarthNet S1 (10m) → coarse anchor, left as-is

Public API:
    assign_gsd_bucket(dataset, native_gsd=None) -> str
    create_proxy_sample(sample, proxy_gsd=None) -> dict
    curriculum_mix_ratio(stage) -> (float, float)
    DUAL_RESOLUTION_DATASETS — set of datasets that get dual-res branching
"""

from __future__ import annotations

import copy
from typing import Any


# ---------------------------------------------------------------------------
# Known uniform-GSD datasets  (literal tag mapping)
# ---------------------------------------------------------------------------

_KNOWN_GSD: dict[str, str] = {
    "rsvqa_hr":   "[GSD:0.15m]",
    "sn6_opt":    "[GSD:0.5m]",
    "levir_cd":   "[GSD:0.5m]",
    "sn6_sar":    "[GSD:0.5m]",
    "sardet":     "[GSD:5m]",      # midpoint of 2–10m RISAT band
    "bigen":      "[GSD:10m]",     # BigEarthNet S1 anchor
    "oscd":       "[GSD:10m]",     # Sentinel-2, same native GSD as BigEarthNet
    "sen2lulc":   "[GSD:10m]",     # Sentinel-2
}

# Datasets eligible for R4 dual-resolution branching.
# Resolved spec ambiguity (Section 2 R4 prose names CDVQA; the Tier 1
# table's per-row Treatment column is more specific and is authoritative
# here): every Tier 1 row whose Treatment column literally says "R4"
# gets it — vrsbench, rsvqa_hr, levir_cd, sn6_opt. CDVQA's row says
# "native res kept" (no R4) and OSCD's row doesn't mention R4 either;
# both have fixed/known native resolutions already in the VHR range,
# so doubling them buys no resolution diversity.
DUAL_RESOLUTION_DATASETS: set[str] = {
    "vrsbench",
    "rsvqa_hr",
    "levir_cd",
    "sn6_opt",
}

# Proxy bucket names
CARTOSAT_PROXY = "CARTOSAT-proxy"
RISAT_PROXY = "RISAT-proxy"

# Default proxy GSD (Cartosat-2S ~2m)
DEFAULT_OPTICAL_PROXY_GSD = 2.0
# Default SAR proxy GSD (RISAT band midpoint ~5m)
DEFAULT_SAR_PROXY_GSD = 5.0


# ---------------------------------------------------------------------------
# GSD bucket assignment
# ---------------------------------------------------------------------------

def assign_gsd_bucket(
    dataset: str,
    native_gsd: float | None = None,
) -> str:
    """Assign the gsd_bucket string for a sample.

    Parameters
    ----------
    dataset : dataset identifier (matches SCHEMA enum values).
    native_gsd : per-tile GSD in metres, if known.  For datasets with
                 variable GSD (VRSBench, CDVQA), this is typically None
                 and the categorical bucket is returned.

    Returns
    -------
    GSD bucket string: either a literal tag like "[GSD:0.5m]" or a
    categorical bucket like "VHR-native" / "CARTOSAT-proxy".
    """
    if dataset in _KNOWN_GSD:
        return _KNOWN_GSD[dataset]

    # Variable/unknown GSD datasets
    if native_gsd is not None:
        return f"[GSD:{_fmt_gsd(native_gsd)}]"

    # No native GSD provided → categorical bucket
    if dataset in ("vrsbench", "cdvqa"):
        return "VHR-native"

    # Fallback: unknown dataset, unknown GSD
    return "VHR-native"


def _fmt_gsd(gsd: float) -> str:
    """Format GSD value for bucket tag."""
    if gsd == int(gsd):
        return f"{int(gsd)}m"
    return f"{gsd:.2f}m"


# ---------------------------------------------------------------------------
# R4 — Dual-resolution proxy sample creation
# ---------------------------------------------------------------------------

def create_proxy_sample(
    sample: dict[str, Any],
    proxy_gsd: float | None = None,
) -> dict[str, Any]:
    """Create a proxy-branch copy of a sample for dual-resolution branching.

    The proxy sample has:
        - gsd_bucket set to CARTOSAT-proxy or RISAT-proxy
        - A synthetic id suffix "_proxy"
        - Original image paths preserved (the tier script handles actual
          GSD transformation of the images; this just sets metadata)

    Parameters
    ----------
    sample : original training sample dict.
    proxy_gsd : explicit proxy GSD; if None, inferred from modality.

    Returns
    -------
    New dict (deep copy) with proxy metadata.
    """
    proxy = copy.deepcopy(sample)

    # Determine proxy bucket from modality
    modality = proxy.get("modality", "optical")
    if modality == "sar":
        proxy_bucket = RISAT_PROXY
        default_gsd = DEFAULT_SAR_PROXY_GSD
    else:
        proxy_bucket = CARTOSAT_PROXY
        default_gsd = DEFAULT_OPTICAL_PROXY_GSD

    gsd = proxy_gsd if proxy_gsd is not None else default_gsd

    proxy["gsd_bucket"] = f"{proxy_bucket}[GSD:{_fmt_gsd(gsd)}]"
    proxy["id"] = f"{proxy['id']}_proxy"

    return proxy


# ---------------------------------------------------------------------------
# R4 — Curriculum mixing schedule
# ---------------------------------------------------------------------------

# Stage 2 curriculum: (native_ratio, proxy_ratio)
_CURRICULUM = {
    1: (1.0, 0.0),     # Stage 1: all native (no dual-res yet)
    2: (0.8, 0.2),     # Stage 2 start: 80:20 native:proxy
    3: (0.5, 0.5),     # Stage 2 mid: 50:50
    4: (0.3, 0.7),     # Stage 2 end: 30:70
}


def curriculum_mix_ratio(stage: int) -> tuple[float, float]:
    """Return (native_ratio, proxy_ratio) for the given training stage.

    Parameters
    ----------
    stage : training stage number (1–4 mapped to curriculum steps).

    Returns
    -------
    (native_ratio, proxy_ratio) as floats summing to 1.0.
    """
    if stage in _CURRICULUM:
        return _CURRICULUM[stage]
    if stage > 4:
        return _CURRICULUM[4]  # Stay at final ratio
    return _CURRICULUM[1]
