"""Shared stratified subsampling for tier scripts with a Selection requirement.

Several Tier 1/2/3 datasets specify a stratified selection fraction (e.g.
"10% stratified by change magnitude", "5% stratified by building-density
quartile", "stratified uniform across 6 categories"). This module gives
every tier script the same tested implementation instead of each rolling
its own `random.sample`, which silently degrades to plain uniform sampling.

Public API:
    stratified_sample(items, key_fn, fraction, seed) -> list
    quantile_bucket(value, edges) -> int
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable, TypeVar

T = TypeVar("T")


def stratified_sample(
    items: list[T],
    key_fn: Callable[[T], object],
    fraction: float,
    seed: int = 42,
) -> list[T]:
    """Sample `fraction` of items, proportionally within each key_fn(item) group.

    Every group contributes at least 1 item (if it has any), so small
    strata aren't wiped out by rounding down.
    """
    if fraction >= 1.0:
        return list(items)
    if not items:
        return []

    groups: dict[object, list[T]] = defaultdict(list)
    for it in items:
        groups[key_fn(it)].append(it)

    rng = random.Random(seed)
    out: list[T] = []
    for group in groups.values():
        rng.shuffle(group)
        n = max(1, round(len(group) * fraction))
        out.extend(group[: min(n, len(group))])

    rng.shuffle(out)
    return out


def quantile_bucket(value: float, edges: list[float]) -> int:
    """Bucket a scalar into len(edges)+1 bins given ascending edges.

    quantile_bucket(x, [0.25, 0.5, 0.75]) returns 0..3 (quartiles).
    """
    for i, e in enumerate(edges):
        if value <= e:
            return i
    return len(edges)


def quantile_edges(values: list[float], n_buckets: int) -> list[float]:
    """Compute the (n_buckets - 1) quantile edges that split `values` evenly."""
    if not values or n_buckets <= 1:
        return []
    import numpy as np

    qs = [100.0 * i / n_buckets for i in range(1, n_buckets)]
    return [float(v) for v in np.percentile(values, qs)]
