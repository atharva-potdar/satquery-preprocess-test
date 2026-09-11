"""Shared I/O for preprocessing tiers.

Handles:
    - Manifest tracking (id, dataset, split, processed, output_shard)
    - PNG writing (8-bit, 3-channel, lossless)
    - JSONL line writing
    - Tar sharding (≤2000 files per shard)

Public API:
    Manifest  — tracks processing state per sample id
    write_png(arr, path) — write 8-bit 3-channel PNG
    append_jsonl(sample, path) — append one JSONL line
    ShardWriter — manages tar shards with file-count caps
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

class Manifest:
    """Track which sample ids have been processed.

    Backed by a simple JSONL file.  On restart, load existing manifest
    and skip any id already marked processed=True.

    File format (one line per sample):
        {"id": "...", "dataset": "...", "split": "...", "processed": true, "output_shard": "shard_000"}
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._entries: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                self._entries[entry["id"]] = entry

    def is_processed(self, sample_id: str) -> bool:
        return self._entries.get(sample_id, {}).get("processed", False)

    def mark_processed(
        self,
        sample_id: str,
        dataset: str,
        split: str,
        output_shard: str,
    ) -> None:
        entry = {
            "id": sample_id,
            "dataset": dataset,
            "split": split,
            "processed": True,
            "output_shard": output_shard,
        }
        self._entries[sample_id] = entry
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def count_processed(self) -> int:
        return sum(1 for e in self._entries.values() if e.get("processed"))

    def count_total(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# PNG writing
# ---------------------------------------------------------------------------

def write_png(arr: np.ndarray, path: str | Path) -> None:
    """Write a numpy array as an 8-bit, 3-channel, lossless PNG.

    Parameters
    ----------
    arr : (H, W, 3) uint8 array, or (H, W) / (H, W, 1) that will be
          converted to 3-channel by replication.
    path : output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.concatenate([arr, arr, arr], axis=-1)

    assert arr.ndim == 3 and arr.shape[2] == 3, \
        f"Expected (H, W, 3), got {arr.shape}"

    img = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    img.save(str(path), format="PNG", compress_level=0)  # lossless, no compression


# ---------------------------------------------------------------------------
# JSONL writing
# ---------------------------------------------------------------------------

def append_jsonl(sample: dict[str, Any], path: str | Path) -> None:
    """Append one sample dict as a JSONL line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Tar shard writer
# ---------------------------------------------------------------------------

class ShardWriter:
    """Manage output sharding into tar archives with file-count caps.

    Each shard is a tar.gz containing up to max_files PNGs.
    When a shard reaches capacity, a new one is automatically started.

    Usage:
        writer = ShardWriter(output_dir, prefix="tier1_oscd", max_files=2000)
        shard_name = writer.add(png_path)
        writer.close()
    """

    def __init__(
        self,
        output_dir: str | Path,
        prefix: str = "shard",
        max_files: int = 2000,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.max_files = max_files
        self._shard_count = 0
        self._file_count = 0
        self._current_tar: tarfile.TarFile | None = None
        self._open_new_shard()

    def _open_new_shard(self) -> None:
        if self._current_tar is not None:
            self._current_tar.close()
        shard_name = f"{self.prefix}_shard_{self._shard_count:04d}.tar.gz"
        shard_path = self.output_dir / shard_name
        self._current_tar = tarfile.open(str(shard_path), "w:gz")
        self._file_count = 0

    @property
    def current_shard_name(self) -> str:
        return f"{self.prefix}_shard_{self._shard_count:04d}.tar.gz"

    def add(self, file_path: str | Path) -> str:
        """Add a file to the current shard.  Returns the shard name.

        If the shard is full, a new one is started automatically.
        """
        file_path = Path(file_path)
        if self._file_count >= self.max_files:
            self._shard_count += 1
            self._open_new_shard()

        self._current_tar.add(str(file_path), arcname=file_path.name)
        self._file_count += 1
        return self.current_shard_name

    def close(self) -> None:
        if self._current_tar is not None:
            self._current_tar.close()
            self._current_tar = None
