"""Tests for preprocess/tier1_vrsbench.py — caption + grounding + VQA + dual-resolution.

Parses VRSBench_train.json (LLaVA format) and verifies:
    1. Conversation parsing: [caption]/[refer]/[vqa] prefix extraction
    2. Referring coordinate conversion: {<x><y><x><y>} → [0,1000]
    3. R4 dual-resolution: each sample → native + proxy rows
    4. Manifest/resumability: first run, second-run-skips, idempotent, partial resume
    5. Task-bbox consistency: grounding → non-null bbox, caption/vqa → null bbox
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from preprocess.common.gsd import assign_gsd_bucket
from preprocess.common.io import Manifest
from preprocess.tier1_vrsbench import (
    parse_conversation,
    referring_to_qwen,
    process_entry,
    run_tier1_vrsbench,
)


# ---------------------------------------------------------------------------
# Fixtures: create synthetic VRSBench_train.json
# ---------------------------------------------------------------------------


def _make_entry(
    image: str = "P0000.png",
    task: str = "caption",
    instruction: str = "Describe this image in detail.",
    response: str = "A detailed aerial image.",
    entry_id: str = "test_001",
) -> dict:
    """Create a synthetic VRSBench_train.json entry."""
    prefix = f"[{task}]"
    return {
        "id": entry_id,
        "image": image,
        "conversations": [
            {"from": "human", "value": f"<image>\n{prefix} {instruction}"},
            {"from": "gpt", "value": response},
        ],
    }


@pytest.fixture
def vrsbench_dir():
    """Create a temporary VRSBench directory with synthetic train JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        entries = [
            _make_entry("P0000.png", "caption", "Describe this image.",
                        "An urban scene with buildings."),
            _make_entry("P0000.png", "refer", "Where is the aircraft?",
                        "{<45><45><59><59>}", entry_id="test_002"),
            _make_entry("P0000.png", "vqa", "What is this?",
                        "airport", entry_id="test_003"),
            _make_entry("P0001.png", "caption", "Describe this image.",
                        "A coastal area with water.", entry_id="test_004"),
            _make_entry("P0001.png", "vqa", "How many buildings?",
                        "5", entry_id="test_005"),
        ]

        with open(root / "VRSBench_train.json", "w") as f:
            json.dump(entries, f)

        yield root


# ---------------------------------------------------------------------------
# Unit tests: parse_conversation
# ---------------------------------------------------------------------------


class TestParseConversation:
    def test_caption(self):
        entry = _make_entry(task="caption", instruction="Describe this image.",
                            response="An urban scene with buildings.")
        task, inst, resp = parse_conversation(entry)
        assert task == "caption"
        assert inst == "Describe this image."
        assert resp == "An urban scene with buildings."

    def test_refer(self):
        entry = _make_entry(task="refer", instruction="Where is the aircraft?",
                            response="{<45><45><59><59>}")
        task, inst, resp = parse_conversation(entry)
        assert task == "refer"
        assert inst == "Where is the aircraft?"
        assert resp == "{<45><45><59><59>}"

    def test_vqa(self):
        entry = _make_entry(task="vqa", instruction="What is this?",
                            response="airport")
        task, inst, resp = parse_conversation(entry)
        assert task == "vqa"
        assert inst == "What is this?"
        assert resp == "airport"

    def test_unknown_prefix(self):
        entry = {
            "id": "x", "image": "x.png",
            "conversations": [
                {"from": "human", "value": "<image>\nSomething else"},
                {"from": "gpt", "value": "answer"},
            ]
        }
        task, inst, resp = parse_conversation(entry)
        assert task == "unknown"

    def test_strips_newline(self):
        entry = {
            "id": "x", "image": "x.png",
            "conversations": [
                {"from": "human", "value": "<image>\n\n[caption] Describe it."},
                {"from": "gpt", "value": "A scene."},
            ]
        }
        task, inst, resp = parse_conversation(entry)
        assert task == "caption"
        assert inst == "Describe it."


# ---------------------------------------------------------------------------
# Unit tests: referring_to_qwen
# ---------------------------------------------------------------------------


class TestReferringToQwen:
    def test_basic_conversion(self):
        bbox = referring_to_qwen("{<45><45><59><59>}")
        assert bbox == [[450.0, 450.0, 590.0, 590.0]]

    def test_zero_coordinates(self):
        bbox = referring_to_qwen("{<0><0><10><10>}")
        assert bbox == [[0.0, 0.0, 100.0, 100.0]]

    def test_max_coordinates(self):
        bbox = referring_to_qwen("{<99><99><100><100>}")
        assert bbox is not None
        assert bbox[0][2] == 1000.0

    def test_reversed_coordinates_returns_none(self):
        bbox = referring_to_qwen("{<59><59><45><45>}")
        assert bbox is None

    def test_zero_area_returns_none(self):
        bbox = referring_to_qwen("{<50><50><50><60>}")
        assert bbox is None

    def test_invalid_format_returns_none(self):
        bbox = referring_to_qwen("not a bbox")
        assert bbox is None

    def test_empty_string_returns_none(self):
        bbox = referring_to_qwen("")
        assert bbox is None


# ---------------------------------------------------------------------------
# Unit tests: process_entry
# ---------------------------------------------------------------------------


class TestProcessEntry:
    def test_caption_produces_sample(self):
        entry = _make_entry(task="caption")
        samples = process_entry(entry, None)
        assert len(samples) == 1
        assert samples[0]["task"] == "caption"
        assert samples[0]["bbox"] is None

    def test_refer_produces_sample_with_bbox(self):
        entry = _make_entry(task="refer", response="{<10><20><30><40>}")
        samples = process_entry(entry, None)
        assert len(samples) == 1
        assert samples[0]["task"] == "grounding"
        assert samples[0]["bbox"] == [[100.0, 200.0, 300.0, 400.0]]

    def test_vqa_produces_sample(self):
        entry = _make_entry(task="vqa", response="airport")
        samples = process_entry(entry, None)
        assert len(samples) == 1
        assert samples[0]["task"] == "vqa"
        assert samples[0]["bbox"] is None

    def test_invalid_referring_skipped(self):
        entry = _make_entry(task="refer", response="invalid")
        samples = process_entry(entry, None)
        assert len(samples) == 0

    def test_empty_response_skipped(self):
        entry = _make_entry(task="caption", response="")
        samples = process_entry(entry, None)
        assert len(samples) == 0

    def test_image_path_set_when_available(self, vrsbench_dir):
        images_dir = vrsbench_dir / "images"
        images_dir.mkdir(exist_ok=True)
        # Create a dummy image
        from PIL import Image
        import numpy as np
        img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        img.save(str(images_dir / "P0000.png"))

        entry = _make_entry("P0000.png", "caption")
        samples = process_entry(entry, images_dir)
        assert len(samples) == 1
        assert len(samples[0]["image_path"]) == 1
        assert Path(samples[0]["image_path"][0]).exists()


# ---------------------------------------------------------------------------
# R4 dual-resolution: native + proxy rows
# ---------------------------------------------------------------------------


class TestDualResolution:
    def test_each_sample_produces_two_rows(self, vrsbench_dir):
        output_dir = vrsbench_dir / "output"
        stats = run_tier1_vrsbench(vrsbench_dir, output_dir)
        assert stats["processed"] == 5

        with open(output_dir / "vrsbench.jsonl") as f:
            lines = [json.loads(l) for l in f if l.strip()]

        # 5 entries × 2 rows (native + proxy) = 10 rows
        assert len(lines) == 10

    def test_native_and_proxy_differ(self, vrsbench_dir):
        output_dir = vrsbench_dir / "output"
        run_tier1_vrsbench(vrsbench_dir, output_dir)

        with open(output_dir / "vrsbench.jsonl") as f:
            lines = [json.loads(l) for l in f if l.strip()]

        # Check that each pair has different gsd_bucket
        buckets = [l["gsd_bucket"] for l in lines]
        assert "VHR-native" in buckets
        assert any("CARTOSAT-proxy" in b for b in buckets)


# ---------------------------------------------------------------------------
# Manifest/resumability (same 5-test pattern as OSCD)
# ---------------------------------------------------------------------------


class TestManifestResumability:
    def test_first_run_processes_all(self, vrsbench_dir):
        output_dir = vrsbench_dir / "output"
        stats = run_tier1_vrsbench(vrsbench_dir, output_dir)
        assert stats["processed"] == 5
        assert stats["skipped"] == 0

    def test_second_run_skips_all(self, vrsbench_dir):
        output_dir = vrsbench_dir / "output"
        stats1 = run_tier1_vrsbench(vrsbench_dir, output_dir)
        assert stats1["processed"] == 5

        stats2 = run_tier1_vrsbench(vrsbench_dir, output_dir)
        assert stats2["processed"] == 0
        assert stats2["skipped"] == 5

    def test_idempotent_output(self, vrsbench_dir):
        output_dir = vrsbench_dir / "output"
        run_tier1_vrsbench(vrsbench_dir, output_dir)
        jsonl_path = output_dir / "vrsbench.jsonl"
        hash1 = hashlib.md5(jsonl_path.read_bytes()).hexdigest()

        run_tier1_vrsbench(vrsbench_dir, output_dir)
        hash2 = hashlib.md5(jsonl_path.read_bytes()).hexdigest()
        assert hash1 == hash2

    def test_partial_run_resumes(self, vrsbench_dir):
        output_dir = vrsbench_dir / "output"
        manifest_path = output_dir / "manifest.parquet"
        jsonl_path = output_dir / "vrsbench.jsonl"

        # Manually process first 2 entries
        with open(vrsbench_dir / "VRSBench_train.json") as f:
            data = json.load(f)

        manifest = Manifest(manifest_path)
        native_bucket = assign_gsd_bucket("vrsbench", None)
        from preprocess.common.gsd import create_proxy_sample
        from preprocess.common.io import append_jsonl

        for i, entry in enumerate(data[:2]):
            samples = process_entry(entry, None, sample_id=f"{i:07d}")
            for s in samples:
                s["gsd_bucket"] = native_bucket
                append_jsonl(s, jsonl_path)
                proxy = create_proxy_sample(s)
                append_jsonl(proxy, jsonl_path)

            manifest.mark_processed(
                sample_id=f"{i:07d}",
                dataset="vrsbench", split="train", output_shard="local",
            )

        # Full run should process remaining 3
        stats = run_tier1_vrsbench(vrsbench_dir, output_dir)
        assert stats["processed"] == 3
        assert stats["skipped"] == 2


# ---------------------------------------------------------------------------
# Validation: produced samples pass frozen schema
# ---------------------------------------------------------------------------


class TestValidation:
    def test_all_samples_validate(self, vrsbench_dir):
        with open(vrsbench_dir / "VRSBench_train.json") as f:
            data = json.load(f)

        output_dir = vrsbench_dir / "output"
        native_bucket = assign_gsd_bucket("vrsbench", None)

        for entry in data:
            samples = process_entry(entry, None)
            for sample in samples:
                from preprocess.validator import validate_sample
                sample["gsd_bucket"] = native_bucket
                # Skip validation if no image_path (images not available locally)
                if not sample["image_path"]:
                    continue
                ok, errs = validate_sample(sample)
                assert ok, f"Validation failed for {sample['id']}: {errs}"


# ---------------------------------------------------------------------------
# Task-bbox consistency
# ---------------------------------------------------------------------------


class TestTaskBboxConsistency:
    def test_grounding_has_nonnull_bbox(self, vrsbench_dir):
        with open(vrsbench_dir / "VRSBench_train.json") as f:
            data = json.load(f)
        for entry in data:
            if "[refer]" in entry["conversations"][0]["value"]:
                samples = process_entry(entry, None)
                for s in samples:
                    assert s["bbox"] is not None, f"Grounding sample {s['id']} has null bbox"

    def test_caption_vqa_have_null_bbox(self, vrsbench_dir):
        with open(vrsbench_dir / "VRSBench_train.json") as f:
            data = json.load(f)
        for entry in data:
            conv_text = entry["conversations"][0]["value"]
            if "[caption]" in conv_text or "[vqa]" in conv_text:
                samples = process_entry(entry, None)
                for s in samples:
                    assert s["bbox"] is None, f"{s['task']} sample {s['id']} has non-null bbox"


# ---------------------------------------------------------------------------
# Issue 7: dynamic proxy scale factor
# Issue 9: proxy PNG compression
# ---------------------------------------------------------------------------


class TestGenerateProxyImage:
    """Tests for the generate_proxy_image() function (Issues 7 + 9)."""

    def test_dynamic_scale_factor_non_square(self, tmp_path):
        """Issue 7: a 256×128 source image should yield a 64×32 proxy (4× reduction)."""
        import numpy as np
        from PIL import Image
        from preprocess.tier1_vrsbench import generate_proxy_image

        src = tmp_path / "src.png"
        img = Image.fromarray(np.zeros((128, 256, 3), dtype=np.uint8))  # H=128, W=256
        img.save(str(src))

        proxy_dir = tmp_path / "proxy"
        result = generate_proxy_image(src, proxy_dir, "src.png", scale_factor=4)
        assert result is not None
        proxy = Image.open(result)
        w, h = proxy.size  # PIL: (width, height)
        assert w == 64, f"Expected proxy width 64, got {w}"
        assert h == 32, f"Expected proxy height 32, got {h}"

    def test_small_image_clamped_to_min_32(self, tmp_path):
        """Issue 7: a 16×16 image divided by 4 = 4, must be clamped to 32."""
        import numpy as np
        from PIL import Image
        from preprocess.tier1_vrsbench import generate_proxy_image

        src = tmp_path / "tiny.png"
        img = Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8))
        img.save(str(src))

        proxy_dir = tmp_path / "proxy"
        result = generate_proxy_image(src, proxy_dir, "tiny.png", scale_factor=4)
        assert result is not None
        proxy = Image.open(result)
        w, h = proxy.size
        assert w >= 32, f"Expected width >= 32 (clamped), got {w}"
        assert h >= 32, f"Expected height >= 32 (clamped), got {h}"

    def test_proxy_smaller_than_compress_level_0(self, tmp_path):
        """Issue 9: compress_level=6 proxy file must be smaller than level=0 equivalent."""
        import numpy as np
        from PIL import Image
        from preprocess.tier1_vrsbench import generate_proxy_image

        # Use a solid-color image — highly compressible
        src = tmp_path / "big.png"
        arr = np.full((512, 512, 3), 128, dtype=np.uint8)
        img = Image.fromarray(arr)
        img.save(str(src))

        proxy_dir_6 = tmp_path / "proxy_6"
        result = generate_proxy_image(src, proxy_dir_6, "big.png")
        assert result is not None
        size_6 = Path(result).stat().st_size

        # Write same content at level 0 for comparison
        proxy_l0 = tmp_path / "proxy_l0.png"
        proxy_img = Image.open(result)
        proxy_img.save(str(proxy_l0), format="PNG", compress_level=0)
        size_0 = proxy_l0.stat().st_size

        assert size_6 <= size_0, (
            f"compress_level=6 ({size_6} B) should be <= level=0 ({size_0} B)"
        )

    def test_idempotent_existing_proxy_not_overwritten(self, tmp_path):
        """generate_proxy_image() returns existing path without regenerating."""
        import numpy as np
        from PIL import Image
        from preprocess.tier1_vrsbench import generate_proxy_image

        src = tmp_path / "src.png"
        Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(str(src))

        proxy_dir = tmp_path / "proxy"
        result1 = generate_proxy_image(src, proxy_dir, "src.png")
        mtime1 = Path(result1).stat().st_mtime

        result2 = generate_proxy_image(src, proxy_dir, "src.png")
        mtime2 = Path(result2).stat().st_mtime

        assert result1 == result2
        assert mtime1 == mtime2, "Proxy was regenerated when it should have been reused"

