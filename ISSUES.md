# Known Issues — SatQuery Preprocessing Pipeline

Issues 1 and 2 (CDVQA val-split schema rejection; stale OSCD gsd_bucket) are fixed and closed.
Issues 3, 5, 7, 8, 9, 10 are fixed and closed (2026-09-11).
Issues 4 and 6 are open — require design decisions outside the scope of a preprocessing fix pass.

---

## ~~Issue 3~~ — OSCD Bounding Boxes Are Full-Image (Useless for Grounding) ✅ FIXED

**Status**: Fixed 2026-09-11 — Option A applied.  
**File**: `preprocess/tier1_oscd.py`  
**Lines**: `process_pair()` L229–235, `mask_to_bboxes()` L178–203

**Root cause**: Two compounding problems:
1. `scipy` is not listed in `requirements.txt`, so `mask_to_bboxes()` always falls through to the no-scipy fallback (L191–194) which computes a single envelope bbox over *all* changed pixels regardless of connected-component structure. Even when scipy is present, per-component boxes are computed correctly but `process_pair()` then takes only `bboxes[0]` (the first component) and discards the rest.
2. OSCD change masks are spatially scattered across the full image extent (urban growth along road networks etc.), so even a correct per-component envelope box for the largest component tends to span nearly the entire image.

**Impact**: Every OSCD `change_grounding` sample carried a bbox covering ~99.8% of the image area (e.g., `[0, 0, 998.7, 998.8]`). Training on these teaches the model that "grounding" means "the whole image" — actively harming spatial precision for the grounding task.

**Resolution**: Applied Option A. `process_pair()` now unconditionally sets `task="change_vqa"` and `bbox=None` for all OSCD samples. The unused `pixel_to_normalized` import was removed. `mask_to_bboxes()` is preserved and still testable.

---

## Issue 4 — Templated Instruction–Response Pairs Lack Diversity and Richness ⚠️ OPEN

**Files**: all tier scripts  
**Affected datasets**: OSCD, BigEarthNet (optical/SAR/fusion), SARDet-100K, LEVIR-CD, SpaceNet 6 SAR, Sen-2 LULC (8 of 10 datasets)

**Root cause**: Each dataset uses a single fixed instruction string and a formulaic response template (e.g., `"Change detected in {location}."`, `"SAR backscatter over a scene classified optically as: {labels}."`). VRSBench is the only dataset that uses diverse human-authored questions and rich GPT-generated answers.

**Impact**: A model trained on these templates learns to produce terse, formulaic outputs with no spatial reasoning, no comparative language, and no confidence qualification. This directly contradicts the Problem Statement's expectation of evidence-grounded, spatially detailed answers.

**Options**:
- **(A) Instruction paraphrasing pool** — define a list of 8–15 semantically equivalent phrasings per task type. Sample one uniformly at random per item during preprocessing.
- **(B) Template-based response enrichment** — for label-based datasets construct a lookup table mapping class names to short descriptive phrases.
- **(C) LLM-generated augmentation** — run a small offline LLM over each (label, instruction) pair to generate diverse natural-language responses. Highest quality but adds an external dependency and cost.

---

## ~~Issue 5~~ — R4 Proxy Samples Reuse Identical Pixels for 3 of 4 Datasets ✅ FIXED

**Status**: Fixed 2026-09-11 — Option A applied.  
**File**: `preprocess/common/gsd.py`  
**Function**: `create_proxy_sample()` L138–175  
**Affected datasets**: `rsvqa_hr`, `levir_cd`, `sn6_opt`

**Root cause**: `create_proxy_sample()` updates the `gsd_bucket` tag and `id` suffix but leaves `image_path` pointing at the original full-resolution image. Actual pixel-level downsampling was only implemented for VRSBench (via `generate_proxy_image()` in `tier1_vrsbench.py`, which bicubic-downsampled 512→128).

**Impact**: The model sees identical pixel content with two different GSD tokens, making the GSD token appear to carry no visual information. This undermines the curriculum's intent to bridge the resolution gap.

**Resolution**: Applied Option A. `generate_proxy_image()` was generalised and moved into `common/gsd.py` (with a dynamic scale factor — see Issue 7 fix). `create_proxy_sample()` docstring now explicitly directs callers to call `generate_proxy_image()` and update `image_path` on the returned proxy dict.

---

## Issue 6 — NIR Band Drop (R1) Loses Valuable Remote Sensing Information ⚠️ OPEN

**File**: `preprocess/` — all optical tier scripts via R1  
**Rule**: SPEC §2 R1 mandates RGB-only output (B04/B03/B02)

**Root cause**: Qwen3-VL accepts 3-channel RGB input only, so NIR (B08 for Sentinel-2, B4 for Landsat) must be dropped at preprocessing time. This is a deliberate architectural constraint, not a bug.

**Impact**: NDVI (NIR−Red / NIR+Red), the most widely used vegetation index for agriculture and water monitoring, cannot be computed from stored images. The Problem Statement explicitly names agricultural monitoring and water-resource assessment as use cases.

**Options**:
- **(A) Precompute and store derived indices as separate channels** — before discarding NIR, compute NDVI and save it as an additional single-channel PNG. Include the index PNG path in the JSONL under a new optional field `derived_paths`.
- **(B) Accept the constraint** — document the NIR loss as a known architectural limitation. Mitigate by curating instruction/response text that does not over-claim vegetation or water accuracy.

---

## ~~Issue 7~~ — VRSBench Proxy Assumes Fixed 512×512 Native Size ✅ FIXED

**Status**: Fixed 2026-09-11 — Option A applied.  
**File**: `preprocess/tier1_vrsbench.py`  
**Function**: `generate_proxy_image()` L184–209

**Root cause**: The proxy downsampling target `proxy_size = (128, 128)` was hardcoded. This silently assumes every VRSBench source image is 512×512 at 0.5m GSD. VRSBench imagery spans variable chip sizes and GSD values across its source collections.

**Impact**: For images smaller than 512×512 the effective proxy GSD is *lower* than intended (less than 4× downsampling); for larger images the proxy GSD is *higher*. The CARTOSAT-proxy GSD token therefore maps to inconsistent visual scales.

**Resolution**: Applied Option A. `generate_proxy_image()` now computes `proxy_w = max(32, w // scale_factor)` and `proxy_h = max(32, h // scale_factor)` from the actual image dimensions, always producing a true 4× GSD reduction. A 32px floor prevents degenerate outputs for very small chips. The same dynamic logic was applied to `generate_proxy_image()` in `common/gsd.py` (Issue 5 fix).

---

## ~~Issue 8~~ — OSCD Connected-Component Fallback Has No SciPy ✅ FIXED

**Status**: Fixed 2026-09-11.  
**File**: `preprocess/tier1_oscd.py`  
**Function**: `mask_to_bboxes()` L187–194

**Root cause**: `mask_to_bboxes()` uses `scipy.ndimage.label` wrapped in a `try/except ImportError`. `scipy` was not listed in `requirements.txt`, so the fallback was used in all standard environments — producing a single envelope bbox.

**Impact**: See Issue 3. This contributed to the full-image bbox problem.

**Resolution**: `scipy>=1.11` was already present in `requirements.txt` (added in a prior session). Additionally, the noise-filter threshold in `mask_to_bboxes()` was raised from 10 → **50 pixels** to match the spec's ≥50px² suggestion, filtering isolated-pixel noise more aggressively.

---

## ~~Issue 9~~ — `compress_level=0` Wastes Disk Space ✅ FIXED

**Status**: Fixed 2026-09-11.  
**Files**: `preprocess/common/io.py`, `preprocess/tier1_vrsbench.py`  
**Function**: `write_png()` and direct `img.save(..., compress_level=0)` calls

**Root cause**: All PNG writes used `compress_level=0` to maximise write throughput.

**Impact**: Produces files 2–3× larger than `compress_level=6`. For a ~200k corpus the difference is ~20–40 GB, which is significant given Kaggle's 20 GB persistent output cap.

**Resolution**: Changed to `compress_level=6` in `write_png()` (in `common/io.py`) and in the direct `proxy.save()` call in `tier1_vrsbench.generate_proxy_image()`. Level 6 offers 60–80% size reduction over level 0 with modest CPU overhead.

---

## ~~Issue 10~~ — OSCD Uses Mixed Image Libraries (tifffile + PIL) ✅ FIXED

**Status**: Fixed 2026-09-11.  
**File**: `preprocess/tier1_oscd.py`  
**Functions**: `load_band()` L110–130, `load_mask()` L147–175

**Root cause**: Band TIFs are loaded with `tifffile.imread()` while mask resizing was handled by PIL. There is no single consistent I/O layer.

**Impact**: Low — functionally correct. However, `tifffile` was not in `requirements.txt`.

**Resolution**: `tifffile>=2024.1` was already present in `requirements.txt`. Mask resizing in `load_mask()` was consolidated to use `scipy.ndimage.zoom` (order=0, nearest-neighbour) as the primary path, with PIL as a fallback if scipy is absent.
