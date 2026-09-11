# Known Issues — SatQuery Preprocessing Pipeline

Issues 1 and 2 (CDVQA val-split schema rejection; stale OSCD gsd_bucket) are fixed and closed. Issues below are open, ordered by approximate training-quality impact.

---

## Issue 3 — OSCD Bounding Boxes Are Full-Image (Useless for Grounding)

**File**: `preprocess/tier1_oscd.py`  
**Lines**: `process_pair()` L229–235, `mask_to_bboxes()` L178–203

**Root cause**: Two compounding problems:
1. `scipy` is not listed in `requirements.txt`, so `mask_to_bboxes()` always falls through to the no-scipy fallback (L191–194) which computes a single envelope bbox over *all* changed pixels regardless of connected-component structure. Even when scipy is present, per-component boxes are computed correctly but `process_pair()` then takes only `bboxes[0]` (the first component) and discards the rest.
2. OSCD change masks are spatially scattered across the full image extent (urban growth along road networks etc.), so even a correct per-component envelope box for the largest component tends to span nearly the entire image.

**Impact**: Every OSCD `change_grounding` sample carries a bbox that covers ~99.8% of the image area (e.g., `[0, 0, 998.7, 998.8]`). Training on these teaches the model that "grounding" means "the whole image" — actively harming spatial precision for the grounding task.

**Options**:
- **(A) Demote to `change_vqa`** — set `bbox=None` for all OSCD samples and emit `task: "change_vqa"` unconditionally. Simple one-line change, immediately stops harmful signal. Loses OSCD grounding data entirely.
- **(B) Full fix** — add `scipy` to `requirements.txt`, pass the full `bboxes` list (not just `bboxes[0]`) into the sample, add a minimum-area threshold to filter noise (suggest ≥50px²), update instruction/response templates to reference specific changed regions. Requires design decisions on response phrasing for multi-region outputs.

---

## Issue 4 — Templated Instruction–Response Pairs Lack Diversity and Richness

**Files**: all tier scripts  
**Affected datasets**: OSCD, BigEarthNet (optical/SAR/fusion), SARDet-100K, LEVIR-CD, SpaceNet 6 SAR, Sen-2 LULC (8 of 10 datasets)

**Root cause**: Each dataset uses a single fixed instruction string and a formulaic response template (e.g., `"Change detected in {location}."`, `"SAR backscatter over a scene classified optically as: {labels}."`). VRSBench is the only dataset that uses diverse human-authored questions and rich GPT-generated answers.

**Impact**: A model trained on these templates learns to produce terse, formulaic outputs with no spatial reasoning, no comparative language, and no confidence qualification. This directly contradicts the Problem Statement's expectation of evidence-grounded, spatially detailed answers.

**Options**:
- **(A) Instruction paraphrasing pool** — define a list of 8–15 semantically equivalent phrasings per task type. Sample one uniformly at random per item during preprocessing.
- **(B) Template-based response enrichment** — for label-based datasets construct a lookup table mapping class names to short descriptive phrases.
- **(C) LLM-generated augmentation** — run a small offline LLM over each (label, instruction) pair to generate diverse natural-language responses. Highest quality but adds an external dependency and cost.

---

## Issue 5 — R4 Proxy Samples Reuse Identical Pixels for 3 of 4 Datasets

**File**: `preprocess/common/gsd.py`  
**Function**: `create_proxy_sample()` L138–175  
**Affected datasets**: `rsvqa_hr`, `levir_cd`, `sn6_opt`

**Root cause**: `create_proxy_sample()` updates the `gsd_bucket` tag and `id` suffix but leaves `image_path` pointing at the original full-resolution image. Actual pixel-level downsampling is only implemented for VRSBench (via `generate_proxy_image()` in `tier1_vrsbench.py`, which bicubic-downsamples 512→128).

**Impact**: The model sees identical pixel content with two different GSD tokens, making the GSD token appear to carry no visual information. This undermines the curriculum's intent to bridge the resolution gap.

**Options**:
- **(A) Generalise `generate_proxy_image()`** — move the bicubic downsampling logic from `tier1_vrsbench.py` into `common/gsd.py` or a new `common/proxy.py`.
- **(B) Scale-factor table** — store a per-dataset `proxy_scale` factor in `_KNOWN_GSD` or a companion dict and apply it uniformly in `create_proxy_sample()`.

---

## Issue 6 — NIR Band Drop (R1) Loses Valuable Remote Sensing Information

**File**: `preprocess/` — all optical tier scripts via R1  
**Rule**: SPEC §2 R1 mandates RGB-only output (B04/B03/B02)

**Root cause**: Qwen3-VL accepts 3-channel RGB input only, so NIR (B08 for Sentinel-2, B4 for Landsat) must be dropped at preprocessing time. This is a deliberate architectural constraint, not a bug.

**Impact**: NDVI (NIR−Red / NIR+Red), the most widely used vegetation index for agriculture and water monitoring, cannot be computed from stored images. The Problem Statement explicitly names agricultural monitoring and water-resource assessment as use cases.

**Options**:
- **(A) Precompute and store derived indices as separate channels** — before discarding NIR, compute NDVI and save it as an additional single-channel PNG. Include the index PNG path in the JSONL under a new optional field `derived_paths`.
- **(B) Accept the constraint** — document the NIR loss as a known architectural limitation. Mitigate by curating instruction/response text that does not over-claim vegetation or water accuracy.

---

## Issue 7 — VRSBench Proxy Assumes Fixed 512×512 Native Size

**File**: `preprocess/tier1_vrsbench.py`  
**Function**: `generate_proxy_image()` L184–209

**Root cause**: The proxy downsampling target `proxy_size = (128, 128)` is hardcoded. This silently assumes every VRSBench source image is 512×512 at 0.5m GSD. VRSBench imagery spans variable chip sizes and GSD values across its source collections.

**Impact**: For images smaller than 512×512 the effective proxy GSD is *lower* than intended (less than 4× downsampling); for larger images the proxy GSD is *higher*. The CARTOSAT-proxy GSD token therefore maps to inconsistent visual scales.

**Options**:
- **(A) Dynamic scale factor** — compute `proxy_size = (max(32, w // 4), max(32, h // 4))` from actual dimensions to always apply a 4× reduction.
- **(B) GSD-aware downsampling** — read the image's native GSD from VRSBench metadata and compute scale factor exactly.

---

## Issue 8 — OSCD Connected-Component Fallback Has No SciPy

**File**: `preprocess/tier1_oscd.py`  
**Function**: `mask_to_bboxes()` L187–194

**Root cause**: `mask_to_bboxes()` uses `scipy.ndimage.label` wrapped in a `try/except ImportError`. `scipy` is not listed in `requirements.txt`, so the fallback is used in all standard environments — producing a single envelope bbox.

**Impact**: See Issue 3. This contributes to the full-image bbox problem.

**Fix**: Add `scipy>=1.10` to `requirements.txt`. The fallback can remain as a safety net but shouldn't be the normal path.

---

## Issue 9 — `compress_level=0` Wastes Disk Space

**File**: `preprocess/common/io.py`  
**Function**: `write_png()` and direct `img.save(..., compress_level=0)` calls

**Root cause**: All PNG writes use `compress_level=0` to maximise write throughput.

**Impact**: Produces files 2–3× larger than `compress_level=6`. For a ~200k corpus the difference is ~20–40 GB, which is significant given Kaggle's 20 GB persistent output cap.

**Fix**: Change to `compress_level=6` in `write_png()` (and any direct `img.save()` calls). Level 6 offers 60–80% size reduction over level 0 with modest CPU overhead.

---

## Issue 10 — OSCD Uses Mixed Image Libraries (tifffile + PIL)

**File**: `preprocess/tier1_oscd.py`  
**Functions**: `load_band()` L110–130, `load_mask()` L147–175

**Root cause**: Band TIFs are loaded with `tifffile.imread()` while mask resizing is handled by PIL. There is no single consistent I/O layer.

**Impact**: Low — functionally correct. However, `tifffile` is not in `requirements.txt`.

**Fix**: Add `tifffile>=2023.1` to `requirements.txt` and optionally consolidate mask resizing to use `scipy.ndimage.zoom` (nearest-neighbour) rather than PIL.
