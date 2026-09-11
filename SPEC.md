# SatQuery AI — Final Dataset & Preprocessing Specification (v4)

**Status**: Frozen for code generation. All factual corrections applied, all operational constraints resolved, schema frozen, leakage risks eliminated.

---

## 1. Problem Statement Mandates (Traceability)

| # | Mandate | Primary Datasets |
|---|---|---|
| 1 | RS domain adaptation via BigEarthNet | BigEarthNet (reBEN v2.0) |
| 2 | Single-image VQA + {captioning ∨ grounding} | VRSBench, RSVQA-HR, BigEarthNet, Sen-2 LULC |
| 3 | Bi-temporal change description/VQA | CDVQA (primary), LEVIR-CD, OSCD |
| 4 | Cross-modal optical–SAR joint analysis | BigEarthNet, SpaceNet 6 (optical+SAR), SARDet-100K |
| 5 | Agentic orchestration | Architecture layer (not data) |

---

## 2. Universal Preprocessing Rules (R1–R7, Final)

| Rule | Specification |
|---|---|
| **R1 — Channel policy** | All optical imagery truncated to B4/B3/B2 (RGB). NIR dropped from *all* pixel tensors, no exceptions. Cross-modal fusion signal built from SAR-only physics (VV, VH, VV/VH ratio). |
| **R2 — Radiometric normalization** | Percentile clipping (2nd/98th) computed **once globally per sensor-and-band** (e.g., one stat for Sentinel-2 B4, one for Cartosat-proxy VHR RGB, one for Sentinel-1 VV in dB). Not per-image. |
| **R3 — SAR pseudo-RGB (physics-corrected)** | 1. Linear → dB: `dB = 10·log10(x + ε)`<br>2. Clip: VV ∈ [-25, 0] dB, VH ∈ [-30, -5] dB<br>3. Map: `R = VV(dB)`, `G = VH(dB)`, `B = (VV−VH)` in dB-space post-clipping<br>4. Quad-pol (SpaceNet6): `R=HH, G=VV, B=VH` |
| **R4 — Dual-resolution branching** | For benchmark datasets (VRSBench, RSVQA-HR, CDVQA): each training image → **two independent samples**:<br>• Native branch → `gsd_bucket` per assignment rule below<br>• Proxy branch → `CARTOSAT-proxy` (optical ~2m) or RISAT-band matched (SAR ~2–10m)<br>Both samples interleaved in JSONL. Curriculum mixing: 80:20 → 50:50 → 30:70 native:proxy across Stage 2. |
| **R5 — SAR GSD calibration** | Optical VHR → Cartosat-2S (~2m MS). SAR → RISAT proxy band (Fine Resolution Stripmap/Spotlight, ~2–10m). SpaceNet6 SAR (0.5m) & SARDet-100K finer patches downsampled into this band. BigEarthNet S1 (10m) left as coarse anchor. |
| **R6 — Bounding box format** | Normalized `[0,1000]`, ordered `(x_topleft, y_topleft), (x_bottomright, y_bottomright)` — **Qwen convention**. Not y-first. |
| **R7 — Format standardization** | All outputs: 8-bit, 3-channel, lossless PNG. Bi-temporal/cross-modal pairs stored as 1×2 spatial concatenation **plus** two source images separately (agentic routing). |

---

## 3. Master Dataset Allocation (Final, Leakage-Proof)

### TIER 0 — Primary Domain Adaptation Backbone

| Dataset | Purpose | Usage | Selection | Treatment |
|---|---|---|---|---|
| **BigEarthNet (reBEN v2.0 / BigEarthNet-MM)** | Mandate 1: RS domain adaptation | **~35% corpus** (~60k of ~200k effective) | Stratified across 19-class CORINE, 10 countries, controlled cloud-contaminated minority | R1–R3, native 10m (R5 anchor). 3 products/pair: optical RGB, SAR pseudo-RGB, 1×2 concat. |

### TIER 1 — VHR Visual Acuity, Grounding, Optical Bi-Temporal (Cartosat Proxy)

| Dataset | Purpose | Usage | Selection | Treatment |
|---|---|---|---|---|
| **VRSBench (train)** | Mandate 2: captioning + grounding + VQA | 100% train split (×2 via R4) | Full stratified; **test excluded** | R1, R2, R4, R6. Native GSD variable → bucket `VHR-native`. |
| **RSVQA-HR (train)** | Mandate 2: VQA baseline | 100% train split (×2 via R4) | Full; **test excluded** | R1, R2, R4. **Native 0.15m (USGS HRO)** → `[GSD:0.15m]`. |
| **CDVQA (official, on SECOND imagery)** | Mandate 3: bi-temporal change VQA | **2,000 pairs** (1,600 train + 400 val) | **Official train+val ONLY. Both test splits (~968 pairs) EXCLUDED.** | 1×2 concat, native res kept. `pair_type: bitemporal`. |
| **LEVIR-CD** | Bi-temporal building-change grounding | 10% (stratified all change-magnitude deciles) | Stratified by change magnitude, not top decile | R4 dual-res; masks→boxes R6. `pair_type: bitemporal`. |
| **SpaceNet 6 (Optical)** | VHR building grounding + fusion half | 5% | Stratified by building-density quartile | R1, R2, R4. WorldView-2 0.5m pan-sharpened RGB. |
| **OSCD (optical, all 24 pairs)** | Bi-temporal regularizer | 100% (14 train + 10 test-labeled) | **No prescribed benchmark → both usable** | R1 (13-band→RGB), R2, 1×2 concat, masks→boxes R6. `pair_type: bitemporal`. |

### TIER 2 — SAR Physics & Cross-Modal Literacy (RISAT Proxy)

| Dataset | Purpose | Usage | Selection | Treatment |
|---|---|---|---|---|
| **SARDet-100K** | SAR object vocab (ship, aircraft, bridge, tank, car, harbor) | 10% | Stratified uniform across 6 categories | R1/R3, R5 GSD calibration. |
| **SpaceNet 6 (SAR)** | Cross-modal fusion + SAR artifacts (layover/shadow) | 5% | Paired 1:1 with optical selection | Quad-pol R3 (`R=HH,G=VV,B=VH`), R5 downsample. |
| **BigEarthNet S1** | Coarse SAR anchor | (in Tier 0) | — | — |

**OSCD-SAR**: **Dropped** — not a packaged dataset. Documented as known limitation.

### TIER 3 — Contextual Regularization (Indian Geography)

| Dataset | Purpose | Usage | Selection | Treatment |
|---|---|---|---|---|
| **Sen-2 LULC** | Indian-subcontinent LULC taxonomy | 5% (~10k of 213,761) | Stratified uniform across 7 classes | Native 10m (already B4/B3/B2). Masks→boxes R6, mapped to 7-class Indian taxonomy. |

---

## 4. Frozen JSONL Schema (All Tiers)

```json
{
  "id": "string (globally unique, e.g. 'bigen_001234')",
  "dataset": "string (source: 'bigen' | 'vrsbench' | 'rsvqa_hr' | 'cdvqa' | 'levir_cd' | 'sn6_opt' | 'oscd' | 'sardet' | 'sn6_sar' | 'sen2lulc')",
  "task": "string (vqa | caption | grounding | change_vqa | change_grounding | fusion_vqa | fusion_grounding)",
  "image_path": ["string", ...],          // ALWAYS a list. len=1 single-image, len=2 bitemporal/cross-modal.
  "pair_type": "single | bitemporal | cross-modal",  // EXPLICIT. Drives ChatML template selection.
  "gsd_bucket": "string",                 // ASSIGNMENT RULE BELOW.
  "split": "train | val_internal",        // Internal validation split (~2-3% stratified from train).
  "instruction": "string (user question, NO GSD tag)",
  "response": "string (target answer)",
  "bbox": [[x1,y1,x2,y2], ...] or null,  // [0,1000], (x_topleft,y_topleft),(x_bottomright,y_bottomright)
  "modality": "optical | sar | optical+sar"   // Note: "optical+sar" not "cross-modal" (avoids collision with pair_type)
}
```

**Ordering convention (enforced by validator)**: For `len(image_path)==2`, index 0 = earlier/pre/optical, index 1 = later/post/SAR. Never mixed.

**Validator**: A `jsonschema`-based validator written **first**, before any tier preprocessing code. Every tier's output passes through it before merge.

---

## 5. `gsd_bucket` Assignment Rule (Explicit)

> **Datasets with known, uniform native GSD** (RSVQA-HR: 0.15m; SpaceNet6 optical: 0.5m; LEVIR-CD: 0.5m; SpaceNet6 SAR: 0.5m; SARDet-100K post-R5: 2–10m; BigEarthNet: 10m) use **literal tags** `[GSD:Xm]` for both native and proxy branches.<br>
> **Datasets with variable/unknown per-tile native GSD** (VRSBench, any mixed-source aerial imagery) use **categorical buckets** `VHR-native` / `CARTOSAT-proxy` instead.

No mixed enums — the rule is authoritative.

---

## 6. Task → Dataset → Agent-Tool Mapping

| PS Mandate | Fine-Tuning Datasets | Inference Specialist |
|---|---|---|
| Single-image VQA | RSVQA-HR, VRSBench, BigEarthNet (scene), Sen-2 LULC | `RS-VQA` |
| Captioning / grounding | VRSBench (both natively) | `RS-Caption` / `RS-Grounding` |
| Bi-temporal change VQA/description | CDVQA (primary), LEVIR-CD, OSCD | `Change-VQA`, `Change-Mask` (if masks) |
| Cross-modal optical–SAR | BigEarthNet (primary), SpaceNet6 pair, SARDet-100K | `Optical-SAR Fusion` |
| Agentic orchestration | — | Controller: classify → compatibility → registry → execute → assemble |

---

## 7. Training Curriculum (QDoRA on 2×T4, ~12–14 GPU-hrs)

**Effective corpus after R4 doubling**: ~200k samples (was ~170k pre-R4).

| Stage | Focus | Datasets | Samples (eff.) | Steps (est.) | GPU-hrs | Checkpoint |
|---|---|---|---|---|---|---|
| **1** | Domain adaptation | BigEarthNet (dominant) | ~80k | ~10k–14k | 4–5 | `stage1_adapter` |
| **2** | Single-image + resolution curriculum | VRSBench, RSVQA-HR (dual-res) | ~60k | ~7k–10k | 4–5 | `stage2_adapter` |
| **3** | Multi-image reasoning | CDVQA, LEVIR-CD, OSCD, SpaceNet6 pair, SARDet-100K | ~60k | ~7k–10k | 3–4 | `stage3_adapter` (final) |

- **Replay**: 10% previous-stage data mixed in each stage.
- **Budget**: ~25k–35k total optimizer steps, effective batch 8–16 (grad accum), **~1.5–2 epochs over 200k effective samples**.
- **Sessions**: 2–3 Kaggle 12-hour sessions with adapter checkpoint resume.
- **DoRA go/no-go**: Benchmark 50 steps QDoRA vs QLoRA at Stage 1 start; single config flag `use_dora: true/false`.
- **Internal validation**: 2–3% stratified slice from each tier's training data, tagged `split: "val_internal"`, monitored every eval step.

---

## 8. Kaggle Implementation Constraints (Hard Limits)

| Constraint | Value | Mitigation |
|---|---|---|
| Persistent output (`/kaggle/working`) | 20 GB | Write only final PNGs + JSONL here |
| Scratch space (`/kaggle/tmp`) | ~60 GB | Stream BigEarthNet HF shards here, filter in-memory, discard after |
| Dataset file count cap | 500 files | **Shard output into tar/zip archives (~2k files each) from the start** |
| Session wall | 12 hours | Checkpoint resume; idempotent preprocessing with manifest |
| GPU | 2× T4 (16 GB each) | Unsloth FastVisionModel QDoRA/QLoRA, grad accum |

---

## 9. Dataset Acquisition & Download Strategy

| Dataset | Source | Method | Notes |
|---|---|---|---|
| **BigEarthNet S2** | HF `lc-col/bigearthnet` (HDF5) | `huggingface_hub` streaming → `/kaggle/tmp` → filter → PNG → `/kaggle/working` | Select ~60k via CORINE/country stratification using `metadata.parquet` |
| **BigEarthNet S1** | Kaggle `javidtheimmortal/bigearthnetsentinel1` | `kagglehub.dataset_download` | Community mirror |
| **VRSBench** | Kaggle `ayaanmustafa/vrs-bench` or HF `xiang709/VRSBench` | `kagglehub` or `load_dataset` | 12.5 GB |
| **RSVQA-HR** | Kaggle `vishalravichandran/rsvqa-dataset` | `kagglehub` | **Verify**: USGS 15cm, not Dutch 8cm. Spot-check annotations. |
| **CDVQA + SECOND** | GitHub JSONs + Google Drive images | `gdown` for SECOND (2.3 GB), `requests` for CDVQA JSONs | **Load only `train`/`val` split files. Assert split names match exactly.** |
| **LEVIR-CD** | Kaggle `mdrifaturrahman33/levir-cd-change-detection` | `kagglehub` | 3.79 GB |
| **SpaceNet 6** | Kaggle `sandhiwangiyana/spacenet-6-multisensor-allweather-mapping` | `kagglehub` | 40 GB train split (optical+SAR) |
| **OSCD** | Kaggle `sumit07125/oscd-onera-satellite-change-detection` | `kagglehub` | 489 MB |
| **SARDet-100K** | Kaggle `greatbird/sardet-100k` | `kagglehub` | 7.25 GB |
| **Sen-2 LULC** | User-provided (Mendeley) | **Pre-mounted as Kaggle input** | No automation; user supplies link/data |

**Post-download sanity check (every dataset)**: Count files, spot-check 5–10 annotations against paper schema, verify split proportions. Fail fast if mirror is truncated/reshuffled.

---

## 10. Preprocessing Pipeline Architecture (Resumable, Checkpointable)

```
preprocess/
├── validator.py          # jsonschema validator (WRITTEN FIRST)
├── manifest.parquet      # Tracks: id, dataset, split, processed: bool, output_shard
├── common/
│   ├── io.py             # PNG write, tar shard write, manifest update
│   ├── gsd.py            # Bucket assignment (rule above), dual-res logic
│   ├── sar.py            # R3 pseudo-RGB, R5 downsample
│   ├── bbox.py           # R6 format conversion
│   └── stats.py          # R2 global percentile computation (two-pass)
├── tier0_bigen.py
├── tier1_vrsbench.py
├── tier1_rsvqa_hr.py
├── tier1_cdvqa.py        # Loads ONLY train/val split files; asserts names
├── tier1_levir_cd.py
├── tier1_sn6_opt.py
├── tier1_oscd.py
├── tier2_sardet.py
├── tier2_sn6_sar.py
├── tier3_sen2lulc.py
├── split_internal_val.py  # Carves 2-3% stratified val_internal from each tier's train
└── merge_and_package.py  # Concats JSONL, verifies schema, tars shards
```

**Manifest logic**: On start, load `manifest.parquet`. Skip any `id` with `processed=True`. After each sample/shard, append row and flush. Survives session restart.

**Output sharding**: Each tier writes PNGs into `/kaggle/working/<tier>/shard_XXX/` (≤2,000 files). `merge_and_package.py` tars each shard, writes final `dataset.jsonl`, produces `dataset.tar.gz` for Kaggle Dataset upload.

---

## 11. Critical Technical Specs (Verbatim for MiMo)

> **Bounding box format:** Normalized to `[0,1000]`, ordered as `(x_topleft, y_topleft), (x_bottomright, y_bottomright)` — Qwen convention. Not y-first.

> **GSD tagging:** Goes in a **system turn** (ChatML), not concatenated into user instruction. Uses discrete buckets per assignment rule — not raw floats.

> **`image_path` is always a list.** `len=1` for single-image, `len=2` for bitemporal/cross-modal. **Order is meaningful and fixed**: index 0 = earlier/pre/optical, index 1 = later/post/SAR.

> **`pair_type` field is mandatory** and drives template selection: `single` → 1-image ChatML, `bitemporal` → "Image 1 [pre], Image 2 [post]", `cross-modal` → "Image 1 [optical], Image 2 [SAR]".

> **`modality` value for paired data is `"optical+sar"`** — not `"cross-modal"` — to avoid collision with `pair_type`.

> **Internal validation split**: 2–3% stratified from each tier's training data, tagged `split: "val_internal"`, excluded from training sampler, used for in-loop metric monitoring.

---

## 12. Known Limitations (Documented)

1. **No SAR-only bi-temporal change data** — OSCD-SAR dropped (not a packaged dataset). Documented as explicit gap.
2. **Community Kaggle mirrors** — Not guaranteed byte-identical to source. Sanity checks mandatory.
3. **BigEarthNet streaming** — Depends on HF Hub availability/rate limits. Manifest resumability essential.
4. **DoRA unoptimized** — May be slower than QLoRA. Benchmark at Stage 1 start.
5. **Sen-2 LULC** — Requires manual Mendeley download; notebook assumes pre-mounted input.

---

## 13. Handoff Checklist for MiMo

- [x] Write `validator.py` with frozen JSONL schema (including `pair_type`, ordered `image_path`, `gsd_bucket` rule, `split`, `modality` enum with `"optical+sar"`), plus the empty-bbox-on-grounding gap closed
- [x] Write `common/stats.py` for two-pass global percentile computation (R2)
- [x] Write `common/sar.py` with R3 physics + R5 downsample, plus `sar_intensity_pseudo_gray()` for genuinely single-pol sources (no fabricated second channel)
- [x] Write `common/bbox.py` with R6 Qwen format
- [x] Write `common/gsd.py` with bucket assignment rule + curriculum schedule. `DUAL_RESOLUTION_DATASETS` resolved to {vrsbench, rsvqa_hr, levir_cd, sn6_opt} — every Tier 1 row whose Treatment column says "R4", and only those
- [x] Write `common/sample.py` — shared stratified_sample/quantile_bucket, used everywhere a Tier table cell says "stratified"
- [x] Implement `common/io.py` (Manifest, write_png, append_jsonl, ShardWriter)
- [x] Implement `common/concat.py` (R7 horizontal concatenation utility)
- [x] Implement `tier0_bigen.py` — real metadata.parquet + s2_npy/s1_npy loading (not a metadata-only stub); emits optical, +SAR, +cross-modal fusion rows per patch when S1 is present (Mandate 4)
- [x] Implement `tier1_oscd.py`
- [x] Implement `tier1_vrsbench.py`
- [x] Implement `tier1_rsvqa_hr.py` — R4 wired (was imported, unused), non-PNG sources converted (R7)
- [x] Implement `tier1_cdvqa.py`
- [x] Implement `tier1_levir_cd.py` — R4 wired; real change-magnitude-decile stratified selection (was plain random.sample)
- [x] Implement `tier1_sn6_opt.py` — R4 wired; real building-density-quartile stratified selection; bbox list and reported building count now agree; writes `selected_tile_ids.json` for SAR pairing
- [x] Implement `tier2_sardet.py` — real class-uniform stratified selection; honest single-channel dB render (no fabricated VH = VV*0.8)
- [x] Implement `tier2_sn6_sar.py` — uses the real `sar_pseudo_rgb`/`sar_intensity_pseudo_gray` physics (was reimplemented inline, bypassing R3 entirely); paired 1:1 with `tier1_sn6_opt.py`'s selection via `--optical-dir`, emitting cross-modal fusion rows
- [x] Implement `tier3_sen2lulc.py` — `sample_fraction` now actually subsamples (was a dead parameter — risked processing all ~213k entries against the 20GB Kaggle output cap); real class-uniform stratification; fixed `" Masks"` typo; fixed CSV string labels ("2") not resolving to class names
- [x] Implement `merge_and_package.py` — fixed the bug where it always skipped `_train.jsonl`/`_val_internal.jsonl` and re-read the pre-split original, silently dropping every val_internal row; sharded tarring via `ShardWriter` (≤2000 files/shard) instead of one monolithic tar.gz; duplicate ids are now actually dropped, not just reported
- [x] Write `split_internal_val.py`
- [x] Add post-download sanity checks for every dataset — now actually raise `SanityCheckError` on missing/empty structure instead of unconditionally reporting "pass"
- [x] CDVQA loader: asserts split filenames match `train`/`val` exactly; rejects any `test`/`test2` entries
- [x] Write Stage 1/2/3 training scripts with config-flag `use_dora`, checkpoint resume, 10% replay — **scaffolding only**, `train_step`/`validate` are marked stubs (NaN loss) and print a loud warning on `run()`; no real forward/backward pass is wired in
- [x] Add DoRA vs QLoRA benchmark script — memory/speed are real measurements; loss comparison honestly reports `"unmeasured"` rather than the old fake identical-formula loss curve that always "picked a winner"

### Known deferred items (not silently left broken — flagged here)

- `training/train.py` and `training/benchmark_dora.py` need a real ChatML-formatting + forward/backward pass wired against the actual Unsloth/Qwen3-VL API. That needs a live Kaggle GPU session to write and verify — don't trust either script's loss output until that lands.
- OSCD/LEVIR-CD/SpaceNet6 mask→bbox conversion keeps only the overall bounding box of all connected components, not one box per separate change region, even though the schema supports multiple boxes per sample.
- R4 proxy images (`create_proxy_sample`) relabel `gsd_bucket` but reuse the same source image for every dataset except VRSBench, which does a real bicubic downsample. Not pixel-accurate GSD simulation for rsvqa_hr/levir_cd/sn6_opt yet.
- `tier0_bigen.py`'s input contract (`metadata.parquet` + `s2_npy/`/`s1_npy/`) is this script's own Kaggle-side format, not raw BigEarthNet-MM HDF5 — adjust `load_bigen_arrays()` if the real acquisition notebook's HF stream produces different field names.

**Test suite**: 292/292 tests passing.
