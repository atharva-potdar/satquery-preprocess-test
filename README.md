# SatQuery AI

An agentic vision-language assistant for remote sensing imagery analysis with single/paired images, cross-modal, and bi-temporal change detection capabilities.

## Datasets

### OSCD (Onera Satellite Change Detection)
- **Purpose:** Bi-temporal change detection training/evaluation
- **Source:** https://github.com/cdsdata/OSCD
- **Location:** `data/oscd/`
- **Structure:** 24 location pairs, each with pre-event/post-event images and change masks
- **Bands:** B02 (Blue), B03 (Green), B04 (Red), B08 (NIR) from Sentinel-2
- **Download:** Clone repository or download from linked source

### VRSBench
- **Purpose:** Visual remote sensing benchmark for captioning, VQA, and grounding
- **Source:** https://huggingface.co/datasets/Shitao/VRSBench
- **Location:** `data/vrsbench/`
- **Files:**
  - `VRSBench_train.json` - Training data (142,390 samples, LLaVA format)
  - `VRSBench_EVAL_*.json` - Evaluation sets (caption, VQA, referring)
  - `Images_train.zip` - Training images (20,264 PNGs, 512×512)
  - `Images_val.zip` - Validation images
  - `Annotations_train.zip` / `Annotations_val.zip` - Per-image annotations (not used directly)

### Additional Datasets (Referenced in Preprocessing)
- **RSVQA-HR:** High-resolution remote sensing VQA
- **SpaceNet 6 (SN6):** Building footprint extraction
- **BigEarthNet:** Multi-label land cover classification
- **SarDet:** SAR object detection

## Data Setup

```bash
# Clone/download datasets to data/ directory
# OSCD
cd data/oscd
git clone https://github.com/cdsdata/OSCD.git .

# VRSBench (from HuggingFace)
cd data/vrsbench
# Download VRSBench_train.json and image zips
```

## Preprocessing

```bash
# Run tier-specific preprocessing
python -m preprocess.tier1_oscd --input-dir data/oscd --output-dir data/oscd_output
python -m preprocess.tier1_vrsbench --input-dir data/vrsbench --output-dir data/vrsbench_output
```

## Testing

```bash
pytest tests/ -v
```

## Project Structure

```
satquery/
├── app/                    # Gradio web interface
├── configs/                # Path configuration
├── data/                   # Dataset storage (gitignored)
├── models/                 # Model definitions
├── preprocess/             # Data preprocessing scripts
│   ├── common/             # Shared utilities (bbox, stats, sar, gsd, concat, io)
│   ├── tier1_oscd.py       # OSCD preprocessing
│   └── tier1_vrsbench.py   # VRSBench preprocessing
├── tests/                  # Test suite
├── training/               # Training scripts
└── requirements.txt        # Dependencies
```
