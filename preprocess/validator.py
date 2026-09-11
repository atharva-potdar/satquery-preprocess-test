"""SatQuery JSONL schema validator.

Validates every training/inference sample against the frozen schema (Section 4)
and enforces cross-field consistency rules.

Public API:
    validate_sample(sample) -> (is_valid, errors)
    validate_jsonl(path) -> (count, errors)
    cross_field_checks(sample) -> list[str]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator

# ---------------------------------------------------------------------------
# Frozen JSONL schema — Section 4 of the v4 specification
# ---------------------------------------------------------------------------

# Bbox uses oneOf: null | array-of-arrays.  The [0,1000] range constraint on
# each element is enforced by _validate_bbox_values() below because Draft7
# does not support prefixItems (2020-12 feature).
_BBOX_SCHEMA = {
    "oneOf": [
        {"type": "null"},
        {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 4,
                "maxItems": 4,
            },
        },
    ]
}

SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "id",
        "dataset",
        "task",
        "image_path",
        "pair_type",
        "gsd_bucket",
        "split",
        "instruction",
        "response",
        "bbox",
        "modality",
    ],
    "properties": {
        "id": {"type": "string", "pattern": "^[a-z0-9_]+$"},
        "dataset": {
            "enum": [
                "bigen",
                "vrsbench",
                "rsvqa_hr",
                "cdvqa",
                "levir_cd",
                "sn6_opt",
                "oscd",
                "sardet",
                "sn6_sar",
                "sen2lulc",
            ]
        },
        "task": {
            "enum": [
                "vqa",
                "caption",
                "grounding",
                "change_vqa",
                "change_grounding",
                "fusion_vqa",
                "fusion_grounding",
            ]
        },
        "image_path": {
            "type": "array",
            "minItems": 1,
            "maxItems": 2,
            "items": {"type": "string", "minLength": 1},
        },
        "pair_type": {"enum": ["single", "bitemporal", "cross-modal"]},
        "gsd_bucket": {"type": "string", "minLength": 1},
        "split": {"enum": ["train", "val_internal"]},
        "instruction": {"type": "string", "minLength": 1},
        "response": {"type": "string", "minLength": 1},
        "bbox": _BBOX_SCHEMA,
        "modality": {"enum": ["optical", "sar", "optical+sar"]},
    },
    "additionalProperties": False,
}

_validator = Draft7Validator(SCHEMA)


def _validate_bbox_values(bbox: list[list[float]]) -> list[str]:
    """Validate that each bbox coordinate is in [0, 1000]."""
    errors: list[str] = []
    for i, box in enumerate(bbox):
        if len(box) != 4:
            errors.append(f"bbox[{i}] must have exactly 4 elements, got {len(box)}")
            continue
        for j, val in enumerate(box):
            if not isinstance(val, (int, float)):
                errors.append(f"bbox[{i}][{j}] must be a number, got {type(val).__name__}")
            elif val < 0 or val > 1000:
                errors.append(
                    f"bbox[{i}][{j}]={val} out of range [0, 1000]"
                )
    return errors


# ---------------------------------------------------------------------------
# Enum sets for cross-field checks
# ---------------------------------------------------------------------------

_GROUNDING_TASKS = {"grounding", "change_grounding", "fusion_grounding"}
_VQA_LIKE_TASKS = {"vqa", "caption", "change_vqa", "fusion_vqa"}
MultiImagePairTypes = {"bitemporal", "cross-modal"}


# ---------------------------------------------------------------------------
# Cross-field consistency checks (beyond schema validation)
# ---------------------------------------------------------------------------


def cross_field_checks(sample: dict[str, Any]) -> list[str]:
    """Validate consistency rules between fields.

    Rules enforced:
        R-pair  pair_type=single  <->  len(image_path)==1
        R-pair  pair_type in {bitemporal,cross-modal} <-> len(image_path)==2
        R-bbox  grounding/change_grounding/fusion_grounding -> bbox is non-null
        R-bbox  vqa/caption/change_vqa/fusion_vqa -> bbox is null
        R-bbox  change_grounding requires pair_type=bitemporal AND len(image_path)==2
        R-bbox  fusion_grounding requires pair_type=cross-modal AND len(image_path)==2
        R-mod   pair_type=cross-modal -> modality == "optical+sar"
    """
    errors: list[str] = []
    pair_type = sample.get("pair_type")
    image_path = sample.get("image_path", [])
    task = sample.get("task")
    bbox = sample.get("bbox")
    modality = sample.get("modality")
    n_images = len(image_path)

    # R-pair: pair_type <-> image_path length
    if pair_type == "single" and n_images != 1:
        errors.append(
            f"pair_type='single' requires exactly 1 image, got {n_images}"
        )
    if pair_type in MultiImagePairTypes and n_images != 2:
        errors.append(
            f"pair_type='{pair_type}' requires exactly 2 images, got {n_images}"
        )

    # R-bbox: task <-> bbox presence
    if task in _GROUNDING_TASKS and bbox is None:
        errors.append(
            f"task='{task}' requires non-null bbox, got null"
        )
    if task in _VQA_LIKE_TASKS and bbox is not None:
        errors.append(
            f"task='{task}' requires null bbox, got non-null"
        )

    # R-bbox: change_grounding -> bitemporal + 2 images
    if task == "change_grounding":
        if pair_type != "bitemporal":
            errors.append(
                f"task='change_grounding' requires pair_type='bitemporal', "
                f"got pair_type='{pair_type}'"
            )
        if n_images != 2:
            errors.append(
                f"task='change_grounding' requires 2 images, got {n_images}"
            )

    # R-bbox: fusion_grounding -> cross-modal + 2 images
    if task == "fusion_grounding":
        if pair_type != "cross-modal":
            errors.append(
                f"task='fusion_grounding' requires pair_type='cross-modal', "
                f"got pair_type='{pair_type}'"
            )
        if n_images != 2:
            errors.append(
                f"task='fusion_grounding' requires 2 images, got {n_images}"
            )

    # R-mod: cross-modal -> optical+sar
    if pair_type == "cross-modal" and modality != "optical+sar":
        errors.append(
            f"pair_type='cross-modal' requires modality='optical+sar', "
            f"got modality='{modality}'"
        )

    return errors


# ---------------------------------------------------------------------------
# Public validation API
# ---------------------------------------------------------------------------


def validate_sample(sample: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate a single JSONL row against schema + cross-field rules.

    Returns (is_valid, list_of_error_messages).
    """
    errors: list[str] = []

    # Schema validation — include field path for clarity
    for err in _validator.iter_errors(sample):
        path = ".".join(str(p) for p in err.absolute_path) or "<root>"
        errors.append(f"schema [{path}]: {err.message}")

    # Bbox value range check (Draft7 can't express [0,1000] per-element)
    bbox = sample.get("bbox")
    if isinstance(bbox, list) and not errors:
        errors.extend(_validate_bbox_values(bbox))

    # Cross-field checks (only if schema passed — fields may be missing)
    if not errors:
        errors.extend(cross_field_checks(sample))

    return (len(errors) == 0, errors)


def validate_jsonl(path: str | Path) -> tuple[int, list[str]]:
    """Validate every line in a JSONL file.

    Returns (total_lines, list_of_all_errors_with_line_prefix).
    """
    path = Path(path)
    all_errors: list[str] = []
    count = 0

    with open(path, "r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            count += 1
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                all_errors.append(f"line {line_num}: JSON parse error: {exc}")
                continue

            valid, errs = validate_sample(sample)
            if not valid:
                for e in errs:
                    all_errors.append(f"line {line_num}: {e}")

    return (count, all_errors)
