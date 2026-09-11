"""R7: 1×2 spatial concatenation for bi-temporal/cross-modal image pairs.

Generates a side-by-side concatenated PNG for visual-evidence display
(inference-time, Gradio gallery). Not called by tier scripts during
training preprocessing — only used at inference/GUI time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def concat_horizontal(
    img_left: np.ndarray | Image.Image,
    img_right: np.ndarray | Image.Image,
    *,
    target_height: int | None = None,
    separator_width: int = 2,
    separator_color: int = 128,
) -> Image.Image:
    """Concatenate two images side-by-side (1×2 horizontal).

    Args:
        img_left: Left image (H, W, 3) uint8 ndarray or PIL Image.
        img_right: Right image (H, W, 3) uint8 ndarray or PIL Image.
        target_height: If set, resize both images to this height before
            concatenating. If None, uses left image's height and resizes
            right to match.
        separator_width: Width of the vertical separator line in pixels.
        separator_color: Grayscale value for the separator (0-255).

    Returns:
        PIL Image of shape (H, W_left + separator + W_right, 3) uint8.
    """
    if isinstance(img_left, Image.Image):
        img_left = np.array(img_left)
    if isinstance(img_right, Image.Image):
        img_right = np.array(img_right)

    if img_left.ndim != 3 or img_right.ndim != 3:
        raise ValueError(f"Expected 3-channel images, got ndim={img_left.ndim} and {img_right.ndim}")
    if img_left.shape[2] != 3 or img_right.shape[2] != 3:
        raise ValueError(f"Expected 3 channels, got {img_left.shape[2]} and {img_right.shape[2]}")

    left_pil = Image.fromarray(img_left)
    right_pil = Image.fromarray(img_right)

    if target_height is not None:
        # Resize both to target height, preserving aspect ratio
        l_ratio = target_height / left_pil.height
        left_pil = left_pil.resize(
            (max(1, int(left_pil.width * l_ratio)), target_height), Image.BILINEAR
        )
        r_ratio = target_height / right_pil.height
        right_pil = right_pil.resize(
            (max(1, int(right_pil.width * r_ratio)), target_height), Image.BILINEAR
        )
    else:
        # Resize right to match left's height
        if right_pil.height != left_pil.height:
            ratio = left_pil.height / right_pil.height
            right_pil = right_pil.resize(
                (max(1, int(right_pil.width * ratio)), left_pil.height), Image.BILINEAR
            )

    canvas_width = left_pil.width + separator_width + right_pil.width
    canvas_height = left_pil.height

    canvas = Image.new("RGB", (canvas_width, canvas_height), (separator_color,) * 3)
    canvas.paste(left_pil, (0, 0))
    canvas.paste(right_pil, (left_pil.width + separator_width, 0))

    return canvas


def save_concat(
    left_path: Path,
    right_path: Path,
    output_path: Path,
    **kwargs,
) -> None:
    """Load two images and save their 1×2 concatenation.

    Args:
        left_path: Path to left/before image.
        right_path: Path to right/after image.
        output_path: Path to save concatenated PNG.
        **kwargs: Passed to concat_horizontal.
    """
    left = Image.open(left_path)
    right = Image.open(right_path)
    result = concat_horizontal(left, right, **kwargs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(str(output_path))
