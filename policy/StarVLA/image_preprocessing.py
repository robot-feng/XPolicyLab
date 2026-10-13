"""Resize already-decoded RGB arrays without altering channel order."""

from __future__ import annotations

import cv2
import numpy as np


BACKENDS = ("opencv_area", "pillow_bicubic")


def validate_resize_backend(value):
    if value not in BACKENDS:
        raise ValueError(f"image_resize_backend must be one of {BACKENDS}, got {value!r}")
    return value


def resize_rgb(image: np.ndarray, size, backend="opencv_area"):
    validate_resize_backend(backend)
    if len(size) != 2 or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in size):
        raise ValueError("image_size must contain two positive integers [width, height]")
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3 or not image.size:
        raise ValueError("resize_rgb requires a nonempty HWC uint8 RGB array")
    if backend == "opencv_area":
        return cv2.resize(image, tuple(size), interpolation=cv2.INTER_AREA)
    # Pillow is part of the StarVLA model environment; keep it out of core
    # imports. BICUBIC is the genuine loader's default for RGB PIL images.
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("pillow_bicubic requires Pillow in the policy environment") from exc
    return np.asarray(Image.fromarray(image).resize(tuple(size), resample=Image.Resampling.BICUBIC)).copy()
