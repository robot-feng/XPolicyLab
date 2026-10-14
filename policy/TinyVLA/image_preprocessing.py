"""TinyVLA's HDF5 image boundary, using XPolicyLab's shared RGB codec."""

import cv2
import numpy as np

from XPolicyLab.utils.process_data import decode_image_bit


def read_camera_images(images, camera_names, frame_index, imsize):
    """Decode one RGB frame per camera before inspecting dimensions or resizing.

    Preserve TinyVLA's existing resize rule: widths unequal to ``imsize``
    resize to 320x180 with OpenCV's default linear interpolation. The HDF5
    ``compress`` attribute does not determine byte color order; the shared
    decoder owns the marker and legacy container handling.
    """
    if isinstance(imsize, (bool, np.bool_)) or not isinstance(imsize, (int, np.integer)) or imsize < 1:
        raise ValueError("TinyVLA imsize must be a positive integer")
    decoded = {}
    for name in camera_names:
        try:
            frame = np.asarray(decode_image_bit(images[name][frame_index]))
            # One selected HDF5 frame may retain an older singleton container.
            # A genuine multi-frame value is rejected instead of taking frame 0.
            if frame.ndim == 4 and frame.shape[0] == 1:
                frame = frame[0]
            if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8 or min(frame.shape[:2]) < 1:
                raise ValueError(f"expected one HWC uint8 RGB frame, got {frame.shape}/{frame.dtype}")
            if imsize != frame.shape[1]:
                frame = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_LINEAR)
            decoded[name] = np.ascontiguousarray(frame)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"TinyVLA camera {name!r}, frame {frame_index}: {exc}") from exc
    return decoded
