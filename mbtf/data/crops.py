"""High-resolution square crops (RGB + motion) around targets for Tier-2 classification."""
import cv2
import numpy as np

from .motion import current_slot, three_frame_motion
from .transforms import normalize


def crop_square(img, cx, cy, side):
    h, w = img.shape[:2]
    side = int(round(side))
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1, y1 = x0 + side, y0 + side
    pad = [max(0, -y0), max(0, y1 - h), max(0, -x0), max(0, x1 - w)]
    crop = img[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    if any(pad):
        crop = cv2.copyMakeBorder(crop, *pad, cv2.BORDER_REPLICATE)
    return crop


def crop_triplet(frames_bgr, box, crop_size, motion_cfg, context=1.5, min_side=24, jitter=None, hflip=False):
    """frames_bgr: original-resolution triplet; box x1y1x2y2 in original pixels.

    jitter: optional (dx, dy, ds) fractions for training-time augmentation.
    Returns (rgb_tensor, motion_tensor).
    """
    x1, y1, x2, y2 = box[:4]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(max(x2 - x1, y2 - y1) * context, min_side)
    if jitter is not None:
        dx, dy, ds = jitter
        cx += dx * side
        cy += dy * side
        side *= ds
    crops = []
    for f in frames_bgr:
        c = crop_square(f, cx, cy, side)
        interp = cv2.INTER_AREA if c.shape[0] > crop_size else cv2.INTER_LINEAR
        crops.append(cv2.resize(c, (crop_size, crop_size), interpolation=interp))
    mode = motion_cfg.get("mode", "causal")
    mot = three_frame_motion(crops, mode=mode, gain=motion_cfg.get("gain", 4.0), blur=motion_cfg.get("blur", 3),
                             stabilize=False)
    rgb = cv2.cvtColor(crops[current_slot(mode)], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if hflip:
        rgb, mot = rgb[:, ::-1], mot[:, ::-1]
    return normalize(rgb), normalize(mot)
