import cv2
import numpy as np
import torch

from .motion import current_slot, three_frame_motion

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def letterbox(img, size, content_scale=1.0):
    """Resize the longer side to size*content_scale and pad bottom/right with zeros to size x size.

    Padding only at bottom/right means original coords = input coords / scale.
    """
    h, w = img.shape[:2]
    s = size * content_scale / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    r = cv2.resize(img, (nw, nh), interpolation=interp)
    out = np.zeros((size, size) + img.shape[2:], img.dtype)
    out[:nh, :nw] = r
    return out, s


def normalize(img01_hwc):
    """HxWx3 float RGB in [0,1] -> 3xHxW normalized tensor (ImageNet stats, also used for motion maps)."""
    x = (img01_hwc.astype(np.float32) - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


def preprocess_triplet(frames_bgr, size, motion_cfg, content_scale=1.0, hflip=False):
    """3 BGR frames (oldest->newest) -> (rgb_tensor, motion_tensor, scale)."""
    lb = []
    scale = 1.0
    for f in frames_bgr:
        o, scale = letterbox(f, size, content_scale)
        lb.append(o)
    mot = three_frame_motion(lb, mode=motion_cfg.get("mode", "causal"), gain=motion_cfg.get("gain", 4.0),
                             blur=motion_cfg.get("blur", 3), stabilize=motion_cfg.get("stabilize", False))
    cur = lb[current_slot(motion_cfg.get("mode", "causal"))]
    rgb = cv2.cvtColor(cur, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if hflip:
        rgb = rgb[:, ::-1]
        mot = mot[:, ::-1]
    return normalize(rgb), normalize(mot), scale


def denormalize_to_bgr(t):
    x = t.detach().cpu().numpy().transpose(1, 2, 0) * STD + MEAN
    return cv2.cvtColor((np.clip(x, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
