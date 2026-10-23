"""CenterNet-style dense targets + Johnson-criteria gate targets."""
import math

import numpy as np

STRIDE = 8


def johnson_rgb_weight(px, mid=10.4, tau=2.0, floor=0.1):
    """Target share of attention for the RGB (appearance) stream for a target spanning `px` pixels.

    Johnson criteria (line pairs across the critical dimension, 1 lp ~ 2 px):
        detection 1 lp (2 px) | orientation 1.4 lp (2.8 px) | recognition 4 lp (8 px) | identification 6.4 lp (12.8 px)
    Below recognition there is too little appearance to rely on -> motion dominates; above identification
    appearance is informative -> RGB dominates. A sigmoid centred between the two encodes that prior.
    """
    s = 1.0 / (1.0 + math.exp(-(px - mid) / max(tau, 1e-6)))
    return floor + (1.0 - 2.0 * floor) * s


def gaussian_radius(h, w, min_overlap=0.7):
    a1, b1 = 1, h + w
    c1 = w * h * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + math.sqrt(b1 ** 2 - 4 * a1 * c1)) / 2
    a2, b2 = 4, 2 * (h + w)
    c2 = (1 - min_overlap) * w * h
    r2 = (b2 + math.sqrt(b2 ** 2 - 4 * a2 * c2)) / 2
    a3, b3 = 4 * min_overlap, -2 * min_overlap * (h + w)
    c3 = (min_overlap - 1) * w * h
    r3 = (b3 + math.sqrt(b3 ** 2 - 4 * a3 * c3)) / 2
    return min(r1, r2, r3)


def draw_gaussian(hm, cx, cy, r):
    d = 2 * r + 1
    sigma = d / 6
    ys, xs = np.ogrid[-r:r + 1, -r:r + 1]
    g = np.exp(-(xs * xs + ys * ys) / (2 * sigma * sigma + 1e-12))
    H, W = hm.shape
    l, ri = min(cx, r), min(W - cx, r + 1)
    t, b = min(cy, r), min(H - cy, r + 1)
    region = hm[cy - t:cy + b, cx - l:cx + ri]
    gp = g[r - t:r + b, r - l:r + ri]
    if region.size and gp.size:
        np.maximum(region, gp, out=region)


def build_targets(boxes, classes, num_classes, input_size, max_objs, gate_cfg, stride=STRIDE):
    """boxes: (n,4) x1y1x2y2 in INPUT pixels. Returns dict of numpy arrays."""
    H = W = input_size // stride
    hm = np.zeros((num_classes, H, W), np.float32)
    wh = np.zeros((max_objs, 2), np.float32)
    off = np.zeros((max_objs, 2), np.float32)
    ind = np.zeros((max_objs,), np.int64)
    mask = np.zeros((max_objs,), np.float32)
    gate_t = np.zeros((max_objs,), np.float32)
    k = 0
    for b, c in zip(boxes, classes):
        if k >= max_objs:
            break
        bw, bh = float(b[2] - b[0]), float(b[3] - b[1])
        if bw <= 0.5 or bh <= 0.5:
            continue
        cx, cy = (b[0] + b[2]) / 2 / stride, (b[1] + b[3]) / 2 / stride
        ix, iy = int(np.clip(cx, 0, W - 1)), int(np.clip(cy, 0, H - 1))
        r = max(0, int(gaussian_radius(math.ceil(bh / stride), math.ceil(bw / stride))))
        draw_gaussian(hm[int(c)], ix, iy, r)
        wh[k] = (bw / stride, bh / stride)
        off[k] = (cx - ix, cy - iy)
        ind[k] = iy * W + ix
        mask[k] = 1.0
        gate_t[k] = johnson_rgb_weight(math.sqrt(bw * bh), gate_cfg.get("johnson_mid_px", 10.4),
                                       gate_cfg.get("johnson_tau", 2.0), gate_cfg.get("floor", 0.1))
        k += 1
    return {"hm": hm, "wh": wh, "off": off, "ind": ind, "mask": mask, "gate_t": gate_t}
