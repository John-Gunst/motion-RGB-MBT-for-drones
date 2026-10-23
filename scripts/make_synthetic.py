#!/usr/bin/env python
"""Synthetic sky sequences with moving drones, birds and bugs — for verifying the pipeline end to end
before spending GPU hours on real data (it is NOT a substitute for real training data).

    python scripts/make_synthetic.py --root data/synthetic --sequences 12 --frames 60
    python scripts/train.py --set data.root=data/synthetic ...
"""
import argparse
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np

CLASSES = ["drone", "bird", "bug"]


def sky(w, h, rng):
    top = np.array([rng.uniform(150, 220), rng.uniform(120, 180), rng.uniform(60, 110)])
    bot = np.array([rng.uniform(220, 250), rng.uniform(200, 235), rng.uniform(170, 220)])
    t = np.linspace(0, 1, h)[:, None, None]
    img = (top * (1 - t) + bot * t) * np.ones((1, w, 1))
    noise = cv2.GaussianBlur(rng.normal(0, 1, (h // 8, w // 8)).astype(np.float32), (0, 0), 3)
    noise = cv2.resize(noise, (w, h))
    img += (noise / (np.abs(noise).max() + 1e-6) * 25)[..., None]
    # a treeline so a panning camera gives real background motion
    xs = np.arange(w)
    line = (h * 0.85 + 12 * np.sin(xs / 37.0) + 8 * np.sin(xs / 11.0)).astype(int)
    for x in xs:
        img[line[x]:, x] = (40, 70, 45)
    return np.clip(img, 0, 255).astype(np.uint8)


def draw_drone(img, cx, cy, s, t):
    c = (40, 40, 40)
    a = s / 2
    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        cv2.line(img, (int(cx), int(cy)), (int(cx + dx * a), int(cy + dy * a * 0.6)), c, max(1, int(s / 12)))
        cv2.circle(img, (int(cx + dx * a), int(cy + dy * a * 0.6)), max(1, int(s / 6)), (70, 70, 70), -1)
    cv2.rectangle(img, (int(cx - s / 6), int(cy - s / 10)), (int(cx + s / 6), int(cy + s / 10)), (20, 20, 20), -1)
    return cx - a - s / 6, cy - 0.6 * a - s / 6, cx + a + s / 6, cy + 0.6 * a + s / 6


def draw_bird(img, cx, cy, s, t):
    flap = math.sin(t * 0.9) * s * 0.35
    c = (35, 30, 30)
    p = np.array([[cx - s / 2, cy - flap], [cx - s / 6, cy + s * 0.08], [cx, cy], [cx + s / 6, cy + s * 0.08],
                  [cx + s / 2, cy - flap]], np.int32)
    cv2.polylines(img, [p], False, c, max(1, int(s / 10)), cv2.LINE_AA)
    cv2.ellipse(img, (int(cx), int(cy + s * 0.05)), (max(1, int(s / 8)), max(1, int(s / 14))), 0, 0, 360, c, -1)
    return cx - s / 2, min(cy - abs(flap), cy) - 2, cx + s / 2, cy + max(abs(flap), s * 0.12) + 2


def draw_bug(img, cx, cy, s, t):
    cv2.ellipse(img, (int(cx), int(cy)), (max(1, int(s / 2)), max(1, int(s / 3))), (t * 40) % 180, 0, 360,
                (25, 25, 15), -1)
    return cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2


DRAW = [draw_drone, draw_bird, draw_bug]
SIZE = [(4, 70), (5, 60), (2, 9)]


def make_sequence(d, n_frames, w, h, rng, pan):
    bg = sky(w + 200, h, rng)
    objs = []
    for tid in range(1, rng.randint(1, 4) + 1):
        c = rng.randint(0, 2)
        s = rng.uniform(*SIZE[c])
        objs.append({"cls": c, "tid": tid, "s": s, "x": rng.uniform(0.1, 0.9) * w, "y": rng.uniform(0.1, 0.7) * h,
                     "vx": rng.uniform(-4, 4), "vy": rng.uniform(-2, 2), "ds": rng.uniform(-0.15, 0.15)})
    (d / "frames").mkdir(parents=True, exist_ok=True)
    anns = {}
    for t in range(n_frames):
        ox = int(100 + (80 * math.sin(t / 25.0) if pan else 0))
        img = bg[:, ox:ox + w].copy()
        img = np.clip(img.astype(np.int16) + rng.normal(0, 2, img.shape).astype(np.int16), 0, 255).astype(np.uint8)
        boxes = []
        for o in objs:
            if o["cls"] == 2:  # bugs move erratically
                o["vx"] += rng.uniform(-1.5, 1.5)
                o["vy"] += rng.uniform(-1.5, 1.5)
                o["vx"], o["vy"] = np.clip(o["vx"], -6, 6), np.clip(o["vy"], -6, 6)
            o["x"] += o["vx"]
            o["y"] += o["vy"]
            o["s"] = float(np.clip(o["s"] + o["ds"], SIZE[o["cls"]][0], SIZE[o["cls"]][1]))
            if not (0 < o["x"] < w) or not (0 < o["y"] < h * 0.8):
                o["vx"], o["vy"] = -o["vx"], -o["vy"]
            x1, y1, x2, y2 = DRAW[o["cls"]](img, o["x"], o["y"], o["s"], t)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            if x2 - x1 > 1 and y2 - y1 > 1:
                boxes.append([round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1), o["cls"], o["tid"]])
        cv2.imwrite(str(d / "frames" / f"{t:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if boxes:
            anns[str(t)] = boxes
    (d / "annotations.json").write_text(json.dumps({"width": w, "height": h, "classes": CLASSES, "frames": anns}))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/synthetic")
    ap.add_argument("--sequences", type=int, default=12)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng_py = random.Random(a.seed)
    root = Path(a.root)
    names = []
    for i in range(a.sequences):
        rng = np.random.default_rng(a.seed + i)

        class R:  # tiny adapter so the drawing code can use one rng
            uniform = staticmethod(lambda lo, hi: float(rng.uniform(lo, hi)))
            randint = staticmethod(lambda lo, hi: int(rng.integers(lo, hi + 1)))
            normal = staticmethod(lambda m, s, shape: rng.normal(m, s, shape))

        name = f"syn_{i:03d}"
        make_sequence(root / "sequences" / name, a.frames, a.width, a.height, R, pan=(i % 3 == 0))
        names.append(name)
    rng_py.shuffle(names)
    n_va = max(1, a.sequences // 6)
    split = {"val": names[:n_va], "test": names[n_va:2 * n_va], "train": names[2 * n_va:]}
    (root / "splits").mkdir(parents=True, exist_ok=True)
    for k, v in split.items():
        (root / "splits" / f"{k}.txt").write_text("\n".join(sorted(v)) + "\n")
    print(f"wrote {a.sequences} sequences to {root} ({', '.join(f'{k}={len(v)}' for k, v in split.items())})")


if __name__ == "__main__":
    main()
