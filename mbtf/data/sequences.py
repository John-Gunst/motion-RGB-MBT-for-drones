"""Unified on-disk format.

<root>/sequences/<name>/frames/000000.jpg ...
<root>/sequences/<name>/annotations.json
    {"width": W, "height": H, "classes": [...],
     "frames": {"0": [[x1, y1, x2, y2, class_id, track_id], ...], ...}}   # frames absent/empty = no targets
<root>/splits/{train,val,test}.txt   one sequence name per line
"""
import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


class Sequence:
    def __init__(self, root, name, class_names=None):
        self.name = name
        d = Path(root) / "sequences" / name
        self.dir = d
        self.frames = sorted(p for p in (d / "frames").iterdir() if p.suffix.lower() in IMG_EXT)
        meta = json.loads((d / "annotations.json").read_text())
        self.width, self.height = int(meta["width"]), int(meta["height"])
        remap = None
        if class_names is not None and "classes" in meta:
            # remap the sequence's class ids onto the config's class list by name
            remap = {}
            for i, n in enumerate(meta["classes"]):
                if n in class_names:
                    remap[i] = class_names.index(n)
        self._boxes = {}
        for k, v in meta.get("frames", {}).items():
            a = np.asarray(v, np.float32).reshape(-1, 6) if len(v) else np.zeros((0, 6), np.float32)
            if remap is not None and len(a):
                keep = np.array([int(c) in remap for c in a[:, 4]], bool)
                a = a[keep]
                a[:, 4] = [remap[int(c)] for c in a[:, 4]]
            self._boxes[int(k)] = a

    def __len__(self):
        return len(self.frames)

    def boxes(self, t):
        return self._boxes.get(t, np.zeros((0, 6), np.float32))

    def read(self, t):
        img = cv2.imread(str(self.frames[t]), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"cannot read {self.frames[t]}")
        return img


def read_split(root, split):
    p = Path(root) / "splits" / f"{split}.txt"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found — run scripts/prepare_data.py split --root {root}")
    return [l.strip() for l in p.read_text().splitlines() if l.strip()]


@lru_cache(maxsize=8)
def _load(root, split, classes):
    return [Sequence(root, n, list(classes)) for n in read_split(root, split)]


def load_split(root, split, classes):
    return _load(str(root), split, tuple(classes))
