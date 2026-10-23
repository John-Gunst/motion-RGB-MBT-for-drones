"""Frozen-backbone feature cache.

Backbones never receive gradients, so their outputs for a given frame are constant. Caching them
(fp16 memmaps, one per sequence and feature level) removes the backbone from every training step:
only the neck, fusion block and heads run, which is what makes 72-epoch schedules affordable on a
student GPU. Layout:  <cache_dir>/<key>/<split>/<seq>/{rgb_l2,rgb_l3,mot_l2,mot_l3}.npy + index.json
"""
import json
from pathlib import Path

import numpy as np
import torch

FEATS = ("rgb_l2", "rgb_l3", "mot_l2", "mot_l3")


def cache_key(cfg):
    m = cfg.data.motion
    return f"{cfg.model.backbone}_{cfg.data.input_size}_{m.get('mode', 'causal')}_g{m.get('gain', 4.0)}_s{int(bool(m.get('stabilize', False)))}"


def cache_root(cfg, split):
    return Path(cfg.data.cache_dir) / cache_key(cfg) / split


class SequenceCache:
    def __init__(self, d):
        self.dir = Path(d)
        self.index = json.loads((self.dir / "index.json").read_text())
        self.frames = self.index["frames"]
        self.row_of = {t: i for i, t in enumerate(self.frames)}
        self.scale = self.index["scale"]
        self._mm = None

    def _open(self):
        if self._mm is None:  # opened lazily so each DataLoader worker gets its own handle
            self._mm = {k: np.load(self.dir / f"{k}.npy", mmap_mode="r") for k in FEATS}
        return self._mm

    def get(self, t, hflip=False):
        mm = self._open()
        r = self.row_of[t]
        out = {}
        for k in FEATS:
            a = np.array(mm[k][r])
            out[k] = torch.from_numpy(a)
        return out


def is_complete(d):
    p = Path(d) / "index.json"
    if not p.exists():
        return False
    return json.loads(p.read_text()).get("complete", False)
