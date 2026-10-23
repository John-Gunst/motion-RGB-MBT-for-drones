import math
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from .cache import SequenceCache, cache_root, is_complete
from .crops import crop_triplet
from .motion import triplet_indices
from .sequences import load_split
from .targets import build_targets


def _frame_items(seqs, step):
    return [(si, t) for si, s in enumerate(seqs) for t in range(0, len(s), max(1, int(step)))]


class DetectionDataset(Dataset):
    """One sample = frame t of a sequence (+ its two motion partners), with dense CenterNet targets.

    cached=True reads frozen-backbone features from scripts/cache_features.py instead of images.
    """

    def __init__(self, cfg, split, cached=False, train=False, max_frames=None):
        self.cfg, self.split, self.cached, self.train = cfg, split, cached, train
        self.seqs = load_split(cfg.data.root, split, cfg.classes)
        self.C = len(cfg.classes)
        self.size = cfg.data.input_size
        step = cfg.data.frame_step_train if split == "train" else cfg.data.frame_step_eval
        if cached:
            root = cache_root(cfg, split)
            self.caches = []
            for s in self.seqs:
                d = root / s.name
                if not is_complete(d):
                    raise FileNotFoundError(f"feature cache missing/incomplete for {d}; run scripts/cache_features.py")
                self.caches.append(SequenceCache(d))
            self.items = [(si, t) for si, c in enumerate(self.caches) for t in c.frames]
        else:
            self.items = _frame_items(self.seqs, step)
        if max_frames and len(self.items) > max_frames:
            rng = random.Random(0)
            self.items = sorted(rng.sample(self.items, max_frames))

    def __len__(self):
        return len(self.items)

    def sample_weights(self):
        """Tier-1 curriculum: oversample frames that contain small (<small_obj_px) targets."""
        boost = float(self.cfg.data.get("small_obj_oversample", 1.0))
        thr = float(self.cfg.data.get("small_obj_px", 10))
        w = np.ones(len(self.items), np.float64)
        if boost <= 1.0:
            return w
        for i, (si, t) in enumerate(self.items):
            s = self.seqs[si]
            b = s.boxes(t)
            if len(b):
                sc = self.size / max(s.width, s.height)
                px = np.sqrt(np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)) * sc
                if (px < thr).any():
                    w[i] = boost
        return w

    def __getitem__(self, i):
        si, t = self.items[i]
        s = self.seqs[si]
        cfg = self.cfg
        gt = s.boxes(t)
        sample = {}
        hflip = False
        if self.cached:
            c = self.caches[si]
            sample["feats"] = c.get(t)
            scale = c.scale
        else:
            from .transforms import preprocess_triplet
            content = 1.0
            if self.train:
                t1 = cfg.get("tier1", {})
                if random.random() < t1.get("zoom_prob", 0.0):
                    lo, hi = t1.get("zoom_out", [1.0, 1.0])
                    content = random.uniform(lo, hi)
                hflip = random.random() < t1.get("hflip_prob", 0.0)
            idx = triplet_indices(t, len(s), cfg.data.motion.get("mode", "causal"))
            frames = [s.read(j) for j in idx]
            rgb, mot, scale = preprocess_triplet(frames, self.size, cfg.data.motion, content, hflip)
            sample["rgb"], sample["mot"] = rgb, mot
        boxes_in = gt[:, :4] * scale
        if hflip and len(boxes_in):
            content_w = s.width * scale
            boxes_in = boxes_in.copy()
            boxes_in[:, [0, 2]] = content_w - boxes_in[:, [2, 0]]
        tg = build_targets(boxes_in, gt[:, 4].astype(int), self.C, self.size, cfg.model.max_objs, cfg.model.gate)
        for k, v in tg.items():
            sample[k] = torch.from_numpy(v)
        sample["meta"] = {"seq": s.name, "frame": t, "scale": scale, "width": s.width, "height": s.height,
                          "gt": gt.copy()}
        return sample


def collate(batch):
    out = {}
    for k in batch[0]:
        if k == "meta":
            out[k] = [b[k] for b in batch]
        elif k == "feats":
            out[k] = {f: torch.stack([b[k][f] for b in batch]) for f in batch[0][k]}
        else:
            out[k] = torch.stack([b[k] for b in batch])
    return out


class FrameDataset(Dataset):
    """Preprocessed (rgb, motion) inputs for the feature-caching pass."""

    def __init__(self, cfg, seqs, step):
        self.cfg, self.seqs = cfg, seqs
        self.items = _frame_items(seqs, step)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from .transforms import preprocess_triplet
        si, t = self.items[i]
        s = self.seqs[si]
        idx = triplet_indices(t, len(s), self.cfg.data.motion.get("mode", "causal"))
        rgb, mot, scale = preprocess_triplet([s.read(j) for j in idx], self.cfg.data.input_size, self.cfg.data.motion)
        return rgb, mot, si, t, scale


class CropDataset(Dataset):
    """Tier 2: high-resolution crops of every annotated target that is large enough to be *recognised*
    (Johnson: >= 4 line pairs ~ 8 px across), from original-resolution frames."""

    def __init__(self, cfg, split, train=False, max_items=None):
        self.cfg, self.train = cfg, train
        t2 = cfg.tier2
        self.seqs = load_split(cfg.data.root, split, cfg.classes)
        self.items = []
        step = t2.get("frame_step", 3) if train else max(1, t2.get("frame_step", 3))
        for si, s in enumerate(self.seqs):
            for t in range(0, len(s), step):
                b = s.boxes(t)
                for j, box in enumerate(b):
                    px = math.sqrt(max(box[2] - box[0], 0) * max(box[3] - box[1], 0))
                    if px >= t2.min_obj_px:
                        self.items.append((si, t, j))
        if max_items and len(self.items) > max_items:
            rng = random.Random(0)
            self.items = sorted(rng.sample(self.items, max_items))

    def __len__(self):
        return len(self.items)

    def labels(self):
        return np.array([int(self.seqs[si].boxes(t)[j][4]) for si, t, j in self.items], np.int64)

    def __getitem__(self, i):
        si, t, j = self.items[i]
        s = self.seqs[si]
        box = s.boxes(t)[j]
        idx = triplet_indices(t, len(s), self.cfg.data.motion.get("mode", "causal"))
        frames = [s.read(k) for k in idx]
        jitter, hflip = None, False
        if self.train:
            jitter = (random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), random.uniform(0.85, 1.25))
            hflip = random.random() < 0.5
        t2 = self.cfg.tier2
        rgb, mot = crop_triplet(frames, box, t2.crop_size, self.cfg.data.motion, t2.context, t2.min_crop_px,
                                jitter, hflip)
        return rgb, mot, int(box[4])
