#!/usr/bin/env python
"""Run the frozen backbone once over every train/val/test frame and store fp16 features.

    python scripts/cache_features.py --splits train val test
Re-runs skip sequences that are already complete. The cache is shared by every variant that uses
the same backbone / input size / motion settings (see mbtf/data/cache.py:cache_key).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from mbtf import engine  # noqa: E402
from mbtf.cli import base_parser, build_cfg  # noqa: E402
from mbtf.data.cache import FEATS, cache_root, is_complete  # noqa: E402
from mbtf.data.datasets import FrameDataset  # noqa: E402
from mbtf.data.sequences import load_split  # noqa: E402
from mbtf.models.backbones import CHANNELS, FrozenResNet  # noqa: E402


def main():
    p = base_parser(__doc__)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--yes", action="store_true", help="don't ask before writing a large cache")
    args = p.parse_args()
    cfg = build_cfg(args)
    device = torch.device(args.device)
    bb = FrozenResNet(cfg.model.backbone, cfg.model.get("pretrained", True)).to(device)
    c2, c3 = CHANNELS[cfg.model.backbone]
    S = cfg.data.input_size
    h2, h3 = S // 8, S // 16
    per_frame = 2 * 2 * (c2 * h2 * h2 + c3 * h3 * h3)  # 2 modalities, fp16

    plan = []
    for split in args.splits:
        seqs = load_split(cfg.data.root, split, cfg.classes)
        step = cfg.data.frame_step_train if split == "train" else cfg.data.frame_step_eval
        todo = [s for s in seqs if not is_complete(cache_root(cfg, split) / s.name)]
        n = sum(len(range(0, len(s), step)) for s in todo)
        plan.append((split, todo, step, n))
    total = sum(x[3] for x in plan)
    print(f"cache: {cache_root(cfg, '<split>')}  |  {total} frames to encode  |  ~{total * per_frame / 2**30:.1f} GiB")
    if total == 0:
        print("nothing to do")
        return
    if not args.yes and total * per_frame > 20 * 2 ** 30:
        if input("more than 20 GiB — continue? [y/N] ").strip().lower() != "y":
            print("tip: raise data.frame_step_train / frame_step_eval or use a smaller input_size")
            return

    for split, todo, step, n in plan:
        if not todo:
            continue
        ds = FrameDataset(cfg, todo, step)
        dl = DataLoader(ds, args.batch_size, num_workers=cfg.data.num_workers)
        mms, metas = {}, {}
        for si, s in enumerate(todo):
            d = cache_root(cfg, split) / s.name
            d.mkdir(parents=True, exist_ok=True)
            frames = list(range(0, len(s), step))
            T = len(frames)
            shapes = {"rgb_l2": (T, c2, h2, h2), "rgb_l3": (T, c3, h3, h3), "mot_l2": (T, c2, h2, h2),
                      "mot_l3": (T, c3, h3, h3)}
            mms[si] = {k: np.lib.format.open_memmap(d / f"{k}.npy", mode="w+", dtype=np.float16, shape=shapes[k])
                       for k in FEATS}
            metas[si] = {"dir": d, "frames": frames, "row": {t: i for i, t in enumerate(frames)}, "scale": None,
                         "left": T}
        done = 0
        for rgb, mot, sidx, ts, scale in dl:
            with torch.no_grad(), engine.amp_ctx(device, True):
                r2, r3 = bb(rgb.to(device))
                m2, m3 = bb(mot.to(device))
            out = {"rgb_l2": r2, "rgb_l3": r3, "mot_l2": m2, "mot_l3": m3}
            out = {k: v.half().cpu().numpy() for k, v in out.items()}
            for b in range(len(ts)):
                si, t = int(sidx[b]), int(ts[b])
                meta = metas[si]
                row = meta["row"][t]
                for k in FEATS:
                    mms[si][k][row] = out[k][b]
                meta["scale"] = float(scale[b])
                meta["left"] -= 1
                if meta["left"] == 0:
                    for k in FEATS:
                        mms[si][k].flush()
                    idx = {"frames": meta["frames"], "scale": meta["scale"], "complete": True,
                           "backbone": cfg.model.backbone, "input_size": S}
                    (meta["dir"] / "index.json").write_text(json.dumps(idx))
            done += len(ts)
            print(f"\r[{split}] {done}/{n}", end="", flush=True)
        print()
    print("cache complete")


if __name__ == "__main__":
    main()
