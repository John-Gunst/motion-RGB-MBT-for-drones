#!/usr/bin/env python
"""Evaluate checkpoints: AP^bb (overall, by size bucket, per class, drone-by-size), P/R/F1/Acc,
and — for Tier-2 checkpoints — Top-1/Top-5, accuracy, precision, recall, F1 on target crops.

    python scripts/evaluate.py --ckpt runs/mbt_k4/sched*/ckpt_final.pt --split test --cached
Writes <ckpt_dir>/eval_<split>_<ckpt_stem>.json next to each checkpoint.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from mbtf import engine  # noqa: E402
from mbtf.cli import cfg_from_ckpt, load_ckpt  # noqa: E402
from mbtf.data.datasets import CropDataset, DetectionDataset, collate  # noqa: E402
from mbtf.metrics import clean  # noqa: E402
from mbtf.models import MBTFDetector  # noqa: E402


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", nargs="+", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--cached", action="store_true")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--max-crops", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--set", nargs="*", default=[], help="e.g. data.root=... (architecture keys are ignored)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)
    for path in args.ckpt:
        ck = load_ckpt(path)
        cfg = cfg_from_ckpt(ck, args.set)
        model = MBTFDetector(cfg).to(device)
        model.load_state_dict(ck["model"])
        res = {"ckpt": str(path), "variant": ck.get("variant"), "fusion": cfg.model.fusion,
               "bottleneck_tokens": cfg.model.bottleneck_tokens, "gate": cfg.model.gate.get("enabled", True),
               "epoch": ck.get("epoch"), "schedule": ck.get("schedule"), "split": args.split}
        ds = DetectionDataset(cfg, args.split, cached=args.cached, max_frames=args.max_frames)
        dl = DataLoader(ds, args.batch_size, num_workers=cfg.data.num_workers, collate_fn=collate)
        res["detection"] = engine.evaluate_detection(model, dl, cfg, device, cfg.train.amp)
        if ck.get("tier2_done"):
            cds = CropDataset(cfg, args.split, train=False, max_items=args.max_crops)
            cdl = DataLoader(cds, cfg.tier2.batch_size, num_workers=cfg.data.num_workers)
            res["classification"] = engine.evaluate_classification(model, cdl, cfg, device, cfg.train.amp)
        res = clean(res)
        out = Path(path).parent / f"eval_{args.split}_{Path(path).stem}.json"
        out.write_text(json.dumps(res, indent=2))
        b = res["detection"]["bbox"]
        pr = res["detection"]["prf"]["best"]
        msg = f"{path}: AP {b['AP']} AP50 {b['AP50']} | " + " ".join(
            f"AP_{k} {b.get('AP_' + k)}" for k in cfg.eval.size_buckets) + f" | F1 {pr['f1']:.3f} @ {pr['threshold']}"
        if "classification" in res:
            c = res["classification"]
            msg += f" | top1 {c.get('top1')} top5 {c.get('top5')}"
        print(msg)
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
