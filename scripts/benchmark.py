#!/usr/bin/env python
"""Efficiency table for every variant: #params, MAdds, GFLOPs, train step time, inference latency.

    python scripts/benchmark.py --variants rgb_only motion_only early_fusion late_fusion joint_attn mbt_k4 mbt_k8
Needs no data or trained weights (random inputs). Writes results/benchmark.csv.
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from mbtf.config import load_config  # noqa: E402
from mbtf.models import MBTFDetector  # noqa: E402
from mbtf.profiling import count_madds, count_params, inference_latency, step_time  # noqa: E402
from mbtf.variants import DEFAULT_ORDER, variant_overrides  # noqa: E402


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--variants", nargs="+", default=DEFAULT_ORDER)
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--batch-size", type=int, default=None, help="default: train.batch_size")
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--no-e2e", action="store_true", help="skip end-to-end (backbone-included) step time")
    p.add_argument("--out", default="results/benchmark.csv")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)
    rows = []
    for v in args.variants:
        cfg = load_config(args.config, variant_overrides(v) + ["model.pretrained=false"] + list(args.set))
        bs = args.batch_size or cfg.train.batch_size
        torch.manual_seed(0)
        model = MBTFDetector(cfg).to(device)
        row = {"variant": v, "fusion": cfg.model.fusion, "K": cfg.model.bottleneck_tokens if cfg.model.fusion == "bottleneck" else "",
               "input": cfg.data.input_size, "batch": bs}
        row.update(count_params(model))
        row.update(count_madds(model, cfg, device))
        st = step_time(model, cfg, device, bs, cached=True, iters=args.iters, amp=cfg.train.amp)
        row.update({"step_ms_cached": st["step_ms"], "peak_mem_mb_cached": st["peak_mem_mb"]})
        if not args.no_e2e:
            st2 = step_time(model, cfg, device, bs, cached=False, iters=max(5, args.iters // 3), amp=cfg.train.amp)
            row.update({"step_ms_e2e": st2["step_ms"], "peak_mem_mb_e2e": st2["peak_mem_mb"]})
        row.update(inference_latency(model, cfg, device, iters=args.iters, amp=cfg.train.amp))
        rows.append(row)
        print(f"{v:14s} params {row['params_total'] / 1e6:6.2f}M (trainable {row['params_trainable_det'] / 1e6:5.2f}M) "
              f"| {row['gmadds_full']:7.2f} GMAdds {row['gflops_full']:7.2f} GFLOPs (fusion {row['gflops_fusion']:6.2f}) "
              f"| step {row['step_ms_cached']:7.1f} ms | {row['fps']:6.1f} FPS", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
