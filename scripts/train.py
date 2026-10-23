#!/usr/bin/env python
"""Scale-aware curriculum training.

Tier 1 (detection & motion):  python scripts/train.py --variant mbt_k4 --tier 1 --cached --epochs 72
Tier 2 (classification):      python scripts/train.py --tier 2 --init runs/mbt_k4/sched72/ckpt_final.pt
"""
import csv
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, WeightedRandomSampler  # noqa: E402

from mbtf import engine  # noqa: E402
from mbtf.cli import base_parser, build_cfg, cfg_from_ckpt, load_ckpt, seed_all  # noqa: E402
from mbtf.config import to_dict  # noqa: E402
from mbtf.data.datasets import CropDataset, DetectionDataset, collate  # noqa: E402
from mbtf.losses import compute_loss  # noqa: E402
from mbtf.metrics import clean  # noqa: E402
from mbtf.models import MBTFDetector  # noqa: E402


def make_scaler(device, amp):
    use = amp and device.type == "cuda" and not torch.cuda.is_bf16_supported()
    return torch.amp.GradScaler("cuda", enabled=use)


def save(path, model, cfg, **extra):
    torch.save({"model": model.state_dict(), "cfg": to_dict(cfg), **extra}, path)


# ------------------------------------------------------------------------------------------------------ Tier 1
def train_tier1(args, cfg, device, out):
    model = MBTFDetector(cfg).to(device)
    E = int(cfg.train.epochs)
    save_at = sorted({int(e) for e in cfg.train.save_at if int(e) <= E} | {E})
    ds = DetectionDataset(cfg, "train", cached=args.cached, train=True)
    w = ds.sample_weights()
    sampler = WeightedRandomSampler(torch.as_tensor(w), num_samples=len(ds), replacement=True) if (w != 1).any() else None
    nw = cfg.data.num_workers
    dl = DataLoader(ds, cfg.train.batch_size, shuffle=sampler is None, sampler=sampler, num_workers=nw,
                    collate_fn=collate, drop_last=len(ds) > cfg.train.batch_size, pin_memory=device.type == "cuda",
                    persistent_workers=nw > 0)
    val_dl = None
    if not args.no_val:
        vds = DetectionDataset(cfg, "val", cached=args.cached, max_frames=cfg.train.get("val_max_frames"))
        val_dl = DataLoader(vds, cfg.train.batch_size, num_workers=nw, collate_fn=collate)

    params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("cls_head")]
    opt = torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    iters_per_epoch = max(1, len(dl))
    total = E * iters_per_epoch
    steps = [int(f * total) for f in cfg.train.lr_steps]
    warm = min(cfg.train.warmup_iters, max(1, total // 10))

    def lr_lambda(it):
        f = min(1.0, (it + 1) / warm)
        return f * (0.1 ** sum(it >= s for s in steps))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = make_scaler(device, cfg.train.amp)
    print(f"[tier1] {cfg.model.fusion} K={cfg.model.bottleneck_tokens} | {len(ds)} train frames | "
          f"{iters_per_epoch} it/epoch | {E} epochs | cached={args.cached} | device={device}")

    log_f = open(out / "train_log.csv", "w", newline="")
    log = csv.writer(log_f)
    log.writerow(["epoch", "iter", "lr", "loss", "hm", "wh", "off", "gate", "step_ms"])
    it = 0
    for epoch in range(1, E + 1):
        model.train()
        t_epoch = time.time()
        for batch in dl:
            batch = engine.to_device(batch, device)
            engine.sync(device)
            t0 = time.perf_counter()
            with engine.amp_ctx(device, cfg.train.amp):
                outp = engine.run_model(model, batch)
                losses = compute_loss(outp, batch, cfg)
            opt.zero_grad(set_to_none=True)
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            engine.sync(device)
            step_ms = (time.perf_counter() - t0) * 1000
            if not math.isfinite(losses["total"].item()):
                raise RuntimeError(f"non-finite loss at iter {it}: { {k: v.item() for k, v in losses.items()} }")
            if it % args.log_every == 0:
                row = [epoch, it, opt.param_groups[0]["lr"], losses["total"].item(), losses["hm"].item(),
                       losses["wh"].item(), losses["off"].item(), losses.get("gate", torch.zeros(())).item(), step_ms]
                log.writerow(row)
                log_f.flush()
                print(f"ep {epoch:3d} it {it:6d} lr {row[2]:.2e} loss {row[3]:.4f} (hm {row[4]:.3f} wh {row[5]:.3f} "
                      f"off {row[6]:.3f} gate {row[7]:.3f}) {step_ms:.1f} ms/step", flush=True)
            it += 1
            if args.max_iters and it >= args.max_iters:
                break
        print(f"epoch {epoch} done in {time.time() - t_epoch:.1f}s")
        if epoch in save_at:
            ck = out / ("ckpt_final.pt" if epoch == E else f"ckpt_e{epoch:03d}.pt")
            save(ck, model, cfg, epoch=epoch, schedule=E, variant=args.variant)
            if epoch == E and E in cfg.train.save_at:
                save(out / f"ckpt_e{epoch:03d}.pt", model, cfg, epoch=epoch, schedule=E, variant=args.variant)
            if val_dl is not None:
                m = engine.evaluate_detection(model, val_dl, cfg, device, cfg.train.amp)
                rec = clean({"epoch": epoch, "AP": m["bbox"]["AP"], "AP50": m["bbox"]["AP50"],
                             "f1": m["prf"]["best"]["f1"]})
                print(f"[val] epoch {epoch}: {rec}")
                with open(out / "val_log.jsonl", "a") as f:
                    f.write(json.dumps(rec) + "\n")
        if args.max_iters and it >= args.max_iters:
            save(out / "ckpt_final.pt", model, cfg, epoch=epoch, schedule=E, variant=args.variant)
            break
    log_f.close()
    print(f"done -> {out}")


# ------------------------------------------------------------------------------------------------------ Tier 2
def train_tier2(args, device, out):
    ck = load_ckpt(args.init)
    cfg = cfg_from_ckpt(ck, args.set)
    model = MBTFDetector(cfg).to(device)
    model.load_state_dict(ck["model"])
    t2 = cfg.tier2
    for n, p in model.named_parameters():
        p.requires_grad_(n.startswith("cls_head") or
                         (t2.train_fusion and (n.startswith("fusion") or n.startswith("neck"))))
    params = [p for p in model.parameters() if p.requires_grad]
    ds = CropDataset(cfg, "train", train=True, max_items=args.max_items)
    vds = CropDataset(cfg, "val", train=False, max_items=args.max_items)
    if len(ds) == 0:
        raise RuntimeError(f"no training crops >= {t2.min_obj_px}px; lower tier2.min_obj_px")
    nw = cfg.data.num_workers
    dl = DataLoader(ds, t2.batch_size, shuffle=True, num_workers=nw, drop_last=len(ds) > t2.batch_size,
                    pin_memory=device.type == "cuda", persistent_workers=nw > 0)
    vdl = DataLoader(vds, t2.batch_size, num_workers=nw)
    C = len(cfg.classes)
    weight = None
    if t2.class_balance:
        cnt = np.bincount(ds.labels(), minlength=C).astype(np.float64)
        wv = np.where(cnt > 0, cnt.sum() / np.maximum(cnt, 1) / C, 0.0)
        weight = torch.tensor(wv, dtype=torch.float32, device=device)
        print("class counts:", dict(zip(cfg.classes, cnt.astype(int).tolist())))
    opt = torch.optim.AdamW(params, lr=t2.lr, weight_decay=cfg.train.weight_decay)
    total = t2.epochs * max(1, len(dl))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=t2.lr, total_steps=total, pct_start=0.1)
    scaler = make_scaler(device, cfg.train.amp)
    best, it = -1.0, 0
    print(f"[tier2] {len(ds)} train crops / {len(vds)} val crops | train_fusion={t2.train_fusion}")
    for epoch in range(1, t2.epochs + 1):
        model.train()
        for rgb, mot, y in dl:
            rgb, mot, y = rgb.to(device), mot.to(device), y.to(device)
            with engine.amp_ctx(device, cfg.train.amp):
                logits = model.classify(rgb=rgb, mot=mot)
            loss = F.cross_entropy(logits.float(), y, weight=weight, label_smoothing=t2.label_smoothing)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            if it % args.log_every == 0:
                print(f"[tier2] ep {epoch} it {it} loss {loss.item():.4f}", flush=True)
            it += 1
            if args.max_iters and it >= args.max_iters:
                break
        m = engine.evaluate_classification(model, vdl, cfg, device, cfg.train.amp) if len(vds) else {"top1": 0.0}
        top1 = m.get("top1") or 0.0
        print(f"[tier2] epoch {epoch}: val top1 {top1:.4f} macroF1 {m.get('macro_f1')}")
        if top1 >= best:
            best = top1
            save(out / "ckpt_tier2.pt", model, cfg, epoch=ck.get("epoch"), schedule=ck.get("schedule"),
                 variant=ck.get("variant"), tier2_done=True, tier2_epoch=epoch, val_top1=top1)
        if args.max_iters and it >= args.max_iters:
            break
    print(f"best val top1 {best:.4f} -> {out / 'ckpt_tier2.pt'}")


def main():
    p = base_parser(__doc__)
    p.add_argument("--tier", type=int, default=1, choices=[1, 2])
    p.add_argument("--cached", action="store_true", help="train on cached frozen-backbone features (tier 1)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--init", default=None, help="tier-1 checkpoint to start tier 2 from")
    p.add_argument("--no-val", action="store_true")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--max-iters", type=int, default=0, help="debug: stop after this many iterations")
    p.add_argument("--max-items", type=int, default=None, help="debug: cap number of tier-2 crops")
    args = p.parse_args()
    device = torch.device(args.device)
    if args.tier == 1:
        if args.epochs:
            args.set = list(args.set) + [f"train.epochs={args.epochs}"]
        cfg = build_cfg(args)
        seed_all(cfg.train.seed)
        name = args.variant or f"{cfg.model.fusion}_k{cfg.model.bottleneck_tokens}"
        out = Path(args.out or f"runs/{name}/sched{cfg.train.epochs}")
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(to_dict(cfg), indent=2))
        train_tier1(args, cfg, device, out)
    else:
        if not args.init:
            p.error("--tier 2 needs --init <tier-1 checkpoint>")
        seed_all(0)
        out = Path(args.out or Path(args.init).parent.parent / "tier2")
        out.mkdir(parents=True, exist_ok=True)
        train_tier2(args, device, out)


if __name__ == "__main__":
    main()
