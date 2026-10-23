import contextlib
import time

import numpy as np
import torch

from .metrics import cls_report, coco_ap, prf_sweep
from .models.detector import decode


def amp_ctx(device, enabled=True):
    if not enabled or device.type != "cuda":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if k == "meta":
            out[k] = v
        elif k == "feats":
            out[k] = {f: t.to(device, non_blocking=True) for f, t in v.items()}
        else:
            out[k] = v.to(device, non_blocking=True)
    return out


def run_model(model, batch):
    if "feats" in batch:
        return model(feats=batch["feats"])
    return model(rgb=batch["rgb"], mot=batch["mot"])


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def collect_predictions(model, loader, cfg, device, amp=True):
    model.eval()
    gts, dets, times = [], [], []
    img_id = 0
    for batch in loader:
        batch = to_device(batch, device)
        sync(device)
        t0 = time.perf_counter()
        with amp_ctx(device, amp):
            out = run_model(model, batch)
        boxes, scores, cls = decode(out, cfg.eval.topk)
        sync(device)
        times.append((time.perf_counter() - t0) / len(batch["meta"]))
        boxes, scores, cls = boxes.cpu().numpy(), scores.cpu().numpy(), cls.cpu().numpy()
        for i, m in enumerate(batch["meta"]):
            keep = scores[i] >= cfg.eval.score_thr
            b = boxes[i][keep] / m["scale"]  # letterbox pads bottom/right only -> pure rescale
            b[:, [0, 2]] = b[:, [0, 2]].clip(0, m["width"])
            b[:, [1, 3]] = b[:, [1, 3]].clip(0, m["height"])
            g = m["gt"]
            gts.append((img_id, g[:, :4], g[:, 4].astype(int)))
            dets.append((img_id, b, scores[i][keep], cls[i][keep]))
            img_id += 1
    return gts, dets, float(np.mean(times)) if times else float("nan")


def evaluate_detection(model, loader, cfg, device, amp=True):
    gts, dets, t = collect_predictions(model, loader, cfg, device, amp)
    out = {"bbox": coco_ap(gts, dets, cfg.classes, cfg.eval.size_buckets),
           "segm": None,  # AP^mk: this model has no mask head — see README "About AP^mk"
           "prf": prf_sweep(gts, dets, len(cfg.classes), cfg.eval.iou),
           "eval_frames": len(gts), "latency_ms_per_frame": t * 1000}
    return out


@torch.no_grad()
def evaluate_classification(model, loader, cfg, device, amp=True):
    model.eval()
    logits, ys = [], []
    for rgb, mot, y in loader:
        with amp_ctx(device, amp):
            lo = model.classify(rgb=rgb.to(device), mot=mot.to(device))
        logits.append(lo.float().cpu().numpy())
        ys.append(y.numpy())
    if not logits:
        return {"note": "no crops in this split"}
    return cls_report(np.concatenate(logits), np.concatenate(ys), cfg.classes)
