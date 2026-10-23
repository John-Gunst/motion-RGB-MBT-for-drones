"""Detection metrics (COCO AP^bb overall / per size bucket / per class, P/R/F1/Acc) and classification metrics."""
import contextlib
import io
import math

import numpy as np


# --------------------------------------------------------------------------------------------- COCO-style AP
def coco_ap(gts, dets, class_names, size_buckets):
    """gts:  list of (img_id, boxes (n,4) x1y1x2y2, cls (n,))
    dets: list of (img_id, boxes (k,4), scores (k,), cls (k,))
    size_buckets: {name: [lo_px, hi_px]} on sqrt(area) in the same (original) pixel space.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    images, anns, ann_id = [], [], 1
    for img_id, b, c in gts:
        images.append({"id": int(img_id)})
        for bb, cc in zip(b, c):
            w, h = float(bb[2] - bb[0]), float(bb[3] - bb[1])
            anns.append({"id": ann_id, "image_id": int(img_id), "category_id": int(cc) + 1,
                         "bbox": [float(bb[0]), float(bb[1]), w, h], "area": w * h, "iscrowd": 0})
            ann_id += 1
    cats = [{"id": i + 1, "name": n} for i, n in enumerate(class_names)]
    res = []
    for img_id, b, s, c in dets:
        for bb, ss, cc in zip(b, s, c):
            res.append({"image_id": int(img_id), "category_id": int(cc) + 1,
                        "bbox": [float(bb[0]), float(bb[1]), float(bb[2] - bb[0]), float(bb[3] - bb[1])],
                        "score": float(ss)})
    labels = ["all"] + list(size_buckets)
    out = {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "AR100": 0.0}
    for l in labels[1:]:
        out[f"AP_{l}"] = float("nan")
    if not anns or not res:
        return out
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = {"images": images, "annotations": anns, "categories": cats}
        gt.createIndex()
        dt = gt.loadRes(res)
        E = COCOeval(gt, dt, "bbox")
        E.params.areaRng = [[0, 1e10]] + [[lo ** 2, min(hi, 1e5) ** 2] for lo, hi in size_buckets.values()]
        E.params.areaRngLbl = labels
        E.params.maxDets = [1, 10, 100]
        E.evaluate()
        E.accumulate()
    P = E.eval["precision"]  # [T(iou), R(recall), K(class), A(area), M(maxdets)]
    Rc = E.eval["recall"]    # [T, K, A, M]
    iou_thrs = E.params.iouThrs

    def _ap(t=None, k=None, a=0):
        p = P[:, :, :, a, -1]
        if t is not None:
            p = p[[int(np.argmin(np.abs(iou_thrs - t)))]]
        if k is not None:
            p = p[:, :, [k]]
        p = p[p > -1]
        return float(p.mean()) if p.size else float("nan")

    out["AP"], out["AP50"], out["AP75"] = _ap(), _ap(0.5), _ap(0.75)
    r = Rc[:, :, 0, -1]
    out["AR100"] = float(r[r > -1].mean()) if (r > -1).any() else float("nan")
    for ai, l in enumerate(labels[1:], start=1):
        out[f"AP_{l}"] = _ap(a=ai)
        out[f"AP50_{l}"] = _ap(0.5, a=ai)
    for k, n in enumerate(class_names):
        out[f"AP_cls_{n}"] = _ap(k=k)
        for ai, l in enumerate(labels[1:], start=1):
            out[f"AP_{n}_{l}"] = _ap(k=k, a=ai)  # e.g. AP_drone_tiny: drone AP by size
    return out


# --------------------------------------------------------------------------------------------- P / R / F1 / Acc
def iou_np(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.clip(aa[:, None] + ab[None] - inter, 1e-9, None)


def _greedy(gb, db, ds, iou_thr):
    """Greedy score-ordered matching. Returns list of (det_idx, gt_idx)."""
    if len(gb) == 0 or len(db) == 0:
        return []
    order = np.argsort(-ds)
    ious = iou_np(db[order], gb)
    used = np.zeros(len(gb), bool)
    pairs = []
    for i, row in enumerate(ious):
        row = np.where(used, -1, row)
        j = int(np.argmax(row))
        if row[j] >= iou_thr:
            used[j] = True
            pairs.append((int(order[i]), j))
    return pairs


def prf_at(gts, dets, thr, iou_thr, num_classes):
    tp = fp = fn = 0
    loc_matched = loc_correct = 0
    frame_ok = 0
    for (_, gb, gc), (_, db, ds, dc) in zip(gts, dets):
        k = ds >= thr
        db, ds, dc = db[k], ds[k], dc[k]
        # class-aware matching -> precision / recall / F1
        for c in range(num_classes):
            g, d = gb[gc == c], dc == c
            m = _greedy(g, db[d], ds[d], iou_thr)
            tp += len(m)
            fp += int(d.sum()) - len(m)
            fn += len(g) - len(m)
        # class-agnostic matching -> classification accuracy of localised targets
        m = _greedy(gb, db, ds, iou_thr)
        loc_matched += len(m)
        loc_correct += sum(int(dc[i] == gc[j]) for i, j in m)
        # frame-level presence accuracy (target present & detected, or absent & nothing detected)
        frame_ok += int((len(gb) > 0) == (len(db) > 0))
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-12)
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn,
            "cls_acc_of_localized": loc_correct / max(loc_matched, 1),
            "frame_acc": frame_ok / max(len(gts), 1)}


def prf_sweep(gts, dets, num_classes, iou_thr=0.5, thresholds=None):
    """P/R/F1/Acc at the F1-optimal confidence threshold (reported) and at a fixed 0.3."""
    thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2) if thresholds is None else thresholds
    best = None
    for t in thresholds:
        r = prf_at(gts, dets, float(t), iou_thr, num_classes)
        r["threshold"] = float(t)
        if best is None or r["f1"] > best["f1"]:
            best = r
    fixed = prf_at(gts, dets, 0.3, iou_thr, num_classes)
    return {"best": best, "at_0.3": fixed}


# --------------------------------------------------------------------------------------------- classification
def topk_acc(logits, y, ks=(1, 5)):
    logits, y = np.asarray(logits), np.asarray(y)
    C = logits.shape[1]
    order = np.argsort(-logits, 1)
    out = {}
    for k in ks:
        kk = min(k, C)
        out[f"top{k}"] = float((order[:, :kk] == y[:, None]).any(1).mean()) if len(y) else float("nan")
        if k > C:
            out[f"top{k}_note"] = f"only {C} classes: top-{k} is trivially 1.0"
    return out


def cls_report(logits, y, class_names):
    logits, y = np.asarray(logits), np.asarray(y)
    C = len(class_names)
    pred = logits.argmax(1) if len(y) else np.zeros(0, int)
    cm = np.zeros((C, C), np.int64)
    for a, b in zip(y, pred):
        cm[a, b] += 1
    per = {}
    ps, rs, fs = [], [], []
    for c, n in enumerate(class_names):
        tp = cm[c, c]
        p = tp / max(cm[:, c].sum(), 1)
        r = tp / max(cm[c].sum(), 1)
        f = 2 * p * r / max(p + r, 1e-12)
        per[n] = {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(cm[c].sum())}
        if cm[c].sum() > 0:
            ps.append(p), rs.append(r), fs.append(f)
    out = {"accuracy": float((pred == y).mean()) if len(y) else float("nan"),
           "macro_precision": float(np.mean(ps)) if ps else float("nan"),
           "macro_recall": float(np.mean(rs)) if rs else float("nan"),
           "macro_f1": float(np.mean(fs)) if fs else float("nan"),
           "per_class": per, "confusion_matrix": cm.tolist()}
    out.update(topk_acc(logits, y, (1, 5)))
    return out


def clean(o):
    """NaN -> None for JSON."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if isinstance(o, (np.floating,)):
        return clean(float(o))
    if isinstance(o, (np.integer,)):
        return int(o)
    return o
