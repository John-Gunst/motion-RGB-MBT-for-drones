#!/usr/bin/env python
"""Detection -> (optional crop re-classification) -> SORT -> fading motion trails, on a video file or camera.

    python scripts/infer_video.py --ckpt runs/mbt_k4/tier2/ckpt_tier2.pt --source clip.mp4 --out out.mp4 --show-gate
    python scripts/infer_video.py --ckpt ... --source 0 --display          # webcam
"""
import json
import math
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from mbtf import engine  # noqa: E402
from mbtf.cli import cfg_from_ckpt, load_ckpt  # noqa: E402
from mbtf.data.crops import crop_triplet  # noqa: E402
from mbtf.data.motion import current_slot  # noqa: E402
from mbtf.data.transforms import STD, MEAN, preprocess_triplet  # noqa: E402
from mbtf.models import MBTFDetector, decode  # noqa: E402
from mbtf.tracking import Sort  # noqa: E402
from mbtf.viz import TrailRenderer  # noqa: E402


def inset(canvas, img, slot, label, frac=0.22):
    H, W = canvas.shape[:2]
    w = int(W * frac)
    h = int(img.shape[0] * w / img.shape[1])
    im = cv2.resize(img, (w, h))
    x0 = W - w - 10
    y0 = 10 + slot * (h + 10)
    if y0 + h > H:
        return
    canvas[y0:y0 + h, x0:x0 + w] = im
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (255, 255, 255), 1)
    cv2.putText(canvas, label, (x0 + 4, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--source", required=True, help="video path or camera index")
    p.add_argument("--out", default=None)
    p.add_argument("--display", action="store_true")
    p.add_argument("--det-thr", type=float, default=None)
    p.add_argument("--no-reclassify", action="store_true")
    p.add_argument("--show-motion", action="store_true")
    p.add_argument("--show-gate", action="store_true", help="inset: Johnson gate (red = RGB-reliant, blue = motion-reliant)")
    p.add_argument("--save-tracks", default=None, help="write per-frame tracks as JSON lines")
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    ck = load_ckpt(args.ckpt)
    cfg = cfg_from_ckpt(ck, args.set)
    model = MBTFDetector(cfg).to(device).eval()
    model.load_state_dict(ck["model"])
    reclassify = ck.get("tier2_done", False) and not args.no_reclassify
    det_thr = args.det_thr if args.det_thr is not None else cfg.track.det_thr
    tr = cfg.track
    tracker = Sort(tr.max_age, tr.min_hits, tr.iou_threshold, tr.match_scale)
    trails = TrailRenderer(cfg.classes, tr.trail_len)

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = None
    track_f = open(args.save_tracks, "w") if args.save_tracks else None
    mode = cfg.data.motion.get("mode", "causal")
    buf = deque(maxlen=3)
    frame_idx, t_sum = 0, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            if mode == "centered" and len(buf) >= 2:  # flush the last frame
                buf.append(buf[-1])
            else:
                break
        else:
            buf.append(frame)
        first = len(buf) == 1
        while len(buf) < 3:
            buf.appendleft(buf[0])
        if first and mode == "centered":
            continue  # wait for one future frame before emitting frame 0
        triplet = list(buf)
        cur = triplet[current_slot(mode)].copy()
        t0 = time.perf_counter()
        rgb, mot, scale = preprocess_triplet(triplet, cfg.data.input_size, cfg.data.motion)
        with torch.no_grad(), engine.amp_ctx(device, cfg.train.amp):
            out = model(rgb=rgb[None].to(device), mot=mot[None].to(device))
        boxes, scores, cls = (x[0].cpu().numpy() for x in decode(out, cfg.eval.topk))
        k = scores >= det_thr
        boxes, scores, cls = boxes[k] / scale, scores[k], cls[k]
        H, W = cur.shape[:2]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H)
        if reclassify and len(boxes):
            px = np.sqrt((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
            big = np.where(px >= cfg.tier2.reclassify_min_px)[0]
            if len(big):
                crops = [crop_triplet(triplet, boxes[i], cfg.tier2.crop_size, cfg.data.motion, cfg.tier2.context,
                                      cfg.tier2.min_crop_px) for i in big]
                r = torch.stack([c[0] for c in crops]).to(device)
                m = torch.stack([c[1] for c in crops]).to(device)
                with torch.no_grad(), engine.amp_ctx(device, cfg.train.amp):
                    probs = model.classify(rgb=r, mot=m).float().softmax(1).cpu().numpy()
                cls = cls.copy()
                cls[big] = probs.argmax(1)
        dets = np.concatenate([boxes, scores[:, None], cls[:, None]], 1) if len(boxes) else np.zeros((0, 6))
        tracks = tracker.update(dets)
        t_sum += time.perf_counter() - t0
        trails.update(tracks)
        vis = trails.draw(cur, tracks)
        slot = 0
        if args.show_motion:
            mm = mot.numpy().transpose(1, 2, 0) * STD + MEAN
            mm = (np.clip(mm[..., 2], 0, 1) * 255).astype(np.uint8)[: int(H * scale), : int(W * scale)]
            inset(vis, cv2.applyColorMap(mm, cv2.COLORMAP_INFERNO), slot, "motion |dI| (min)")
            slot += 1
        if args.show_gate and "gate" in out:
            g = out["gate"][0, 0].float().cpu().numpy()
            gh, gw = int(math.ceil(H * scale / 8)), int(math.ceil(W * scale / 8))
            g = (g[:gh, :gw] * 255).astype(np.uint8)
            g = cv2.resize(g, (W, H), interpolation=cv2.INTER_NEAREST)
            inset(vis, cv2.applyColorMap(g, cv2.COLORMAP_JET), slot, "gate: red=RGB blue=motion")
        cv2.putText(vis, f"{frame_idx}  {1.0 / max(t_sum / (frame_idx + 1), 1e-6):.1f} FPS", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        if track_f:
            track_f.write(json.dumps({"frame": frame_idx, "tracks": [
                {"id": t["id"], "box": [float(v) for v in t["box"]], "cls": cfg.classes[t["cls"]], "score": t["score"]}
                for t in tracks]}) + "\n")
        if args.out:
            if writer is None:
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
            writer.write(vis)
        if args.display:
            cv2.imshow("MBTF", vis)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
        frame_idx += 1
        if not ok:
            break
    cap.release()
    if writer:
        writer.release()
    if track_f:
        track_f.close()
    if args.display:
        cv2.destroyAllWindows()
    print(f"{frame_idx} frames, mean pipeline {1000 * t_sum / max(frame_idx, 1):.1f} ms/frame"
          + (f" -> {args.out}" if args.out else ""))


if __name__ == "__main__":
    main()
