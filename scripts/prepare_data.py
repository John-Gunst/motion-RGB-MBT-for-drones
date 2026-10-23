#!/usr/bin/env python
"""Convert datasets into the unified sequence format (see mbtf/data/sequences.py), then make splits.

Sub-commands
  antiuav  Anti-UAV (RGB-T) sequences: <dir>/<seq>/visible.mp4 + visible.json {"exist": [...], "gt_rect": [[x,y,w,h], ...]}
  mot      MOT-style text labels (frame,id,x,y,w,h,conf,class,...) for a video or a frame folder.
           Works for CVAT "MOT 1.1" exports of your own footage, and for VisDrone-MOT (see the README caveat).
  yolo     A folder of sequential frames with YOLO txt labels (class cx cy w h, normalised) — no track ids.
  split    Random sequence-level train/val/test split (never split frames of one video across sets).

Examples
  python scripts/prepare_data.py antiuav --src ~/data/Anti-UAV/train --root data/processed --prefix auv
  python scripts/prepare_data.py mot --video clips/backyard1.mp4 --labels clips/backyard1_gt.txt \
         --class-map 1:drone 2:bird 3:bug --root data/processed --name backyard1
  python scripts/prepare_data.py split --root data/processed --ratios 0.7 0.15 0.15
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def write_sequence(root, name, frames_iter, boxes_by_frame, classes, step=1, max_side=None, jpeg_q=95):
    """frames_iter yields BGR images in order. boxes_by_frame: {src_frame_idx: [[x1,y1,x2,y2,cls,tid], ...]}."""
    d = Path(root) / "sequences" / name
    (d / "frames").mkdir(parents=True, exist_ok=True)
    out_frames = {}
    W = H = None
    k = 0
    for i, img in enumerate(frames_iter):
        if i % step:
            continue
        s = 1.0
        if max_side and max(img.shape[:2]) > max_side:
            s = max_side / max(img.shape[:2])
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        H, W = img.shape[:2]
        cv2.imwrite(str(d / "frames" / f"{k:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
        bx = [[b[0] * s, b[1] * s, b[2] * s, b[3] * s, int(b[4]), int(b[5])] for b in boxes_by_frame.get(i, [])]
        bx = [[float(np.clip(b[0], 0, W)), float(np.clip(b[1], 0, H)), float(np.clip(b[2], 0, W)),
               float(np.clip(b[3], 0, H)), b[4], b[5]] for b in bx]
        bx = [b for b in bx if b[2] - b[0] > 0.5 and b[3] - b[1] > 0.5]
        if bx:
            out_frames[str(k)] = bx
        k += 1
    if k == 0:
        raise RuntimeError(f"{name}: no frames written")
    (d / "annotations.json").write_text(json.dumps({"width": W, "height": H, "classes": classes, "frames": out_frames}))
    n_obj = sum(len(v) for v in out_frames.values())
    print(f"{name}: {k} frames, {n_obj} boxes")


def video_frames(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    while True:
        ok, f = cap.read()
        if not ok:
            break
        yield f
    cap.release()


def folder_frames(path):
    for p in sorted(q for q in Path(path).iterdir() if q.suffix.lower() in IMG_EXT):
        yield cv2.imread(str(p))


def cmd_antiuav(a):
    src = Path(a.src).expanduser()
    seqs = sorted(p for p in src.iterdir() if p.is_dir())
    for s in seqs:
        js, vid = s / "visible.json", s / "visible.mp4"
        if not js.exists() or not vid.exists():
            print(f"skip {s.name} (no visible.json / visible.mp4)")
            continue
        meta = json.loads(js.read_text())
        exist, rects = meta.get("exist", []), meta.get("gt_rect", [])
        boxes = {}
        for i, (e, r) in enumerate(zip(exist, rects)):
            if e and r and len(r) == 4 and r[2] > 0 and r[3] > 0:
                x, y, w, h = r
                boxes[i] = [[x, y, x + w, y + h, 0, 1]]
        write_sequence(a.root, f"{a.prefix}_{s.name}", video_frames(vid), boxes, ["drone"], a.step, a.max_side)


def parse_class_map(items):
    m = {}
    for it in items or []:
        k, v = it.split(":", 1)
        m[int(k)] = v
    return m


def cmd_mot(a):
    cmap = parse_class_map(a.class_map)
    classes = list(dict.fromkeys(cmap.values())) if cmap else ["drone"]
    boxes = defaultdict(list)
    for line in Path(a.labels).read_text().splitlines():
        p = [x for x in line.replace(",", " ").split() if x]
        if len(p) < 6:
            continue
        f, tid = int(float(p[0])), int(float(p[1]))
        x, y, w, h = map(float, p[2:6])
        raw_cls = int(float(p[7])) if len(p) > 7 else 1
        if len(p) > 6 and float(p[6]) == 0 and a.respect_ignore:
            continue
        if cmap and raw_cls not in cmap:
            continue
        c = classes.index(cmap[raw_cls]) if cmap else 0
        boxes[f - a.first_frame].append([x, y, x + w, y + h, c, tid])
    frames = video_frames(a.video) if a.video else folder_frames(a.frames)
    write_sequence(a.root, a.name, frames, boxes, classes, a.step, a.max_side)


def cmd_yolo(a):
    names = a.names
    imgs = sorted(q for q in Path(a.frames).iterdir() if q.suffix.lower() in IMG_EXT)
    lbl_dir = Path(a.labels)
    boxes = {}
    for i, p in enumerate(imgs):
        t = lbl_dir / (p.stem + ".txt")
        if not t.exists():
            continue
        img = cv2.imread(str(p))
        H, W = img.shape[:2]
        for line in t.read_text().splitlines():
            q = line.split()
            if len(q) < 5:
                continue
            c, cx, cy, w, h = int(q[0]), *map(float, q[1:5])
            boxes.setdefault(i, []).append([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H, c, -1])
    print("note: YOLO labels have no track ids; detection/classification training is unaffected.")
    write_sequence(a.root, a.name, (cv2.imread(str(p)) for p in imgs), boxes, names, a.step, a.max_side)


def cmd_split(a):
    root = Path(a.root)
    names = sorted(p.name for p in (root / "sequences").iterdir() if (p / "annotations.json").exists())
    if not names:
        sys.exit(f"no sequences under {root / 'sequences'}")
    rng = random.Random(a.seed)
    rng.shuffle(names)
    n = len(names)
    n_tr = max(1, int(round(a.ratios[0] * n)))
    n_va = max(1, int(round(a.ratios[1] * n))) if n >= 3 else 0
    split = {"train": names[:n_tr], "val": names[n_tr:n_tr + n_va], "test": names[n_tr + n_va:]}
    if not split["test"] and n >= 3:
        split["test"] = [split["train"].pop()]
    if not split["val"]:
        split["val"] = list(split["test"])
    (root / "splits").mkdir(exist_ok=True)
    for k, v in split.items():
        (root / "splits" / f"{k}.txt").write_text("\n".join(sorted(v)) + "\n")
        print(f"{k}: {len(v)} sequences")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", default="data/processed")
    common.add_argument("--step", type=int, default=1, help="keep every n-th frame (keep 1 for motion maps!)")
    common.add_argument("--max-side", type=int, default=None, help="downscale frames whose longer side exceeds this")

    s = sp.add_parser("antiuav", parents=[common])
    s.add_argument("--src", required=True)
    s.add_argument("--prefix", default="auv")
    s.set_defaults(fn=cmd_antiuav)

    s = sp.add_parser("mot", parents=[common])
    s.add_argument("--video")
    s.add_argument("--frames")
    s.add_argument("--labels", required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--class-map", nargs="*", help="raw_id:name pairs, e.g. 1:drone 2:bird 3:bug")
    s.add_argument("--first-frame", type=int, default=1, help="MOT frames are 1-based")
    s.add_argument("--respect-ignore", action="store_true", help="drop rows whose 7th column is 0")
    s.set_defaults(fn=cmd_mot)

    s = sp.add_parser("yolo", parents=[common])
    s.add_argument("--frames", required=True)
    s.add_argument("--labels", required=True)
    s.add_argument("--names", nargs="+", required=True)
    s.add_argument("--name", required=True)
    s.set_defaults(fn=cmd_yolo)

    s = sp.add_parser("split")
    s.add_argument("--root", default="data/processed")
    s.add_argument("--ratios", nargs=3, type=float, default=[0.7, 0.15, 0.15])
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_split)

    a = ap.parse_args()
    if a.cmd == "mot" and not (a.video or a.frames):
        ap.error("mot needs --video or --frames")
    a.fn(a)


if __name__ == "__main__":
    main()
