"""Bounding boxes + fading, alpha-blended historical motion trails per track ID."""
import colorsys
from collections import deque

import cv2
import numpy as np


def id_color(tid):
    h = (tid * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def blend_line(img, p0, p1, color, thick, alpha):
    """Alpha-blend one line segment, touching only its bounding ROI (fast even at 4K)."""
    H, W = img.shape[:2]
    pad = thick + 2
    x0, x1 = max(min(p0[0], p1[0]) - pad, 0), min(max(p0[0], p1[0]) + pad, W)
    y0, y1 = max(min(p0[1], p1[1]) - pad, 0), min(max(p0[1], p1[1]) + pad, H)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    ov = roi.copy()
    cv2.line(ov, (p0[0] - x0, p0[1] - y0), (p1[0] - x0, p1[1] - y0), color, thick, cv2.LINE_AA)
    img[y0:y1, x0:x1] = cv2.addWeighted(ov, alpha, roi, 1 - alpha, 0)


class TrailRenderer:
    def __init__(self, class_names, trail_len=32, max_thick=4, linger=20):
        self.names = class_names
        self.trail_len, self.max_thick, self.linger = trail_len, max_thick, linger
        self.hist = {}
        self.last_seen = {}
        self.frame = 0

    def update(self, tracks):
        self.frame += 1
        for t in tracks:
            b = t["box"]
            c = (int(round((b[0] + b[2]) / 2)), int(round((b[1] + b[3]) / 2)))
            self.hist.setdefault(t["id"], deque(maxlen=self.trail_len)).append(c)
            self.last_seen[t["id"]] = self.frame
        for tid in [k for k, f in self.last_seen.items() if self.frame - f > self.linger]:
            self.hist.pop(tid, None)
            self.last_seen.pop(tid, None)

    def draw(self, img, tracks):
        # trails first (under the boxes); a lost track's trail keeps fading for `linger` frames
        for tid, pts in self.hist.items():
            pts = list(pts)
            n = len(pts)
            fade_lost = max(0.0, 1.0 - (self.frame - self.last_seen[tid]) / max(self.linger, 1))
            col = id_color(tid)
            for i in range(1, n):
                age = (i / (n - 1)) if n > 1 else 1.0  # 0 = oldest, 1 = newest
                alpha = float(np.clip(age ** 1.5 * 0.9 * fade_lost, 0, 1))
                thick = max(1, int(round(1 + (self.max_thick - 1) * age)))
                if alpha > 0.02:
                    blend_line(img, pts[i - 1], pts[i], col, thick, alpha)
        for t in tracks:
            x1, y1, x2, y2 = [int(round(v)) for v in t["box"]]
            col = id_color(t["id"])
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)
            label = f"#{t['id']} {self.names[t['cls']]} {t['score']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ty = max(y1 - 4, th + 4)
            cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), col, -1)
            cv2.putText(img, label, (x1 + 2, ty - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return img
