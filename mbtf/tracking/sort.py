"""SORT: Simple Online and Realtime Tracking (Bewley et al., 2016), dependency-free (numpy + scipy).

Additions for small aerial targets: boxes are enlarged by `match_scale` before IoU association (tiny
fast-moving boxes rarely overlap frame-to-frame), and each track keeps a score-weighted class vote.
"""
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(a, b):
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


def expand(boxes, s):
    if s == 1.0 or len(boxes) == 0:
        return boxes
    c = (boxes[:, :2] + boxes[:, 2:4]) / 2
    half = (boxes[:, 2:4] - boxes[:, :2]) / 2 * s
    return np.concatenate([c - half, c + half], 1)


def _to_z(b):
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, max(w * h, 1e-6), w / max(h, 1e-6)], np.float64)


def _to_box(x):
    s, r = max(x[2], 1e-6), max(x[3], 1e-6)
    w = np.sqrt(s * r)
    h = s / max(w, 1e-6)
    return np.array([x[0] - w / 2, x[1] - h / 2, x[0] + w / 2, x[1] + h / 2])


class KalmanBoxTracker:
    """Constant-velocity Kalman filter on (u, v, s, r) with velocities for u, v, s (as in the SORT paper)."""
    count = 0

    def __init__(self, det):
        self.F = np.eye(7)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = 1.0
        self.H = np.eye(4, 7)
        self.R = np.eye(4)
        self.R[2:, 2:] *= 10.0
        self.P = np.eye(7) * 10.0
        self.P[4:, 4:] *= 1000.0
        self.Q = np.eye(7)
        self.Q[-1, -1] *= 0.01
        self.Q[4:, 4:] *= 0.01
        self.x = np.zeros(7)
        self.x[:4] = _to_z(det)
        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        self.time_since_update = 0
        self.hits = self.hit_streak = self.age = 0
        self.votes = defaultdict(float)
        self.score = float(det[4])
        self.votes[int(det[5])] += float(det[4])

    def predict(self):
        if self.x[6] + self.x[2] <= 0:
            self.x[6] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return _to_box(self.x)

    def update(self, det):
        z = _to_z(det)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(7) - K @ self.H) @ self.P
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.score = float(det[4])
        self.votes[int(det[5])] += float(det[4])

    @property
    def cls(self):
        return max(self.votes, key=self.votes.get)

    def box(self):
        return _to_box(self.x)


class Sort:
    def __init__(self, max_age=15, min_hits=3, iou_threshold=0.1, match_scale=2.0):
        self.max_age, self.min_hits = max_age, min_hits
        self.iou_threshold, self.match_scale = iou_threshold, match_scale
        self.trackers = []
        self.frame_count = 0

    def update(self, dets):
        """dets: (n, 6) [x1, y1, x2, y2, score, cls]. Returns list of dicts for confirmed, currently-matched tracks."""
        self.frame_count += 1
        dets = np.asarray(dets, np.float64).reshape(-1, 6)
        preds = []
        alive = []
        for t in self.trackers:
            p = t.predict()
            if np.all(np.isfinite(p)):
                preds.append(p)
                alive.append(t)
        self.trackers = alive
        preds = np.array(preds).reshape(-1, 4)

        iou = iou_matrix(expand(dets[:, :4], self.match_scale), expand(preds, self.match_scale))
        matched, um_d = [], set(range(len(dets)))
        if iou.size:
            r, c = linear_sum_assignment(-iou)
            for i, j in zip(r, c):
                if iou[i, j] >= self.iou_threshold:
                    matched.append((i, j))
                    um_d.discard(i)
        for i, j in matched:
            self.trackers[j].update(dets[i])
        for i in sorted(um_d):
            self.trackers.append(KalmanBoxTracker(dets[i]))

        out = []
        for t in self.trackers:
            if t.time_since_update == 0 and (t.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                out.append({"id": t.id, "box": t.box(), "cls": t.cls, "score": t.score})
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
        return out
