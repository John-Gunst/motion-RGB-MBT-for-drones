"""3-frame absolute-difference motion maps (optionally camera-motion compensated)."""
import cv2
import numpy as np

_ORB = None


def _orb():
    global _ORB
    if _ORB is None:
        _ORB = cv2.ORB_create(1000)
    return _ORB


def align(ref_gray, mov_gray, min_matches=12):
    """Warp `mov_gray` onto `ref_gray` with a RANSAC homography (global motion compensation).

    Returns (warped, valid_mask). Falls back to identity if not enough features match
    (e.g. featureless sky) — in that case the camera is usually close to static anyway.
    """
    h, w = ref_gray.shape
    ident = (mov_gray, np.full_like(mov_gray, 255))
    orb = _orb()
    k1, d1 = orb.detectAndCompute(ref_gray, None)
    k2, d2 = orb.detectAndCompute(mov_gray, None)
    if d1 is None or d2 is None or len(k1) < min_matches or len(k2) < min_matches:
        return ident
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d2, d1)
    if len(matches) < min_matches:
        return ident
    matches = sorted(matches, key=lambda m: m.distance)[:300]
    src = np.float32([k2[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([k1[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None:
        return ident
    warped = cv2.warpPerspective(mov_gray, H, (w, h), flags=cv2.INTER_LINEAR)
    valid = cv2.warpPerspective(np.full_like(mov_gray, 255), H, (w, h), flags=cv2.INTER_NEAREST)
    valid = cv2.erode(valid, np.ones((5, 5), np.uint8))
    return warped, valid


def triplet_indices(t, n, mode="causal"):
    """Frame indices (oldest -> newest) used to build the motion map for frame t of an n-frame sequence."""
    if mode == "causal":
        return [max(t - 2, 0), max(t - 1, 0), t]
    if mode == "centered":
        return [max(t - 1, 0), t, min(t + 1, n - 1)]
    raise ValueError(f"unknown motion mode {mode!r}")


def current_slot(mode):
    """Position of the 'current' frame t inside the triplet."""
    return 2 if mode == "causal" else 1


def three_frame_motion(frames_bgr, mode="causal", gain=4.0, blur=3, stabilize=False):
    """Classical three-frame differencing.

    For the current frame I_t and the two other frames I_a, I_b of the triplet:
        D_a = |I_t - I_a|,  D_b = |I_t - I_b|,  D_min = min(D_a, D_b)
    D_min keeps only pixels that changed w.r.t. *both* other frames, i.e. the object at time t,
    and suppresses the 'ghost' at the previous positions. All three are kept as channels so a
    3-channel ImageNet backbone can consume the map unchanged.

    Returns HxWx3 float32 in [0, 1].
    """
    g = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames_bgr]
    if blur and blur > 1:
        k = int(blur) | 1
        g = [cv2.GaussianBlur(x, (k, k), 0) for x in g]
    s = current_slot(mode)
    cur = g[s]
    others = [g[i] for i in range(3) if i != s]
    valid = np.full(cur.shape, 255, np.uint8)
    if stabilize:
        aligned = []
        for o in others:
            w_, v_ = align(cur, o)
            aligned.append(w_)
            valid = cv2.bitwise_and(valid, v_)
        others = aligned
    da = cv2.absdiff(cur, others[0]).astype(np.float32) / 255.0
    db = cv2.absdiff(cur, others[1]).astype(np.float32) / 255.0
    m = np.stack([da, db, np.minimum(da, db)], -1) * gain
    m[valid == 0] = 0.0
    return np.clip(m, 0.0, 1.0)
