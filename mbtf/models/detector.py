import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.targets import STRIDE
from .backbones import FrozenResNet, Neck
from .fusion import build_fusion


def _branch(d, cout, mid=128, bias=0.0):
    head = nn.Sequential(nn.Conv2d(d, mid, 3, padding=1), nn.GELU(), nn.Conv2d(mid, cout, 1))
    nn.init.constant_(head[-1].bias, bias)
    return head


class CenterHead(nn.Module):
    """Anchor-free (CenterNet) head: class heatmap + box size + sub-cell offset at stride 8."""

    def __init__(self, d, num_classes):
        super().__init__()
        self.hm = _branch(d, num_classes, bias=-2.19)
        self.wh = _branch(d, 2)
        self.off = _branch(d, 2)

    def forward(self, x):
        return {"hm": self.hm(x), "wh": self.wh(x), "off": self.off(x)}


class CropClassifier(nn.Module):
    def __init__(self, d, num_classes, hidden=256, p=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, hidden), nn.GELU(), nn.Dropout(p),
                                 nn.Linear(hidden, num_classes))

    def forward(self, fmap):
        x = torch.cat([fmap.mean((2, 3)), fmap.amax((2, 3))], 1)
        return self.net(x)


class MBTFDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg.model
        self.fusion_type = m.fusion
        self.num_classes = len(cfg.classes)
        self.use_rgb = m.fusion != "motion"
        self.use_mot = m.fusion != "rgb"
        self.backbone = FrozenResNet(m.backbone, m.get("pretrained", True))
        c2, c3 = self.backbone.channels
        self.neck_rgb = Neck(c2, c3, m.dim) if self.use_rgb else None
        self.neck_mot = Neck(c2, c3, m.dim) if self.use_mot else None
        self.fusion = build_fusion(m)
        n_heads = 2 if m.fusion == "late" else 1
        self.heads = nn.ModuleList([CenterHead(m.dim, self.num_classes) for _ in range(n_heads)])
        self.cls_head = CropClassifier(m.dim, self.num_classes)

    # -- feature extraction (frozen) -------------------------------------------------------------------------
    def extract(self, rgb=None, mot=None):
        feats = {}
        if self.use_rgb:
            feats["rgb_l2"], feats["rgb_l3"] = self.backbone(rgb)
        if self.use_mot:
            feats["mot_l2"], feats["mot_l3"] = self.backbone(mot)
        return feats

    def fuse(self, feats):
        dt = next(self.fusion.parameters()).dtype
        fr = self.neck_rgb(feats["rgb_l2"].to(dt), feats["rgb_l3"].to(dt)) if self.use_rgb else None
        fm = self.neck_mot(feats["mot_l2"].to(dt), feats["mot_l3"].to(dt)) if self.use_mot else None
        return self.fusion(fr, fm)

    # -- detection --------------------------------------------------------------------------------------------
    def forward(self, rgb=None, mot=None, feats=None):
        if feats is None:
            feats = self.extract(rgb, mot)
        fused, aux = self.fuse(feats)
        if self.fusion_type == "late":
            branches = [h(f) for h, f in zip(self.heads, fused)]
            p = torch.stack([b["hm"].float().sigmoid() for b in branches]).mean(0).clamp(1e-6, 1 - 1e-6)
            out = {"hm": torch.log(p / (1 - p)),
                   "wh": torch.stack([b["wh"] for b in branches]).mean(0),
                   "off": torch.stack([b["off"] for b in branches]).mean(0),
                   "branches": branches}
        else:
            out = dict(self.heads[0](fused))
        out.update(aux)
        return out

    # -- Tier-2 crop classification ---------------------------------------------------------------------------
    def classify(self, rgb=None, mot=None, feats=None):
        if feats is None:
            feats = self.extract(rgb, mot)
        fused, _ = self.fuse(feats)
        if isinstance(fused, tuple):
            fused = (fused[0] + fused[1]) / 2
        return self.cls_head(fused)

    def param_groups(self):
        """Parameter ownership, used for freezing and for #params reporting."""
        groups = {"backbone (frozen)": self.backbone, "fusion": self.fusion, "det_heads": self.heads,
                  "cls_head": self.cls_head}
        necks = [n for n in (self.neck_rgb, self.neck_mot) if n is not None]
        groups["necks"] = nn.ModuleList(necks)
        return groups


@torch.no_grad()
def decode(out, K=100, stride=STRIDE):
    """Dense outputs -> top-K boxes (x1y1x2y2, INPUT pixels), scores, classes. Peak NMS via 3x3 max-pool."""
    hm = out["hm"].float().sigmoid()
    B, C, H, W = hm.shape
    keep = (F.max_pool2d(hm, 3, 1, 1) == hm).float()
    hm = hm * keep
    K = min(K, C * H * W)
    scores, inds = hm.reshape(B, -1).topk(K)
    cls = inds // (H * W)
    pix = inds % (H * W)
    ys, xs = (pix // W).float(), (pix % W).float()
    wh = out["wh"].float().reshape(B, 2, -1).gather(2, pix[:, None].expand(-1, 2, -1))
    off = out["off"].float().reshape(B, 2, -1).gather(2, pix[:, None].expand(-1, 2, -1))
    cx, cy = (xs + off[:, 0]) * stride, (ys + off[:, 1]) * stride
    w, h = wh[:, 0].clamp(min=0) * stride, wh[:, 1].clamp(min=0) * stride
    boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)
    return boxes, scores, cls
