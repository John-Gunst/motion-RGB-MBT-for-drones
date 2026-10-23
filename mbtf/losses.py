import torch
import torch.nn.functional as F


def focal_loss(logits, gt):
    """CenterNet penalty-reduced focal loss."""
    pred = logits.float().sigmoid().clamp(1e-4, 1 - 1e-4)
    pos = gt.eq(1).float()
    neg = 1.0 - pos
    pos_loss = torch.log(pred) * (1 - pred) ** 2 * pos
    neg_loss = torch.log(1 - pred) * pred ** 2 * (1 - gt) ** 4 * neg
    n = pos.sum()
    return -(pos_loss.sum() + neg_loss.sum()) / n.clamp(min=1.0)


def gather_feat(x, ind):
    B, C = x.shape[:2]
    return x.reshape(B, C, -1).gather(2, ind[:, None].expand(-1, C, -1)).transpose(1, 2)  # (B, M, C)


def reg_l1(pred, ind, mask, target):
    p = gather_feat(pred.float(), ind)
    m = mask[..., None]
    return (torch.abs(p - target) * m).sum() / (m.sum() * p.shape[-1]).clamp(min=1e-4)


def det_loss(out, batch, w):
    hm = focal_loss(out["hm"], batch["hm"])
    wh = reg_l1(out["wh"], batch["ind"], batch["mask"], batch["wh"])
    off = reg_l1(out["off"], batch["ind"], batch["mask"], batch["off"])
    return {"hm": hm, "wh": wh, "off": off, "total": w["hm"] * hm + w["wh"] * wh + w["off"] * off}


def compute_loss(out, batch, cfg):
    w = cfg.train.loss
    if "branches" in out:  # late fusion: each stream is supervised independently
        parts = [det_loss(b, batch, w) for b in out["branches"]]
        losses = {k: sum(p[k] for p in parts) / len(parts) for k in parts[0]}
    else:
        losses = det_loss(out, batch, w)
    if "gate_logit" in out:
        g = gather_feat(out["gate_logit"].float(), batch["ind"]).squeeze(-1)
        m = batch["mask"]
        bce = F.binary_cross_entropy_with_logits(g, batch["gate_t"], reduction="none")
        gl = (bce * m).sum() / m.sum().clamp(min=1.0)
        losses["gate"] = gl
        losses["total"] = losses["total"] + cfg.model.gate.loss_weight * gl
    return losses
