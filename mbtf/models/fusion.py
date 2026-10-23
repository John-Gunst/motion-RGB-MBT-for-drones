"""Fusion blocks: the proposed Multimodal Bottleneck Fusion and the baselines it is compared against.

Every block maps per-modality feature maps (B, d, H, W) -> fused map (B, d, H, W) plus an aux dict.
Cost per layer with N = H*W tokens per modality:
    joint_attn : O((2N)^2 d)      full self-attention over both modalities (quadratic)
    bottleneck : O(N K d)         K latent tokens read both modalities, modalities read back from K tokens
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import conv_gn_act

USE_SDPA = True
ATTN_MACS = None  # when a list, each attention call appends its analytic MAC count (used by the FLOP counter)


def attention(q, k, v, bias=None):
    if ATTN_MACS is not None:  # QK^T and AV: each B*h*Lq*Lk*dh multiply-accumulates
        B, h, Lq, dh = q.shape
        ATTN_MACS.append(2 * B * h * Lq * k.shape[2] * dh)
    if USE_SDPA:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=None if bias is None else bias.to(q.dtype))
    a = torch.matmul(q, k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
    if bias is not None:
        a = a + bias.to(a.dtype)
    return torch.matmul(a.softmax(-1), v)


class MHA(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        assert d % heads == 0
        self.h = heads
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.o = nn.Linear(d, d)

    def forward(self, x, ctx, bias=None):
        B, Lq, D = x.shape
        Lk, dh = ctx.shape[1], D // self.h
        q = self.q(x).view(B, Lq, self.h, dh).transpose(1, 2)
        k, v = self.kv(ctx).view(B, Lk, 2, self.h, dh).permute(2, 0, 3, 1, 4)
        out = attention(q, k, v, bias)
        return self.o(out.transpose(1, 2).reshape(B, Lq, D))


class FFN(nn.Module):
    def __init__(self, d, ratio=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d * ratio), nn.GELU(), nn.Linear(d * ratio, d))

    def forward(self, x):
        return self.net(x)


class LocalMix(nn.Module):
    """Linear-cost intra-modality mixing on the token grid (depthwise 3x3 + pointwise MLP)."""

    def __init__(self, d, ratio=2):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.dw = nn.Conv2d(d, d, 3, padding=1, groups=d)
        self.mlp = nn.Sequential(nn.Linear(d, d * ratio), nn.GELU(), nn.Linear(d * ratio, d))

    def forward(self, x, H, W):
        B, N, D = x.shape
        y = self.norm(x).transpose(1, 2).reshape(B, D, H, W)
        y = self.dw(y).flatten(2).transpose(1, 2)
        return x + self.mlp(y)


def sincos_2d(H, W, d, device, dtype=torch.float32):
    assert d % 4 == 0
    y, x = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32),
                          torch.arange(W, device=device, dtype=torch.float32), indexing="ij")
    omega = 1.0 / (10000 ** (torch.arange(d // 4, device=device, dtype=torch.float32) / (d // 4)))
    xo, yo = x.flatten()[:, None] * omega, y.flatten()[:, None] * omega
    return torch.cat([xo.sin(), xo.cos(), yo.sin(), yo.cos()], 1).to(dtype)  # (H*W, d)


def to_tokens(x):
    return x.flatten(2).transpose(1, 2)


def to_map(t, H, W):
    B, N, D = t.shape
    return t.transpose(1, 2).reshape(B, D, H, W)


# ---------------------------------------------------------------------------------------------------------------
# Proposed: Multimodal Bottleneck Fusion with Johnson-criteria dynamic modality balancing
# ---------------------------------------------------------------------------------------------------------------
class ModalityGate(nn.Module):
    """Per-location RGB weight g in (0,1); motion weight is 1-g. Supervised at target centres with the
    Johnson-criteria prior (small/distant -> motion, large/close -> appearance)."""

    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(conv_gn_act(2 * d, d // 4, 1), nn.Conv2d(d // 4, 1, 3, padding=1))
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, fr, fm):
        return self.net(torch.cat([fr, fm], 1))  # logits (B,1,H,W)


class BottleneckLayer(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln_z, self.z_attn = nn.LayerNorm(d), MHA(d, heads)
        self.ln_z2, self.z_ffn = nn.LayerNorm(d), FFN(d)
        self.ln_kv = nn.LayerNorm(d)
        self.ln_r, self.r_attn, self.r_mix = nn.LayerNorm(d), MHA(d, heads), LocalMix(d)
        self.ln_m, self.m_attn, self.m_mix = nn.LayerNorm(d), MHA(d, heads), LocalMix(d)

    def forward(self, z, xr, xm, H, W, bias):
        # 1) the K bottleneck tokens query BOTH modalities; the gate enters as an additive log-prior on the keys,
        #    so per location the attention mass is split g : (1-g) between the RGB and motion key.
        ctx = self.ln_kv(torch.cat([xr, xm], 1))
        z = z + self.z_attn(self.ln_z(z), ctx, bias)
        z = z + self.z_ffn(self.ln_z2(z))
        # 2) each modality reads back ONLY from the bottleneck (cross-modal info must pass through K tokens)
        xr = xr + self.r_attn(self.ln_r(xr), z)
        xm = xm + self.m_attn(self.ln_m(xm), z)
        # 3) cheap local mixing inside each modality
        return z, self.r_mix(xr, H, W), self.m_mix(xm, H, W)


class BottleneckFusion(nn.Module):
    def __init__(self, d, heads=8, layers=2, k=4, gate=True):
        super().__init__()
        self.k, self.use_gate = k, gate
        self.z0 = nn.Parameter(torch.randn(k, d) * 0.02)
        self.mod_emb = nn.Parameter(torch.zeros(2, d))
        self.layers = nn.ModuleList([BottleneckLayer(d, heads) for _ in range(layers)])
        self.gate = ModalityGate(d) if gate else None
        self.out = conv_gn_act(d, d, 3)

    def forward(self, fr, fm):
        B, D, H, W = fr.shape
        pos = sincos_2d(H, W, D, fr.device, fr.dtype)
        xr = to_tokens(fr) + pos + self.mod_emb[0]
        xm = to_tokens(fm) + pos + self.mod_emb[1]
        z = self.z0.unsqueeze(0).expand(B, -1, -1)
        aux = {}
        bias = None
        if self.use_gate:
            g_logit = self.gate(fr, fm)
            g = g_logit.float().sigmoid().clamp(1e-4, 1 - 1e-4)
            lg = to_tokens(g).squeeze(-1)  # (B, N)
            bias = torch.cat([lg.log(), (1 - lg).log()], 1)[:, None, None, :]  # (B,1,1,2N) broadcast heads/queries
            aux["gate_logit"] = g_logit
        else:
            g = torch.full((B, 1, H, W), 0.5, device=fr.device, dtype=torch.float32)
        for layer in self.layers:
            z, xr, xm = layer(z, xr, xm, H, W, bias)
        g = g.to(fr.dtype)
        fused = g * to_map(xr, H, W) + (1 - g) * to_map(xm, H, W)
        aux["gate"] = g
        aux["bottleneck"] = z
        return self.out(fused), aux


# ---------------------------------------------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.a = conv_gn_act(d, d, 3)
        self.b = nn.Sequential(nn.Conv2d(d, d, 3, padding=1, bias=False), nn.GroupNorm(32, d))

    def forward(self, x):
        return F.gelu(x + self.b(self.a(x)))


class SingleModality(nn.Module):
    def __init__(self, d, layers, which):
        super().__init__()
        self.which = which
        self.body = nn.Sequential(*[ResBlock(d) for _ in range(layers)])

    def forward(self, fr, fm):
        return self.body(fr if self.which == "rgb" else fm), {}


class EarlyFusion(nn.Module):
    """Feature-level early fusion: concatenate, project, process jointly."""

    def __init__(self, d, layers):
        super().__init__()
        self.proj = conv_gn_act(2 * d, d, 1)
        self.body = nn.Sequential(*[ResBlock(d) for _ in range(layers)])

    def forward(self, fr, fm):
        return self.body(self.proj(torch.cat([fr, fm], 1))), {}


class LateFusion(nn.Module):
    """Two independent streams; their detection heads' outputs are averaged by the detector."""

    def __init__(self, d, layers):
        super().__init__()
        self.r = nn.Sequential(*[ResBlock(d) for _ in range(layers)])
        self.m = nn.Sequential(*[ResBlock(d) for _ in range(layers)])

    def forward(self, fr, fm):
        return (self.r(fr), self.m(fm)), {}


class JointLayer(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln1, self.attn = nn.LayerNorm(d), MHA(d, heads)
        self.ln2, self.ffn = nn.LayerNorm(d), FFN(d)

    def forward(self, x):
        h = self.ln1(x)
        x = x + self.attn(h, h)
        return x + self.ffn(self.ln2(x))


class JointAttnFusion(nn.Module):
    """Full self-attention over the concatenation of both modalities' tokens (quadratic in 2N)."""

    def __init__(self, d, heads=8, layers=2, pool=2):
        super().__init__()
        self.pool = pool
        self.mod_emb = nn.Parameter(torch.zeros(2, d))
        self.layers = nn.ModuleList([JointLayer(d, heads) for _ in range(layers)])
        self.merge = conv_gn_act(2 * d, d, 1)
        self.out = conv_gn_act(d, d, 3)

    def forward(self, fr, fm):
        B, D, H0, W0 = fr.shape
        if self.pool > 1:
            fr_, fm_ = F.avg_pool2d(fr, self.pool), F.avg_pool2d(fm, self.pool)
        else:
            fr_, fm_ = fr, fm
        H, W = fr_.shape[-2:]
        pos = sincos_2d(H, W, D, fr.device, fr.dtype)
        x = torch.cat([to_tokens(fr_) + pos + self.mod_emb[0], to_tokens(fm_) + pos + self.mod_emb[1]], 1)
        for layer in self.layers:
            x = layer(x)
        N = H * W
        r, m = to_map(x[:, :N], H, W), to_map(x[:, N:], H, W)
        if self.pool > 1:
            r = F.interpolate(r, size=(H0, W0), mode="bilinear", align_corners=False) + fr
            m = F.interpolate(m, size=(H0, W0), mode="bilinear", align_corners=False) + fm
        return self.out(self.merge(torch.cat([r, m], 1))), {}


def build_fusion(m):
    t, d, L, h = m.fusion, m.dim, m.layers, m.heads
    if t in ("rgb", "motion"):
        return SingleModality(d, L, t)
    if t == "early":
        return EarlyFusion(d, L)
    if t == "late":
        return LateFusion(d, L)
    if t == "joint_attn":
        return JointAttnFusion(d, h, L, m.get("joint_attn_pool", 2))
    if t == "bottleneck":
        return BottleneckFusion(d, h, L, m.bottleneck_tokens, m.gate.get("enabled", True))
    raise ValueError(f"unknown fusion {t!r}")
