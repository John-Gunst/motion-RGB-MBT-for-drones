"""#params, multiply-adds (MAdds), GFLOPs and step time.

Conventions (state these in your report):
  * MAdds = multiply-accumulate operations counted by fvcore (conv, linear, matmul/bmm), plus the analytic
    attention cost 2*B*heads*Lq*Lk*d_head per attention call (fused attention kernels are opaque to fvcore);
    elementwise ops (norms, activations, softmax) are not counted, as is standard.
  * GFLOPs = 2 x GMAdds.
  * Step time = median wall time of one optimiser step (forward + backward + update) after warm-up,
    with CUDA synchronisation, measured on cached features (the configuration actually trained) and
    separately end-to-end from images (backbone included).
"""
import time

import numpy as np
import torch

from . import engine
from .models import fusion as fusion_mod


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable_det = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and not n.startswith("cls_head"))
    groups = {k: sum(p.numel() for p in m.parameters()) for k, m in model.param_groups().items()}
    return {"params_total": total, "params_trainable_det": trainable_det,
            "params_inference_det": total - groups["cls_head"], **{f"params_{k}": v for k, v in groups.items()}}


class _FullWrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, rgb, mot):
        o = self.m(rgb=rgb, mot=mot)
        return o["hm"], o["wh"], o["off"]


class _CachedWrap(torch.nn.Module):
    def __init__(self, m, keys):
        super().__init__()
        self.m, self.keys = m, keys

    def forward(self, *feats):
        o = self.m(feats=dict(zip(self.keys, feats)))
        return o["hm"], o["wh"], o["off"]


class _FusionWrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, fr, fm):
        f, _ = self.m.fusion(fr, fm)
        return f if not isinstance(f, tuple) else f[0] + f[1]


def _fca(module, inputs):
    from fvcore.nn import FlopCountAnalysis
    fca = FlopCountAnalysis(module, inputs)
    fca.unsupported_ops_warnings(False)
    fca.uncalled_modules_warnings(False)
    fusion_mod.ATTN_MACS = []
    try:
        total = float(fca.total())  # attention kernels are opaque to fvcore -> add their analytic MACs
        total += float(sum(fusion_mod.ATTN_MACS))
    finally:
        fusion_mod.ATTN_MACS = None
    return total


@torch.no_grad()
def count_madds(model, cfg, device):
    """Returns MAdds for: full pipeline (images -> outputs), trainable part (cached feats -> outputs), fusion block."""
    model.eval()
    input_size, d = cfg.data.input_size, cfg.model.dim
    if True:
        x = torch.randn(1, 3, input_size, input_size, device=device)
        full = _fca(_FullWrap(model), (x, x))
        feats = model.extract(x, x)
        keys = list(feats)
        cached = _fca(_CachedWrap(model, keys), tuple(feats[k] for k in keys))
        H = input_size // 8
        f = torch.randn(1, d, H, H, device=device)
        fus = _fca(_FusionWrap(model), (f, f))
    return {"gmadds_full": full / 1e9, "gflops_full": 2 * full / 1e9,
            "gmadds_trainable": cached / 1e9, "gflops_trainable": 2 * cached / 1e9,
            "gmadds_fusion": fus / 1e9, "gflops_fusion": 2 * fus / 1e9}


def _fake_batch(model, cfg, bs, device, cached):
    S, C, H = cfg.data.input_size, len(cfg.classes), cfg.data.input_size // 8
    M = cfg.model.max_objs
    b = {"hm": torch.zeros(bs, C, H, H, device=device), "wh": torch.rand(bs, M, 2, device=device) * 3,
         "off": torch.rand(bs, M, 2, device=device), "ind": torch.randint(0, H * H, (bs, M), device=device),
         "mask": (torch.rand(bs, M, device=device) < 0.05).float(), "gate_t": torch.rand(bs, M, device=device),
         "meta": None}
    b["hm"].view(bs, -1)[:, :: 997] = 1.0
    x = torch.randn(bs, 3, S, S, device=device)
    if cached:
        with torch.no_grad():
            b["feats"] = {k: v.half() for k, v in model.extract(x, x).items()}
    else:
        b["rgb"], b["mot"] = x, x.clone()
    return b


def step_time(model, cfg, device, bs, cached=True, iters=30, warmup=10, amp=True):
    from .losses import compute_loss
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4)
    batch = _fake_batch(model, cfg, bs, device, cached)
    use_scaler = amp and device.type == "cuda" and not torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    ts = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for i in range(warmup + iters):
        engine.sync(device)
        t0 = time.perf_counter()
        with engine.amp_ctx(device, amp):
            out = engine.run_model(model, batch)
            loss = compute_loss(out, batch, cfg)["total"]
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        engine.sync(device)
        if i >= warmup:
            ts.append(time.perf_counter() - t0)
    mem = torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else float("nan")
    return {"step_ms": float(np.median(ts) * 1000), "step_ms_std": float(np.std(ts) * 1000), "peak_mem_mb": mem}


@torch.no_grad()
def inference_latency(model, cfg, device, iters=50, warmup=10, amp=True):
    model.eval()
    S = cfg.data.input_size
    x = torch.randn(1, 3, S, S, device=device)
    ts = []
    for i in range(warmup + iters):
        engine.sync(device)
        t0 = time.perf_counter()
        with engine.amp_ctx(device, amp):
            model(rgb=x, mot=x)
        engine.sync(device)
        if i >= warmup:
            ts.append(time.perf_counter() - t0)
    ms = float(np.median(ts) * 1000)
    return {"infer_ms": ms, "fps": 1000.0 / ms}
