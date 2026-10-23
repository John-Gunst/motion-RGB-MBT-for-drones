"""Named model variants compared in the experiments (all share backbone, neck, heads and training recipe)."""

VARIANTS = {
    "rgb_only":      ["model.fusion=rgb"],
    "motion_only":   ["model.fusion=motion"],
    "early_fusion":  ["model.fusion=early"],
    "late_fusion":   ["model.fusion=late"],
    "joint_attn":    ["model.fusion=joint_attn", "model.joint_attn_pool=1"],
    "joint_attn_p2": ["model.fusion=joint_attn", "model.joint_attn_pool=2"],  # optional: if full-res OOMs
    "mbt_k4":        ["model.fusion=bottleneck", "model.bottleneck_tokens=4"],
    "mbt_k8":        ["model.fusion=bottleneck", "model.bottleneck_tokens=8"],
    "mbt_k4_nogate": ["model.fusion=bottleneck", "model.bottleneck_tokens=4", "model.gate.enabled=false"],
}

DEFAULT_ORDER = ["rgb_only", "motion_only", "early_fusion", "late_fusion", "joint_attn", "mbt_k4", "mbt_k8", "mbt_k4_nogate"]


def variant_overrides(name):
    if name is None:
        return []
    if name not in VARIANTS:
        raise KeyError(f"unknown variant {name!r}; choose from {list(VARIANTS)}")
    return list(VARIANTS[name])
