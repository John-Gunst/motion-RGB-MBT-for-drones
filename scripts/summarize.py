#!/usr/bin/env python
"""Collect eval JSONs + benchmark.csv into report tables (Markdown + CSV) and figures.

    python scripts/summarize.py --runs runs --bench results/benchmark.csv --out results
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mbtf.variants import DEFAULT_ORDER  # noqa: E402

# fixed categorical order (colour follows the variant, never its rank)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
EPOCHS = [12, 24, 36, 72]


def fmt(v, pct=True, nd=1):
    if v is None or v == "":
        return "–"
    try:
        v = float(v)
    except ValueError:
        return str(v)
    return f"{100 * v:.{nd}f}" if pct else f"{v:.{nd}f}"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def load_evals(runs, split):
    det, cls = {}, {}
    for p in sorted(Path(runs).rglob(f"eval_{split}_*.json")):
        r = json.loads(p.read_text())
        v = r.get("variant") or p.relative_to(runs).parts[0]
        ep, sch = r.get("epoch"), r.get("schedule")
        if "classification" in r:
            cls[v] = r
            continue
        if ep is None:
            continue
        key = (v, int(ep))
        prev = det.get(key)
        # prefer a run whose whole schedule ended at this epoch (proper LR decay) over a mid-run snapshot
        if prev is None or (sch == ep and prev.get("schedule") != prev.get("epoch")):
            det[key] = r
    return det, cls


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--bench", default="results/benchmark.csv")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    det, cls = load_evals(a.runs, a.split)
    global EPOCHS
    EPOCHS = sorted({k[1] for k in det}) or EPOCHS
    variants = [v for v in DEFAULT_ORDER if any(k[0] == v for k in det) or v in cls]
    variants += sorted({k[0] for k in det} - set(variants))
    bench = {}
    if Path(a.bench).exists():
        with open(a.bench) as f:
            bench = {r["variant"]: r for r in csv.DictReader(f)}
        variants += [v for v in DEFAULT_ORDER if v in bench and v not in variants]
    md = [f"# Results ({a.split} split)\n"]

    if bench:
        rows = []
        for v in variants:
            b = bench.get(v)
            if not b:
                continue
            rows.append([v, f"{int(b['params_total']) / 1e6:.2f}", f"{int(b['params_trainable_det']) / 1e6:.2f}",
                         fmt(b["gmadds_full"], False, 2), fmt(b["gflops_full"], False, 2),
                         fmt(b["gflops_fusion"], False, 3), fmt(b["step_ms_cached"], False, 1),
                         fmt(b.get("step_ms_e2e"), False, 1), fmt(b["fps"], False, 1)])
        md += ["## Efficiency", md_table(["Model", "#Params (M)", "Trainable (M)", "MAdds (G)", "GFLOPs",
                                          "Fusion GFLOPs", "Step (ms, cached)", "Step (ms, e2e)", "FPS"], rows),
               "\nMAdds/GFLOPs per frame at the configured input size; GFLOPs = 2 x MAdds. "
               "Step = one optimiser step at train.batch_size.\n"]

    rows = []
    for v in variants:
        rows.append([v] + [fmt(det.get((v, e), {}).get("detection", {}).get("bbox", {}).get("AP")) for e in EPOCHS]
                    + ["n/a"])
    md += ["## AP^bb vs training schedule", md_table(["Model"] + [f"AP^bb @{e}ep" for e in EPOCHS] + ["AP^mk"], rows),
           "\nAP^mk is n/a: the model predicts boxes, not instance masks (see README).\n"]

    final = {v: max((k[1] for k in det if k[0] == v), default=None) for v in variants}
    buckets = None
    rows = []
    for v in variants:
        e = final[v]
        if e is None:
            continue
        b = det[(v, e)]["detection"]["bbox"]
        buckets = buckets or [k[len("AP_drone_"):] for k in b if k.startswith("AP_drone_")]
        rows.append([v, e, fmt(b["AP"]), fmt(b["AP50"]), fmt(b["AP75"])] +
                    [fmt(b.get(f"AP_drone_{s}")) for s in buckets] + [fmt(b.get(f"AP_{s}")) for s in buckets])
    if rows:
        md += ["## AP^bb by target size (final checkpoint)",
               md_table(["Model", "Epochs", "AP", "AP50", "AP75"] + [f"Drone AP {s}" for s in buckets]
                        + [f"All AP {s}" for s in buckets], rows), ""]

    rows = []
    for v in variants:
        e = final[v]
        if e is None:
            continue
        pr = det[(v, e)]["detection"]["prf"]["best"]
        rows.append([v, fmt(pr["precision"]), fmt(pr["recall"]), fmt(pr["f1"]), fmt(pr["cls_acc_of_localized"]),
                     fmt(pr["frame_acc"]), pr["threshold"]])
    if rows:
        md += ["## Detection precision / recall / F1 / accuracy (IoU 0.5, F1-optimal threshold)",
               md_table(["Model", "Precision", "Recall", "F1", "Cls acc (localised)", "Frame acc", "Conf thr"], rows),
               ""]

    rows = []
    for v in variants:
        c = cls.get(v, {}).get("classification")
        if not c:
            continue
        rows.append([v, fmt(c.get("top1")), fmt(c.get("top5")), fmt(c.get("accuracy")), fmt(c.get("macro_precision")),
                     fmt(c.get("macro_recall")), fmt(c.get("macro_f1"))])
    if rows:
        md += ["## Tier-2 crop classification", md_table(["Model", "Top-1", "Top-5", "Acc", "Precision (macro)",
                                                           "Recall (macro)", "F1 (macro)"], rows),
               "\nWith <= 5 classes Top-5 is trivially 100% — add classes or report Top-2.\n"]
    (out / "summary.md").write_text("\n".join(md))
    print((out / "summary.md").read_text())
    plots(det, bench, variants, final, out)


def plots(det, bench, variants, final, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping figures")
        return
    color = {v: PALETTE[DEFAULT_ORDER.index(v) % 8] if v in DEFAULT_ORDER else "#777777" for v in variants}
    style = {"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#888888",
             "axes.labelcolor": "#333333", "xtick.color": "#555555", "ytick.color": "#555555",
             "grid.color": "#e6e6e6", "font.size": 10}
    with plt.rc_context(style):
        # AP vs schedule
        fig, ax = plt.subplots(figsize=(7, 4.2))
        any_line = False
        for v in variants:
            pts = [(e, det[(v, e)]["detection"]["bbox"]["AP"]) for e in EPOCHS if (v, e) in det
                   and det[(v, e)]["detection"]["bbox"]["AP"] is not None]
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, [100 * y for y in ys], "-o", lw=2, ms=6, color=color[v], label=v,
                        markeredgecolor="white", markeredgewidth=1.5)
                any_line = True
        if any_line:
            ax.set_xticks(EPOCHS)
            ax.set_xlabel("Training epochs")
            ax.set_ylabel("AP^bb (%)")
            ax.set_title("Box AP vs schedule", loc="left")
            ax.grid(axis="y")
            ax.legend(frameon=False, fontsize=8, ncol=2)
            fig.tight_layout()
            fig.savefig(out / "ap_vs_epochs.png", dpi=160)
        plt.close(fig)

        # drone AP by size (small multiples: one panel per size bucket, shared y)
        vs = [v for v in variants if final.get(v) is not None]
        if vs:
            b0 = det[(vs[0], final[vs[0]])]["detection"]["bbox"]
            buckets = [k[len("AP_drone_"):] for k in b0 if k.startswith("AP_drone_")]
            if buckets:
                fig, axes = plt.subplots(1, len(buckets), figsize=(3 * len(buckets), 3.6), sharey=True,
                                         squeeze=False)
                axes = axes[0]
                for ax, s in zip(axes, buckets):
                    vals = [det[(v, final[v])]["detection"]["bbox"].get(f"AP_drone_{s}") for v in vs]
                    vals = [0.0 if x is None else 100 * x for x in vals]
                    ax.bar(range(len(vs)), vals, color=[color[v] for v in vs], width=0.8, edgecolor="white", lw=2)
                    ax.set_xticks(range(len(vs)))
                    ax.set_xticklabels(vs, rotation=60, ha="right", fontsize=8)
                    ax.set_title(f"{s}", loc="left")
                    ax.grid(axis="y")
                axes[0].set_ylabel("Drone AP^bb (%)")
                fig.suptitle("Drone AP by target size (final checkpoint)", x=0.01, ha="left")
                fig.tight_layout()
                fig.savefig(out / "drone_ap_by_size.png", dpi=160)
                plt.close(fig)

        # accuracy vs compute
        pts = [(v, float(bench[v]["gflops_full"]), det[(v, final[v])]["detection"]["bbox"]["AP"])
               for v in vs if v in bench and det[(v, final[v])]["detection"]["bbox"]["AP"] is not None]
        if pts:
            fig, ax = plt.subplots(figsize=(6.5, 4.2))
            for v, g, ap in pts:
                ax.scatter(g, 100 * ap, s=70, color=color[v], edgecolor="white", lw=2, zorder=3, label=v)
            ax.set_xlabel("GFLOPs per frame")
            ax.set_ylabel("AP^bb (%)")
            ax.set_title("Accuracy vs compute", loc="left")
            ax.grid()
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            fig.savefig(out / "ap_vs_gflops.png", dpi=160)
            plt.close(fig)
    print(f"figures -> {out}")


if __name__ == "__main__":
    main()
