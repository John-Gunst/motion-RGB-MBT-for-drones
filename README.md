# MBTF — Multimodal Bottleneck Fusion for aerial target detection, classification & tracking

RGB appearance + 3-frame motion maps → frozen ImageNet backbone → **K learnable bottleneck tokens** fuse the two
streams in **O(N·K)** instead of O(N²) → CenterNet-style detection head + crop classifier → **SORT** tracking →
**fading motion trails**. A **Johnson-criteria gate** decides, per location, how much the fusion trusts motion vs.
appearance, and a two-tier scale-aware curriculum trains detection first, fine-grained classification second.

```
 frame t-2, t-1, t ──► 3-frame |ΔI| motion map ──┐
                                                 ├─► frozen ResNet (shared, cached) ─► neck_rgb / neck_mot  (B,d,H/8,W/8)
 frame t (RGB) ──────────────────────────────────┘                                         │
                                                                                            ▼
            ┌──────────────────────── Bottleneck fusion layer (×L) ─────────────────────────────────────┐
            │ gate g = σ(Gate([F_rgb,F_mot]))  ← supervised with Johnson prior at target centres        │
            │ Z (K tokens) ← CrossAttn(Z, [X_rgb ; X_mot], bias = [log g ; log(1-g)])   O(2N·K)        │
            │ X_rgb ← X_rgb + CrossAttn(X_rgb, Z) ; X_mot ← X_mot + CrossAttn(X_mot, Z)  O(N·K) each   │
            │ local depthwise mixing inside each modality                                O(N)         │
            │ fused = g·X_rgb + (1-g)·X_mot                                                             │
            └───────────────────────────────────────────────────────────────────────────────────────────┘
                       │                                             │
          CenterNet head (heatmap per class, w/h, offset)    crop classifier (Tier 2, high-res crops)
                       │                                             │
                       └──────► boxes ──► re-classify boxes ≥ 8 px ──► SORT ──► trails renderer
```

Nothing here trains on import — you run training on your GPU. Every script was smoke-tested on CPU with
tiny synthetic data (a few iterations, random-init backbone) to make sure the full pipeline executes.

---

## 1. Install (Fedora / any Linux)

```bash
python -m venv .venv && source .venv/bin/activate
# PyTorch for your CUDA version (see pytorch.org), e.g.:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## 2. Try the whole pipeline in ~5 minutes (synthetic data)

```bash
python scripts/make_synthetic.py --root data/synthetic --sequences 24 --frames 120
EXTRA="data.root=data/synthetic" VARIANTS="rgb_only motion_only mbt_k4" SCHEDULES="12" \
  bash scripts/run_experiments.sh
```
Synthetic drones/birds/bugs on a sky with a treeline (every 3rd sequence pans) — only for checking that
everything works on your machine before committing GPU hours to real data.

## 3. Real data

### Unified format
```
data/processed/sequences/<name>/frames/000000.jpg ...
data/processed/sequences/<name>/annotations.json
   {"width": W, "height": H, "classes": ["drone", ...],
    "frames": {"0": [[x1, y1, x2, y2, class_id, track_id], ...], ...}}
data/processed/splits/{train,val,test}.txt          # sequence names — never split one video across sets
```
Class ids are re-mapped by **name** onto `classes:` in the config, so sources with different class lists merge cleanly.
Motion maps need consecutive frames: convert with `--step 1` and subsample later with `data.frame_step_*`.

### Converters
```bash
# Anti-UAV (RGB-T): <src>/<seq>/visible.mp4 + visible.json {"exist": [...], "gt_rect": [[x,y,w,h], ...]}
python scripts/prepare_data.py antiuav --src ~/datasets/Anti-UAV/train --prefix auv

# Your own footage labelled in CVAT -> export "MOT 1.1" (gt.txt: frame,id,x,y,w,h,not_ignored,class,visibility)
python scripts/prepare_data.py mot --video backyard1.mp4 --labels gt/gt.txt --class-map 1:drone 2:bird 3:bug --name backyard1

# Sequential frames with YOLO labels (no track ids)
python scripts/prepare_data.py yolo --frames clip/images --labels clip/labels --names drone bird bug --name clip1

python scripts/prepare_data.py split --ratios 0.7 0.15 0.15
```

### Dataset caveats (read before you write the report)
* **VisDrone is captured *from* drones** — its targets are pedestrians/cars. It contains no drones, birds or bugs,
  so it is not useful for this task's classes. The `mot` converter will read VisDrone-MOT if you want it as
  background/negative footage, but don't count on it for positives.
* **Anti-UAV** has only a `drone` class. Birds and bugs must come from elsewhere: the Svanström drone-detection
  dataset (drone / bird / airplane / helicopter; its labels ship as MATLAB files, so export them to MOT-style text
  first), the Drone-vs-Bird challenge data, and your own static-camera footage (insects near the lens are easy
  to capture on a summer evening). Check each dataset's license and annotation format on your copy.
* With only 3 classes, **Top-5 accuracy is always 100%** — the code still reports it (and flags it). To make it
  meaningful, add classes (e.g. `classes: [drone, bird, bug, airplane, helicopter, balloon]`) or report Top-2.

## 4. Training (the scale-aware curriculum)

```bash
# 0) cache frozen-backbone features once (shared by every variant): ~4.9 MB/frame at 640 px with resnet18
python scripts/cache_features.py --splits train val test

# 1) Tier 1 — detection & motion. Small-target frames are oversampled (data.small_obj_oversample) and the
#    Johnson gate is supervised so targets < ~10 px lean on the motion stream.
python scripts/train.py --variant mbt_k4 --tier 1 --cached --epochs 72      # saves ckpt at 12/24/36/72

# 2) Tier 2 — classification on high-res crops (original resolution, targets >= 8 px = Johnson recognition)
python scripts/train.py --tier 2 --init runs/mbt_k4/sched72/ckpt_final.pt
```
* Without `--cached`, Tier 1 also applies **zoom-out augmentation** (`tier1.zoom_out`) that pushes targets into
  the < 10 px buckets — the full form of the curriculum, at the cost of running the backbone every step.
* `--set` overrides anything: `--set train.batch_size=8 model.dim=192 data.input_size=960`.
* Panning / handheld cameras: `--set data.motion.stabilize=true` (ORB + RANSAC homography before differencing).
  Changing motion settings or input size changes the cache key, so re-run `cache_features.py`.

### Variants compared (mbtf/variants.py)
| name | fusion | notes |
|---|---|---|
| `rgb_only`, `motion_only` | single modality | unimodal references |
| `early_fusion` | concat features → conv | feature-level early fusion |
| `late_fusion` | two streams + two heads, outputs averaged | |
| `joint_attn` | full self-attention over both modalities' tokens | the quadratic baseline (2N = 12,800 tokens at 640 px) |
| `joint_attn_p2` | same, tokens pooled 2× | optional fallback if `joint_attn` runs out of memory |
| `mbt_k4`, `mbt_k8` | **proposed**, K = 4 / 8 | |
| `mbt_k4_nogate` | proposed without Johnson gate | ablation of dynamic modality balancing |

All variants share the frozen backbone, the necks, the head and the training recipe; baseline conv stacks use the
same depth L so the comparison isolates the fusion strategy.

## 5. Everything in one go

```bash
bash scripts/run_experiments.sh                 # separate run per schedule (12/24/36/72), most faithful
MODE=single bash scripts/run_experiments.sh     # one 72-epoch run per variant, evaluated at 12/24/36/72 (~2x cheaper)
```
Steps: cache → Tier 1 for every variant × schedule → Tier 2 → test-set evaluation → efficiency benchmark →
`results/summary.md` + figures. Finished runs are skipped, so you can stop and resume.

**Budget tip:** separate schedules cost 12+24+36+72 = 144 epochs per variant. If time is tight, use `MODE=single`
(72 epochs per variant; the intermediate checkpoints are mid-schedule snapshots, so say so in the report), or run
the full grid only for `rgb_only`, `motion_only`, `joint_attn`, `mbt_k4`.

## 6. What gets measured (your list → where it comes from)

| Requested | Implementation | Output |
|---|---|---|
| #params | total, trainable, per part (frozen backbone, necks, fusion, heads, cls head) | `results/benchmark.csv` |
| step time | median optimiser-step wall time after warm-up, CUDA-synchronised; on cached features (what you actually train) and end-to-end from images | `benchmark.csv`, also logged per step in `train_log.csv` |
| M.Adds, GFLOPs | fvcore multiply-adds per frame + analytic attention cost (2·B·h·Lq·Lk·d_h); GFLOPs = 2 × MAdds; full pipeline, trainable part, and fusion block alone | `benchmark.csv` |
| AP^bb at 12/24/36/72 epochs | COCO AP@[.5:.95] (pycocotools), plus AP50, AP75, AR100, per-class AP | `eval_test_*.json` → summary table + `ap_vs_epochs.png` |
| AP for different drone sizes | COCO AP restricted to size buckets on √area in original pixels: tiny < 10, small 10–32, medium 32–96, large > 96 (`eval.size_buckets`) — for all classes and for `drone` alone | summary table + `drone_ap_by_size.png` |
| Precision, Recall, F1 | class-aware greedy matching at IoU 0.5, at the F1-optimal confidence and at a fixed 0.3 | summary table |
| Acc (detection) | (a) class accuracy of correctly localised targets, (b) frame-level presence accuracy | summary table |
| Top-1 / Top-5, Acc, P/R/F1 (classification) | Tier-2 crop classifier on test crops; macro P/R/F1, confusion matrix | summary table |
| FPS | batch-1 latency, full model | `benchmark.csv` |

### About AP^mk
AP^mk is mask AP, which requires instance segmentation masks. This model predicts boxes (detection), and none of the
suggested datasets ship masks, so **AP^mk is reported as n/a**. If your rubric requires it, the honest options are
(1) drop it with this explanation, or (2) add a mask head trained on masks you annotate yourself. Don't report box AP
under the mask-AP name.

### Reference compute (from the smoke test, random weights — these numbers depend only on architecture)
| variant @ 640 px | full GFLOPs | fusion GFLOPs |
|---|---|---|
| rgb_only | 72.8 | 30.3 |
| early_fusion | 105.7 | 32.0 |
| late_fusion | 145.6 | 60.5 |
| joint_attn | 445.5 | 371.8 |
| mbt_k4 | 109.1 | 35.4 |
| mbt_k8 | 109.3 | 35.6 |

From 320 → 640 px, MBT fusion cost grows 4× (linear in tokens) while joint attention grows ~12× (quadratic).
Doubling K from 4 to 8 costs < 1% extra. Step times and FPS must be measured on your GPU with `benchmark.py`.

## 7. Inference: detection → SORT → motion trails

```bash
python scripts/infer_video.py --ckpt runs/mbt_k4/tier2/ckpt_tier2.pt --source clip.mp4 --out out.mp4 \
       --show-motion --show-gate --save-tracks tracks.jsonl
python scripts/infer_video.py --ckpt ... --source 0 --display            # webcam
```
* Causal motion mode (default) uses frames t-2, t-1, t, so it runs online with zero latency.
* Boxes ≥ `tier2.reclassify_min_px` are re-classified by the high-res crop head (Johnson recognition threshold).
* SORT association enlarges boxes by `track.match_scale` before IoU — tiny fast targets rarely overlap between frames.
* `--show-gate` overlays the Johnson gate (red = relies on RGB, blue = relies on motion) — a good figure for the report.

## 8. Design notes

* **Motion map.** For current frame I_t and the other two frames I_a, I_b: channels (|I_t−I_a|, |I_t−I_b|,
  min of both). The min keeps only pixels that changed relative to *both* frames — the target at time t — and
  suppresses the ghost at its old position. Three channels let the same frozen ImageNet backbone encode it.
* **Johnson gate.** Target RGB weight = 0.1 + 0.8·σ((p − 10.4)/2), p = √(w·h) in input pixels. 10.4 px sits between
  recognition (4 line pairs ≈ 8 px) and identification (6.4 lp ≈ 12.8 px). It enters the bottleneck attention as an
  additive log-prior on the keys, so at each location attention mass is split g : (1−g) between the RGB and motion key.
* **Why caching works.** Backbones are frozen, so their outputs never change; caching removes them from every training
  step. Disk cost ≈ 4.9 MB/frame (resnet18, 640 px, both modalities, fp16); raise `data.frame_step_train` if needed.
* **Detection head.** CenterNet at stride 8: per-class heatmap with focal loss, box size and sub-cell offset with L1.
  Anchor-free heads suit 3–10 px targets better than anchors.

## 9. Layout
```
configs/default.yaml        every hyper-parameter, commented
mbtf/data/                  motion maps, letterbox, sequences, targets, cache, crops, datasets
mbtf/models/                frozen backbone + necks, fusion blocks (proposed + baselines), detector, decode
mbtf/losses.py              focal + L1 + Johnson-gate BCE
mbtf/tracking/sort.py       SORT (Kalman + Hungarian), dependency-free
mbtf/viz/trails.py          alpha-blended fading trails
mbtf/metrics.py, engine.py  COCO AP by size/class, P/R/F1/Acc, Top-k
mbtf/profiling.py           params, MAdds/GFLOPs, step time, latency
scripts/                    prepare_data, make_synthetic, cache_features, train, evaluate, benchmark,
                            infer_video, summarize, run_experiments.sh
```

## 10. Troubleshooting
* **CUDA OOM on `joint_attn`** — expected at 640 px with batch 16; use `--set train.batch_size=4`, or the `joint_attn_p2` variant (say so in the report).
* **AP stays ~0 on real data** — check boxes in an annotations.json against a frame, check that `classes` names match,
  and check `motion` with `infer_video.py --show-motion` (pan → turn on `stabilize`).
* **NaN loss with fp16** — try `--set train.amp=false` or lower `train.lr`.
* **Cache errors after changing settings** — the cache key includes backbone, input size and motion settings; re-run `cache_features.py`.
