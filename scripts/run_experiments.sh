#!/usr/bin/env bash
# Full experiment grid: cache -> Tier 1 (12/24/36/72 epochs) -> Tier 2 -> eval -> benchmark -> summary.
#
#   bash scripts/run_experiments.sh                      # all variants, separate run per schedule (most faithful)
#   MODE=single bash scripts/run_experiments.sh          # one 72-epoch run per variant, evaluated at 12/24/36/72 (~2x cheaper)
#   VARIANTS="rgb_only mbt_k4" bash scripts/run_experiments.sh
#   EXTRA="data.root=data/synthetic" bash scripts/run_experiments.sh
set -euo pipefail
cd "$(dirname "$0")/.."

VARIANTS=${VARIANTS:-"rgb_only motion_only early_fusion late_fusion joint_attn mbt_k4 mbt_k8 mbt_k4_nogate"}
SCHEDULES=${SCHEDULES:-"12 24 36 72"}
MODE=${MODE:-separate}
CONFIG=${CONFIG:-configs/default.yaml}
EXTRA=${EXTRA:-}
SET=(); [[ -n "$EXTRA" ]] && SET=(--set $EXTRA)

echo "== 1/5 feature cache"
python scripts/cache_features.py --config "$CONFIG" "${SET[@]}" --splits train val test --yes

echo "== 2/5 tier 1 (detection + motion)"
for v in $VARIANTS; do
  if [[ "$MODE" == "single" ]]; then
    last=$(echo $SCHEDULES | tr ' ' '\n' | sort -n | tail -1)
    [[ -f runs/$v/single/ckpt_final.pt ]] || \
      python scripts/train.py --config "$CONFIG" --variant "$v" --tier 1 --cached --epochs "$last" \
        --out "runs/$v/single" --set train.save_at="[${SCHEDULES// /,}]" $EXTRA
  else
    for e in $SCHEDULES; do
      [[ -f runs/$v/sched$e/ckpt_final.pt ]] || \
        python scripts/train.py --config "$CONFIG" --variant "$v" --tier 1 --cached --epochs "$e" \
          --out "runs/$v/sched$e" "${SET[@]}"
    done
  fi
done

echo "== 3/5 tier 2 (classification on high-res crops, from the longest schedule)"
for v in $VARIANTS; do
  if [[ "$MODE" == "single" ]]; then init=runs/$v/single/ckpt_final.pt
  else init=runs/$v/sched$(echo $SCHEDULES | tr ' ' '\n' | sort -n | tail -1)/ckpt_final.pt; fi
  [[ -f runs/$v/tier2/ckpt_tier2.pt ]] || \
    python scripts/train.py --tier 2 --init "$init" --out "runs/$v/tier2" "${SET[@]}"
done

echo "== 4/5 evaluation (test split)"
for v in $VARIANTS; do
  if [[ "$MODE" == "single" ]]; then ckpts=$(ls runs/$v/single/ckpt_e*.pt)
  else ckpts=$(ls runs/$v/sched*/ckpt_final.pt); fi
  python scripts/evaluate.py --ckpt $ckpts runs/$v/tier2/ckpt_tier2.pt --split test --cached "${SET[@]}"
done

echo "== 5/5 efficiency benchmark + summary"
python scripts/benchmark.py --config "$CONFIG" --variants $VARIANTS "${SET[@]}"
python scripts/summarize.py --runs runs --bench results/benchmark.csv --out results
