#!/usr/bin/env bash
# Stage 1: right hand, monophonic. Three seeds of Double DQN, then evaluation and curves.
# Usage: bash run_seeds.sh data/PianoFingeringDataset_v1.2
set -euo pipefail
PIG=${1:?"give the PIG folder"}
for s in 0 1 2; do
  python train.py --pig-dir "$PIG" --hand R --seed "$s" --out "runs/rh_ddqn_s$s"
done
python evaluate.py --pig-dir "$PIG" --ckpt "runs/rh_ddqn_s*/best.pt" --show 2 --out results/rh_ddqn.json
python plot.py runs/rh_ddqn_s0 runs/rh_ddqn_s1 runs/rh_ddqn_s2 --out results/rh_ddqn_curves.png
