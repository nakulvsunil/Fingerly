"""Evaluate trained agents and baselines on the PIG test set (one hand, monophonic passages).

Every method is run on exactly the same passages, so the numbers are comparable to each other
(but not to published PIG numbers, which include chords and both hands).

    python evaluate.py --pig-dir data/PIG --ckpt runs/rh_ddqn_s*/best.pt --out results/rh.json
    python evaluate.py --pig-dir data/PIG --baselines-only --show 2
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from fingerly.baselines import HMM1, Greedy, OptimalCost, RandomValid
from fingerly.cost import CostConfig
from fingerly.data import build_split
from fingerly.env import FingeringEnv, run_policy
from fingerly.metrics import evaluate_predictor, human_agreement

COLUMNS = ["Mgen", "Mhigh", "Msoft", "effort_per_note", "impossible_rate",
           "crossing_rate", "shift_rate", "weak_finger_rate"]


def fmt(v, pct):
    return f"{100 * v:6.2f}" if pct else f"{v:6.3f}"


def print_table(rows):
    pct = {"Mgen", "Mhigh", "Msoft", "impossible_rate", "crossing_rate", "shift_rate", "weak_finger_rate"}
    head = f"{'method':<28}" + "".join(f"{c[:14]:>16}" for c in COLUMNS)
    print(head)
    print("-" * len(head))
    for name, m, sd in rows:
        cells = []
        for c in COLUMNS:
            if c not in m:
                cells.append(f"{'-':>16}")
                continue
            s = fmt(m[c], c in pct)
            if sd and c in sd:
                s += "±" + fmt(sd[c], c in pct).strip()
            cells.append(f"{s:>16}")
        print(f"{name:<28}" + "".join(cells))
    print("(match rates, impossible/crossing/shift/weak-finger rates in %; effort in cost units per note)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pig-dir", required=True)
    ap.add_argument("--ckpt", nargs="*", default=[], help="checkpoint paths or globs (seeds are averaged)")
    ap.add_argument("--hand", choices=["R", "L"], default=None, help="default: taken from the checkpoint")
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--min-len", type=int, default=None)
    ap.add_argument("--val-pieces", type=int, default=None)
    ap.add_argument("--split-seed", type=int, default=None)
    ap.add_argument("--baselines-only", action="store_true")
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--show", type=int, default=0, help="print N example passages with all fingerings")
    ap.add_argument("--out", default=None, help="write all numbers to this JSON file")
    args = ap.parse_args()

    ckpts = sorted({p for pat in args.ckpt for p in glob.glob(pat)})
    if args.ckpt and not ckpts:
        raise SystemExit(f"no checkpoints match {args.ckpt}")
    agents = []
    if ckpts and not args.baselines_only:
        from fingerly.dqn import DQNAgent  # torch is only needed when there are checkpoints

        agents = [DQNAgent.load(p) for p in ckpts]
    meta = agents[0].extra if agents else {}

    hand = args.hand or meta.get("hand", "R")
    min_len = args.min_len or meta.get("min_len", 8)
    val_pieces = args.val_pieces if args.val_pieces is not None else meta.get("val_pieces", 12)
    split_seed = args.split_seed if args.split_seed is not None else meta.get("split_seed", 0)
    span_scale = meta.get("span_scale", 1.0)
    lookahead = meta.get("lookahead", 3)

    split = build_split(args.pig_dir, hand, min_len, val_pieces, split_seed)
    passages = split.test if args.split == "test" else split.val
    cfg = CostConfig(span_scale=span_scale)
    rows, results, preds = [], {}, {}

    hum = human_agreement(passages)
    if hum:
        rows.append(("human agreement (LOO)", hum, None))
        results["human_agreement"] = hum

    if agents:
        env = FingeringEnv(passages, hand=hand, lookahead=lookahead, cost_cfg=cfg)
        per_seed = []
        for path, agent in zip(ckpts, agents):
            policy = agent.policy()
            m = evaluate_predictor(lambda p: run_policy(env, p, policy), passages, cfg)
            per_seed.append(m)
            results[f"dqn:{path}"] = m
            preds.setdefault("DQN (seed 1)", [run_policy(env, p, policy) for p in passages[: args.show]])
        keys = [k for k in per_seed[0] if isinstance(per_seed[0][k], float)]
        mean = {k: float(np.mean([m[k] for m in per_seed])) for k in keys}
        sd = {k: float(np.std([m[k] for m in per_seed])) for k in keys} if len(per_seed) > 1 else None
        name = f"DQN agent (n={len(per_seed)})"
        rows.append((name, mean, sd))
        results["dqn_mean"], results["dqn_std"] = mean, sd

    if not args.no_baselines:
        hmm = HMM1().fit(split.train)
        for b in [RandomValid(cfg), Greedy(cfg), OptimalCost(cfg), hmm]:
            m = evaluate_predictor(b.predict, passages, cfg)
            rows.append((b.name, m, None))
            results[b.name] = m
            preds[b.name] = [b.predict(p) for p in passages[: args.show]]

    print(f"\n{args.split} set: {len(passages)} passages, {sum(map(len, passages))} notes, hand {hand}\n")
    print_table(rows)

    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    for i, p in enumerate(passages[: args.show]):
        print(f"\n--- {p.piece}, passage {i} ({len(p)} notes) ---")
        print(f"{'notes':<20}" + " ".join(f"{names[x % 12] + str(x // 12 - 1):>4}" for x in p.pitches))
        for k, lab in enumerate(p.labels):
            print(f"{'expert ' + str(k + 1):<20}" + " ".join(f"{f if f else '-':>4}" for f in lab))
        for name, ps in preds.items():
            print(f"{name[:19]:<20}" + " ".join(f"{f:>4}" for f in ps[i]))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
