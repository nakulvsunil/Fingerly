"""Finger a PIG-format note file with a trained agent.

The notes of the chosen hand that sit in monophonic passages get the agent's finger
(negative for the left hand); every other note gets 0 ("not fingered by this model").

    python predict.py --ckpt runs/rh_ddqn_s0/best.pt --input data/PIG/FingeringFiles/001-1_fingering.txt \
                      --output out/001_agent_fingering.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fingerly.cost import CostConfig
from fingerly.dqn import DQNAgent
from fingerly.env import FingeringEnv, run_policy
from fingerly.pig import monophonic_runs, read_fingering_file, Passage


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True, help="PIG-format file (its finger column is ignored)")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    agent = DQNAgent.load(args.ckpt)
    meta = agent.extra
    hand = meta.get("hand", "R")
    notes = read_fingering_file(args.input)
    runs = monophonic_runs([n for n in notes if n.hand == hand], meta.get("min_len", 8))

    cfg = CostConfig(span_scale=meta.get("span_scale", 1.0))
    passages = [Passage("input", hand, [n.pitch for n in r], [n.onset for n in r], [n.offset for n in r]) for r in runs]
    if not passages:
        raise SystemExit(f"no monophonic {hand}-hand passages of >= {meta.get('min_len', 8)} notes in {args.input}")
    env = FingeringEnv(passages, hand=hand, lookahead=meta.get("lookahead", 3), cost_cfg=cfg)
    policy = agent.policy()

    assigned = {}
    for r, p in zip(runs, passages):
        for n, f in zip(r, run_policy(env, p, policy)):
            assigned[id(n)] = f if hand == "R" else -f

    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fo:
        fo.write("//Version: PianoFingering_v170101 (fingered by Fingerly)\n")
        for n in sorted(notes, key=lambda n: n.note_id):
            name = f"{names[n.pitch % 12]}{n.pitch // 12 - 1}"
            fo.write(f"{n.note_id}\t{n.onset:.6f}\t{n.offset:.6f}\t{name}\t64\t80\t{n.channel}\t{assigned.get(id(n), 0)}\n")
    print(f"fingered {len(assigned)} of {len(notes)} notes in {len(passages)} passages -> {args.output}")


if __name__ == "__main__":
    main()
