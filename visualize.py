"""Watch a fingering being played on the pygame piano (visual/piano).

The chosen policy fingers the monophonic passages of one hand in a PIG-format file; the piano
then plays them with the hand drawn on the keys, the finger number, the cost of each note and the
rules that fired. Annotator 1's finger is shown next to it when the file has one.

    python visualize.py --ckpt runs/rh_ddqn_s0/best.pt --input data/PianoFingeringDataset_v1.2/FingeringFiles/001-1_fingering.txt
    python visualize.py --policy dp --input .../001-1_fingering.txt --passage 0 --speed 0.75
    python visualize.py --ckpt runs/rh_ddqn_s0/best.pt --input ... --json out/001.json --no-show

Keys: Space pause/resume, R restart, Right/Left faster/slower (x1.25), Esc quit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from fingerly.baselines import Greedy, OptimalCost
from fingerly.cost import CostConfig
from fingerly.export import export_fingering, passages_of
from fingerly.pig import read_fingering_file

PIANO_DIR = Path(__file__).resolve().parent / "visual" / "piano"


def make_fingerer(args):
    """(fingerer, policy label, hand, min_len, cost config) for the chosen policy."""
    if args.policy == "agent":
        if not args.ckpt:
            raise SystemExit("--policy agent needs --ckpt")
        from fingerly.dqn import DQNAgent
        from fingerly.env import FingeringEnv, run_policy

        agent = DQNAgent.load(args.ckpt)
        meta = agent.extra
        hand = meta.get("hand", "R")
        if args.hand and args.hand != hand:
            raise SystemExit(f"{args.ckpt} was trained on the {hand} hand, not {args.hand}")
        cfg = CostConfig(span_scale=meta.get("span_scale", 1.0))
        lookahead, policy = meta.get("lookahead", 3), agent.policy()

        def finger(p):
            return run_policy(FingeringEnv([p], hand=hand, lookahead=lookahead, cost_cfg=cfg), p, policy)

        return finger, f"agent ({Path(args.ckpt).parent.name}/{Path(args.ckpt).name})", hand, meta.get("min_len", 8), cfg

    hand, cfg = args.hand or "R", CostConfig()
    if args.policy == "human":  # annotator 1's fingers, looked up by the passage's onsets
        runs, _ = passages_of(read_fingering_file(args.input), hand, args.min_len)
        by_onsets = {tuple(n.onset for n in r): [n.finger or 1 for n in r] for r in runs}
        return lambda p: by_onsets[tuple(p.onsets)], "human (annotator 1)", hand, args.min_len, cfg
    model = OptimalCost(cfg) if args.policy == "dp" else Greedy(cfg)
    return model.predict, {"dp": "optimal-cost (DP)", "greedy": "greedy"}[args.policy], hand, args.min_len, cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="PIG-format fingering file")
    ap.add_argument("--policy", choices=["agent", "dp", "greedy", "human"], default="agent")
    ap.add_argument("--ckpt", help="trained agent (for --policy agent)")
    ap.add_argument("--hand", choices=["R", "L"], help="default: the agent's hand, else R")
    ap.add_argument("--min-len", type=int, default=8, help="for non-agent policies (the agent uses its own)")
    ap.add_argument("--passage", type=int, help="play only this passage (0-based); default: all of them")
    ap.add_argument("--max-gap", type=float, default=1.0, help="longest pause between passages, seconds")
    ap.add_argument("--json", help="also write the fingering here")
    ap.add_argument("--no-show", action="store_true", help="only write --json, do not open the window")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--screenshot", help="render headless up to --at seconds, save a PNG and exit")
    ap.add_argument("--at", type=float, default=0.0)
    args = ap.parse_args()

    fingerer, label, hand, min_len, cfg = make_fingerer(args)
    try:
        data = export_fingering(args.input, fingerer, label, hand=hand, min_len=min_len, cfg=cfg,
                                passage=args.passage, max_gap=args.max_gap)
    except ValueError as e:
        raise SystemExit(str(e))
    n = len(data["notes"])
    print(f"{data['piece']}: {label}, hand {hand}, {n} notes in "
          f"{1 if args.passage is not None else data['n_passages']} of {data['n_passages']} passages, "
          f"total cost {data['total_cost']:.1f} ({data['total_cost'] / n:.3f} per note)")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(data, indent=1))
        print(f"wrote {args.json}")
    if args.no_show:
        return

    if args.screenshot:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    sys.path.insert(0, str(PIANO_DIR))  # the piano code imports its modules by bare name
    import pygame
    import fingering_player

    fingering_player.run(data, args.speed, args.loop, args.screenshot, args.at)
    pygame.quit()
    if args.screenshot:
        print(f"wrote {args.screenshot}")


if __name__ == "__main__":
    main()
