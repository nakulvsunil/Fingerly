"""Train the masked DQN / Double DQN fingering agent on one hand, monophonic PIG passages.

Examples
--------
    # main run (right hand, Double DQN, pure ergonomic reward)
    python train.py --pig-dir data/PianoFingeringDataset_v1.2 --out runs/rh_ddqn_s0 --seed 0

    # plain DQN, as in Iman & Ahn (2024)
    python train.py --pig-dir data/PIG --no-double --no-dueling --out runs/rh_dqn_s0

    # add the expert bonus (lambda) and 500 synthetic exercises
    python train.py --pig-dir data/PIG --lam 0.5 --synthetic 500 --out runs/rh_lam05_s0

    # no dataset yet: smoke test on synthetic exercises only
    python train.py --synthetic-only --steps 20000 --out runs/smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fingerly.cost import CostConfig
from fingerly.data import build_split, synthetic_split
from fingerly.dqn import DQNConfig, train_dqn
from fingerly.env import FingeringEnv, run_policy
from fingerly.metrics import evaluate_predictor
from fingerly.synth import synthetic_passages


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = ap.add_argument_group("data")
    d.add_argument("--pig-dir", help="PIG root or its FingeringFiles folder")
    d.add_argument("--hand", choices=["R", "L"], default="R")
    d.add_argument("--min-len", type=int, default=8, help="shortest monophonic passage kept")
    d.add_argument("--val-pieces", type=int, default=12)
    d.add_argument("--split-seed", type=int, default=0)
    d.add_argument("--synthetic", type=int, default=0, help="add N synthetic exercises to training")
    d.add_argument("--synthetic-only", action="store_true", help="train/validate on synthetic data only")

    e = ap.add_argument_group("environment / reward")
    e.add_argument("--lookahead", type=int, default=3)
    e.add_argument("--alpha", type=float, default=1.0, help="weight of effort cost")
    e.add_argument("--beta", type=float, default=1.0, help="weight of impossible-move penalty")
    e.add_argument("--lam", type=float, default=0.0, help="weight of expert-match bonus (0 = off)")
    e.add_argument("--no-mask", action="store_true", help="disable action masking (ablation)")
    e.add_argument("--span-scale", type=float, default=1.0, help="hand size: scales stretch limits")
    e.add_argument("--reward-scale", type=float, default=1.0, help="multiply rewards (training only)")

    a = ap.add_argument_group("agent")
    a.add_argument("--steps", type=int, default=200_000)
    a.add_argument("--lr", type=float, default=3e-4)
    a.add_argument("--batch-size", type=int, default=128)
    a.add_argument("--gamma", type=float, default=0.99)
    a.add_argument("--n-step", type=int, default=3)
    a.add_argument("--buffer", type=int, default=200_000)
    a.add_argument("--target-update", type=int, default=1_000)
    a.add_argument("--eps-decay-frac", type=float, default=0.3)
    a.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    a.add_argument("--no-double", action="store_true")
    a.add_argument("--no-dueling", action="store_true")
    a.add_argument("--eval-every", type=int, default=10_000)
    a.add_argument("--select-by", choices=["reward", "match"], default="reward",
                   help="checkpoint selection on validation: mean reward (pure RL) or Mgen")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--device", default="cpu")
    a.add_argument("--out", default="runs/default")
    return ap.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.synthetic_only:
        split = synthetic_split(max(args.synthetic, 500), args.hand, args.seed)
    else:
        split = build_split(args.pig_dir, args.hand, args.min_len, args.val_pieces, args.split_seed)
    train_passages = list(split.train)
    if args.synthetic and not args.synthetic_only:
        train_passages += synthetic_passages(args.synthetic, args.hand, seed=1000 + args.seed)
    if not train_passages or not split.val:
        raise SystemExit("No training or validation passages - check --pig-dir / --min-len / --hand")

    cost_cfg = CostConfig(span_scale=args.span_scale)
    env_kw = dict(hand=args.hand, lookahead=args.lookahead, cost_cfg=cost_cfg,
                  alpha=args.alpha, beta=args.beta, lam=args.lam, use_mask=not args.no_mask)
    env = FingeringEnv(train_passages, **env_kw, reward_scale=args.reward_scale)
    eval_env = FingeringEnv(split.val, **{**env_kw, "use_mask": True})

    cfg = DQNConfig(
        total_steps=args.steps, lr=args.lr, batch_size=args.batch_size, gamma=args.gamma,
        n_step=args.n_step, buffer_size=args.buffer, target_update=args.target_update,
        eps_decay_frac=args.eps_decay_frac, hidden=args.hidden, double=not args.no_double,
        dueling=not args.no_dueling, eval_every=args.eval_every, seed=args.seed, device=args.device,
    )

    def evaluate(agent):
        policy = agent.policy()
        m = evaluate_predictor(lambda p: run_policy(eval_env, p, policy), split.val, cost_cfg)
        res = {"val_reward": -args.alpha * m["effort_per_note"],
               "val_effort": m["effort_per_note"], "val_impossible": m["impossible_rate"]}
        if "Mgen" in m:
            res["val_match"] = m["Mgen"]
        return res

    select_key = "val_match" if args.select_by == "match" else "val_reward"
    if select_key == "val_match" and not split.val[0].labels:
        select_key = "val_reward"

    meta = {"hand": args.hand, "lookahead": args.lookahead, "span_scale": args.span_scale,
            "min_len": args.min_len, "val_pieces": args.val_pieces, "split_seed": args.split_seed,
            "pig_dir": args.pig_dir, "lam": args.lam, "alpha": args.alpha, "beta": args.beta}
    (out / "config.json").write_text(json.dumps({**vars(args), "dqn": cfg.__dict__}, indent=2))
    print(f"training on {len(train_passages)} passages ({sum(map(len, train_passages))} notes), "
          f"validating on {len(split.val)}; obs dim {env.observation_space.shape[0]}")
    train_dqn(env, cfg, out, evaluate=evaluate, select_key=select_key, extra=meta)
    print(f"done. checkpoints and logs in {out}/")


if __name__ == "__main__":
    import sys

    if any(a == "--mode" or a.startswith("--mode=") for a in sys.argv[1:]):
        from fingerly.twohand.train_cli import main as twohand_main  # Part C (README §C.13)

        twohand_main()
    else:
        main()
