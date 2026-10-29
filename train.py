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
import sys
import time
from pathlib import Path
from typing import Dict

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


# ======================================================================================
# Part C: `python train.py --mode lh|given|assign ...` (README §C.9, §C.13)
# ======================================================================================
TWOHAND_DOC = """`python train.py --mode lh|given|assign ...`: Part C training (README §C.9, §C.13).

    R1   --lam 0                       R2  --lam 0.5            R3  --lam 0.5 --elite 1
    2c   --mode assign --init runs/r3_s0/best.pt --curriculum
    A1   --bc-init runs/s1_s0/best.pt  A2  --demos 0.25         A3  --elite 1 --elite-seed-optimum
    DQN  --algo dqn                    Q-learning  --algo qlearn --split test --episodes 500
    speed test  --bench 1000 --bench-piece 033

Training never sees the 30 test pieces (§C.4.6), except tabular Q-learning, which is defined per
test piece with the rule reward only (§C.9.5). The imitation bonus is given on training segments
only; validation runs without it. Every seed and setting goes to <out>/config.json.
"""


def twohand_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=TWOHAND_DOC, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = ap.add_argument_group("data")
    d.add_argument("--pig-dir", required=True)
    d.add_argument("--mode", choices=["lh", "given", "assign"], required=True)
    d.add_argument("--chords", choices=["full", "reduce"], default="full")
    d.add_argument("--chord-tol", type=float, default=0.03)
    d.add_argument("--overlap-tol", type=float, default=0.05)
    d.add_argument("--synthetic", type=int, default=200, help="synthetic two-hand segments added to training")
    d.add_argument("--vmax-percentile", type=float, default=95.0)
    e = ap.add_argument_group("reward")
    e.add_argument("--cost", choices=["parncutt12", "balliauw15"], default="parncutt12")
    e.add_argument("--chord-horizontal", choices=["all", "extremes"], default="all")
    e.add_argument("--lam", type=float, default=0.0)
    e.add_argument("--cross", action="store_true", help="crossing term in 2b")
    e.add_argument("--w-time", type=float, default=1.0)
    e.add_argument("--w-cross", type=float, default=1.0)
    e.add_argument("--no-mask", action="store_true")
    e.add_argument("--span-scale", type=float, default=1.0)
    s = ap.add_argument_group("state")
    s.add_argument("--lookahead", type=int, default=4)
    s.add_argument("--no-hand-pos", action="store_true", help="drop thumb-position features (ablation)")
    a = ap.add_argument_group("agent")
    a.add_argument("--algo", choices=["ddqn", "dqn", "qlearn"], default="ddqn")
    a.add_argument("--steps", type=int, default=None, help="default 1M (2a/2b), 1.5M (2c)")
    a.add_argument("--envs", type=int, default=16)
    a.add_argument("--lr", type=float, default=3e-4)
    a.add_argument("--gamma", type=float, default=0.99)
    a.add_argument("--n-step", type=int, default=3)
    a.add_argument("--batch", type=int, default=256)
    a.add_argument("--buffer", type=int, default=500_000)
    a.add_argument("--train-every", type=int, default=4)
    a.add_argument("--learning-starts", type=int, default=10_000)
    a.add_argument("--tau", type=float, default=0.005)
    a.add_argument("--eps-frac", type=float, default=0.2)
    a.add_argument("--eps-end", type=float, default=0.05)
    a.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    a.add_argument("--no-double", action="store_true")
    a.add_argument("--no-dueling", action="store_true")
    a.add_argument("--elite", type=float, default=0.0, help="1 = on (25%% of each batch); or the share itself")
    a.add_argument("--elite-share", type=float, default=0.25)
    a.add_argument("--elite-seed-optimum", action="store_true", help="A3: M0's optimum per training segment in the elite memory")
    a.add_argument("--bc-init", default=None, help="A1: start from S1's weights")
    a.add_argument("--demos", type=float, default=0.0, help="A2: share of demonstrations, decayed to 0.10")
    a.add_argument("--init", default=None, help="2c: trunk from a 2b checkpoint")
    a.add_argument("--curriculum", action="store_true", help="2c: short synthetic -> all synthetic -> PIG")
    a.add_argument("--select", choices=["auto", "match", "difficulty"], default="auto",
                   help="auto: Mgen when lam > 0 (R2-R4), difficulty when lam = 0 (R1)")
    a.add_argument("--eval-every", type=int, default=25_000)
    a.add_argument("--patience", type=int, default=50_000)
    a.add_argument("--checkpoint-every", type=int, default=50_000)
    a.add_argument("--no-save-buffer", action="store_true")
    a.add_argument("--val-beam", type=int, default=1000, help="beam for the validation M0 in 2c")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--device", default="cpu")
    a.add_argument("--threads", type=int, default=1)
    q = ap.add_argument_group("tabular Q-learning")
    q.add_argument("--split", choices=["test", "val"], default="test")
    q.add_argument("--episodes", type=int, default=500)
    b = ap.add_argument_group("speed test")
    b.add_argument("--bench", type=int, default=0, help="time N environment steps and exit")
    b.add_argument("--bench-piece", default="033")
    ap.add_argument("--out", required=True)
    return ap


def make_evaluator(val_pool, m0: Dict[tuple, float]):
    from fingerly.metrics import rollout, summarize

    def evaluate(agent) -> Dict[str, float]:
        eps = rollout(val_pool, agent.greedy())
        return summarize(eps, m0)

    return evaluate


def twohand_main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    args = twohand_parser().parse_args(argv)
    import torch

    torch.set_num_threads(args.threads)
    from fingerly.dqn import TrainConfig, Trainer, bench, episode_transitions
    from fingerly.synth import twohand_segments
    from fingerly.twohand.env import SegmentPool, env_config
    from fingerly.twohand.prepare import Context, PrepConfig
    from fingerly.twohand.search import m0_costs

    t0 = time.time()
    prep = PrepConfig(chord_tol=args.chord_tol, overlap_tol=args.overlap_tol, chords=args.chords)
    ctx = Context(args.pig_dir, prep, args.vmax_percentile, verbose=True)
    ecfg = env_config(args, ctx.v_max)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    steps = args.steps or (1_500_000 if args.mode == "assign" else 1_000_000)

    if args.algo == "qlearn":
        from fingerly.qlearn import run_qlearning

        segs = ctx.segments(args.split)
        run_qlearning(segs, {**ecfg.__dict__, "lam": 0.0}, out, args.episodes, args.seed)
        return

    tcfg = TrainConfig(algo=args.algo, steps=steps, n_envs=args.envs, lr=args.lr, gamma=args.gamma,
                       n_step=args.n_step, buffer=args.buffer, batch=args.batch, train_every=args.train_every,
                       learning_starts=args.learning_starts, tau=args.tau, eps_end=args.eps_end,
                       eps_frac=args.eps_frac, hidden=tuple(args.hidden), double=not args.no_double,
                       dueling=not args.no_dueling, elite=(args.elite_share if args.elite == 1 else args.elite),
                       demos=args.demos, eval_every=args.eval_every, patience=args.patience,
                       checkpoint_every=args.checkpoint_every, save_buffer=not args.no_save_buffer,
                       seed=args.seed, device=args.device)

    if args.bench:
        segs = [s for s in ctx.split.test if s.piece == args.bench_piece] or ctx.split.val
        pool = SegmentPool(segs, ecfg, bonus=False)
        r = bench(pool, tcfg, args.bench)
        print(json.dumps({k: round(v, 1) for k, v in r.items()}))
        (out / "bench.json").write_text(json.dumps(dict(r, device=args.device, mode=args.mode, steps=args.bench,
                                                        piece=args.bench_piece, threads=args.threads), indent=1))
        return

    train_segs = list(ctx.split.train)
    if args.synthetic:
        train_segs += twohand_segments(args.synthetic, seed=1000 + args.seed, prep=prep)
    pool = SegmentPool(train_segs, ecfg, bonus=True)
    val_pool = SegmentPool(ctx.split.val, ecfg, bonus=False)
    print(f"training pool: {len(pool)} segments; validation: {len(val_pool)} segments; "
          f"obs {ecfg.obs_dim}, actions {ecfg.n_actions}; built in {time.time() - t0:.0f}s", flush=True)
    m0 = m0_costs(val_pool, args.val_beam)

    schedule = None
    if args.curriculum:
        short = SegmentPool(twohand_segments(400, seed=2000 + args.seed, max_len=32, prep=prep), ecfg, bonus=False)
        allsyn = SegmentPool(twohand_segments(400, seed=3000 + args.seed, prep=prep), ecfg, bonus=False)
        schedule = [(0, short), (int(0.1 * steps), allsyn), (int(0.25 * steps), pool)]

    elite_seed = None
    if args.elite_seed_optimum:  # A3: uses search in training, reported separately
        from fingerly.twohand.search import search_segment

        if tcfg.elite <= 0:
            tcfg.elite = args.elite_share
        elite_seed = {}
        for sid, info in enumerate(pool.infos):
            r = search_segment(info, ecfg, args.val_beam)
            acts = iter([s + (31 if (ecfg.mode == "assign" and h == "R") else 0) for h, s in r.fingering])
            tr, ret = episode_transitions(pool, info, lambda ep: next(acts, None), tcfg.n_step, tcfg.gamma)
            elite_seed[sid] = (tr, ret)
        print(f"A3: {len(elite_seed)} optimal episodes seeded into the elite memory", flush=True)

    select = args.select if args.select != "auto" else ("match" if args.lam > 0 else "difficulty")
    key, sign = ("Mgen", 1.0) if select == "match" else ("difficulty_per_note", -1.0)
    meta = dict(env=ecfg.__dict__, prep=prep.__dict__, mode=args.mode, lam=args.lam, select=select,
                pig_dir=args.pig_dir, seed=args.seed, args=vars(args), v_max=ctx.v_max,
                train_pieces=ctx.split.train_pieces, val_pieces=ctx.split.val_pieces)
    trainer = Trainer(pool, tcfg, out, make_evaluator(val_pool, m0), key, sign, meta,
                      bc_init=args.bc_init, init=args.init, elite_seed=elite_seed, pool_schedule=schedule)
    trainer.train()
    print(f"done in {(time.time() - t0) / 60:.1f} min; checkpoints in {out}", flush=True)


if __name__ == "__main__":
    import sys

    if any(a == "--mode" or a.startswith("--mode=") for a in sys.argv[1:]):
        twohand_main()
    else:
        main()
