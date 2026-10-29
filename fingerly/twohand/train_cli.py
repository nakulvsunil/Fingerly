"""`python train.py --mode lh|given|assign ...`: Part C training (README §C.9, §C.13).

    R1   --lam 0                       R2  --lam 0.5            R3  --lam 0.5 --elite 1
    2c   --mode assign --init runs/r3_s0/best.pt --curriculum
    A1   --bc-init runs/s1_s0/best.pt  A2  --demos 0.25         A3  --elite 1 --elite-seed-optimum
    DQN  --algo dqn                    Q-learning  --algo qlearn --split test --episodes 500
    speed test  --bench 1000 --bench-piece 033

Training never sees the 30 test pieces (§C.4.6), except tabular Q-learning, which is defined per
test piece with the rule reward only (§C.9.5). The imitation bonus is given on training segments
only; validation runs without it. Every seed and setting goes to <out>/config.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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


def env_config(args, v_max: float):
    from .env import EnvConfig

    return EnvConfig(mode=args.mode, cost=args.cost, chord_horizontal=args.chord_horizontal,
                     span_scale=args.span_scale, v_max=v_max, w_time=args.w_time, w_cross=args.w_cross,
                     cross=args.cross, lam=args.lam, lookahead=args.lookahead, hand_pos=not args.no_hand_pos,
                     use_mask=not args.no_mask, chord_tol=args.chord_tol, overlap_tol=args.overlap_tol)


def m0_costs(pool, beam: int) -> Dict[tuple, float]:
    """M0 cost per (piece, segment index) for the success rate and gap on validation/test."""
    from .search import search_segment

    return {(i.seg.piece, i.seg.index): search_segment(i, pool.cfg, beam).cost for i in pool.infos}


def make_evaluator(val_pool, m0: Dict[tuple, float]):
    from .metrics import rollout, summarize

    def evaluate(agent) -> Dict[str, float]:
        eps = rollout(val_pool, agent.greedy())
        return summarize(eps, m0)

    return evaluate


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    import torch

    torch.set_num_threads(args.threads)
    from ..synth import twohand_segments
    from .env import SegmentPool
    from .human import Context
    from .learner import TrainConfig, Trainer, bench, episode_transitions
    from .prepare import PrepConfig

    t0 = time.time()
    prep = PrepConfig(chord_tol=args.chord_tol, overlap_tol=args.overlap_tol, chords=args.chords)
    ctx = Context(args.pig_dir, prep, args.vmax_percentile, verbose=True)
    ecfg = env_config(args, ctx.v_max)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    steps = args.steps or (1_500_000 if args.mode == "assign" else 1_000_000)

    if args.algo == "qlearn":
        from ..qlearn import run_qlearning

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
        from .search import search_segment

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
    main()
