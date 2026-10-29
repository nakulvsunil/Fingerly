"""R4 (README §C.9.2): the best R2/R3 policy fine-tuned on each test piece, rule reward only.

    python finetune.py --ckpt runs/r3_lam0.5_s0/best.pt --pig-dir data/PianoFingeringDataset_v1.2 --split test --episodes 50 --out runs/r4

For every piece a fresh copy of the checkpoint is trained for `episodes` passes over the piece's
segments: lr 1e-4, epsilon 0.1, no imitation bonus, no labels (the pianists' fingerings are only
used afterwards, to report d_rel). After each episode the greedy policy is scored; the curve and
the final greedy fingering go to <out>/<piece>.json. Finished pieces are skipped on restart.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from fingerly.twohand.env import EnvConfig, Episode, SegmentPool
from fingerly.twohand.human import Context
from fingerly.twohand.learner import Agent, NStep, Replay, set_seeds
from fingerly.twohand.metrics import rollout, summarize
from fingerly.twohand.prepare import PrepConfig
from fingerly.twohand.rules import make_model
from fingerly.twohand.score import rule_points


def finetune_piece(ckpt, pool: SegmentPool, episodes: int, lr: float, eps: float, seed: int, device: str,
                   batch: int = 256, train_every: int = 4, buffer: int = 50_000):
    set_seeds(seed)
    rng = np.random.default_rng(seed)
    agent = Agent.load(ckpt, device)
    agent.cfg.lr = lr
    for g in agent.opt.param_groups:
        g["lr"] = lr
    cfg = agent.cfg
    buf = Replay(buffer, pool.cfg.obs_dim, pool.cfg.n_actions)
    bal = make_model("balliauw15", pool.cfg.chord_horizontal, pool.cfg.span_scale)
    curve = []

    def score(tag):
        eps_ = rollout(pool, agent.greedy())
        s = summarize(eps_)
        s["balliauw15"] = sum(rule_points(e.info.seg, e.info.order, [(r.hand, r.set) for r in e.records], bal)
                              for e in eps_)
        s["episode"] = tag
        return s, eps_

    s, _ = score(0)
    curve.append(s)
    acc = 0.0
    for k in range(1, episodes + 1):
        for info in pool.infos:
            ep = Episode(pool.cfg, pool.model)
            o, m = ep.reset(info)
            ns = NStep(cfg.n_step, cfg.gamma)
            while not ep.done:
                a = int(agent.act(o[None], m[None], eps, rng)[0])
                r, done = ep.step(a)
                o2, m2 = (ep.obs(), ep.mask()) if not done else (np.zeros_like(o), np.ones_like(m))
                for tr in ns.push(o, a, r, o2, m2, done):
                    buf.add(*tr[:6])
                o, m = o2, m2
                if len(buf) >= batch:
                    acc += 1.0 / train_every
                    while acc >= 1:
                        agent.update(*buf.take(buf.idx(rng, batch)))
                        acc -= 1
        s, eps_ = score(k)
        curve.append(s)
    fing = [dict(piece=e.info.seg.piece, index=e.info.seg.index, fingering=[[r.hand, r.set] for r in e.records])
            for e in eps_]
    return curve, fing


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pig-dir", required=True)
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--pieces", nargs="*", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    meta = torch.load(args.ckpt, map_location="cpu", weights_only=False).get("extra", {})
    env = dict(meta.get("env", {}))
    prep = PrepConfig(**meta["prep"]) if "prep" in meta else PrepConfig()
    ctx = Context(args.pig_dir, prep)
    env.update(lam=0.0, v_max=ctx.v_max)
    ecfg = EnvConfig(**env)
    by = defaultdict(list)
    for s in ctx.segments(args.split):
        by[s.piece].append(s)
    (out / "config.json").write_text(json.dumps(dict(vars(args), env=env), indent=1, default=str))
    t0 = time.time()
    for piece in sorted(by):
        if args.pieces and piece not in args.pieces:
            continue
        f = out / f"{piece}.json"
        if f.exists():
            continue
        t = time.time()
        pool = SegmentPool(by[piece], ecfg, bonus=False)
        curve, fing = finetune_piece(args.ckpt, pool, args.episodes, args.lr, args.eps, args.seed, args.device)
        f.write_text(json.dumps(dict(piece=piece, curve=curve, segments=fing, seconds=time.time() - t,
                                     ckpt=args.ckpt, seed=args.seed), default=float))
        print(f"R4 {piece}: cost {curve[0]['cost']:.0f} -> {curve[-1]['cost']:.0f} "
              f"({len(curve) - 1} episodes, {time.time() - t:.0f}s)", flush=True)
    (out / "done.json").write_text(json.dumps(dict(seconds=time.time() - t0)))


if __name__ == "__main__":
    main()
