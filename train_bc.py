"""S1, the copy-only baseline (README §C.9.1): the agent's network and masks, trained with
cross-entropy to predict the pianists' finger sets on the training pieces. Not RL.

    python train_bc.py --pig-dir data/PianoFingeringDataset_v1.2 --mode given --seed 0 --out runs/s1_s0

States come from playing each annotator's fingering of each training segment through the
environment. Where an annotator's choice is unavailable (another hand, a non pitch-order chord, or
masked) the environment takes the first allowed action and that event is not used as a sample.
The checkpoint with the best validation Mgen is kept as best.pt, in the same format as the RL
agent (it doubles as the A1 initialisation). The test pieces are never used.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from fingerly.twohand.env import Episode, SegmentPool
from fingerly.twohand.prepare import Context
from fingerly.dqn import NEG, Agent, TrainConfig, annotator_action, set_seeds
from fingerly.metrics import rollout, summarize
from fingerly.twohand.prepare import PrepConfig
from fingerly.twohand.env import env_config
from fingerly.twohand.search import m0_costs


def collect(pool: SegmentPool):
    """(obs, mask, action) for every event where an annotator's choice is available."""
    assign = pool.cfg.mode == "assign"
    O, M, A = [], [], []
    used = skipped = 0
    for info in pool.infos:
        if not info.bonus:
            continue
        for k in range(info.seg.n_annotators):
            ep = Episode(pool.cfg, pool.model)
            o, m = ep.reset(info)
            while not ep.done:
                a = annotator_action(info, ep.event_index, assign, k)
                if a is not None and m[a]:
                    O.append(o)
                    M.append(m)
                    A.append(a)
                    used += 1
                else:
                    a = int(np.flatnonzero(m)[0])
                    skipped += 1
                ep.step(a)
                if not ep.done:
                    o, m = ep.obs(), ep.mask()
    return np.array(O, np.float32), np.array(M, bool), np.array(A, np.int64), used, skipped


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pig-dir", required=True)
    ap.add_argument("--mode", choices=["lh", "given", "assign"], default="given")
    ap.add_argument("--cost", choices=["parncutt12", "balliauw15"], default="parncutt12")
    ap.add_argument("--chord-horizontal", choices=["all", "extremes"], default="all")
    ap.add_argument("--chords", choices=["full", "reduce"], default="full")
    ap.add_argument("--lookahead", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", required=True)
    # flags shared with train.py's env_config
    ap.add_argument("--span-scale", type=float, default=1.0)
    ap.add_argument("--w-time", type=float, default=1.0)
    ap.add_argument("--w-cross", type=float, default=1.0)
    ap.add_argument("--cross", action="store_true")
    ap.add_argument("--no-mask", action="store_true")
    ap.add_argument("--no-hand-pos", action="store_true")
    ap.add_argument("--chord-tol", type=float, default=0.03)
    ap.add_argument("--overlap-tol", type=float, default=0.05)
    args = ap.parse_args()
    args.lam = 0.0
    torch.set_num_threads(args.threads)
    set_seeds(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "done.json").exists():
        print(f"{out} already finished")
        return

    t0 = time.time()
    prep = PrepConfig(chord_tol=args.chord_tol, overlap_tol=args.overlap_tol, chords=args.chords)
    ctx = Context(args.pig_dir, prep, verbose=True)
    ecfg = env_config(args, ctx.v_max)
    pool = SegmentPool(ctx.split.train, ecfg, bonus=True)
    val_pool = SegmentPool(ctx.split.val, ecfg, bonus=False)
    O, M, A, used, skipped = collect(pool)
    print(f"S1 data: {used} samples, {skipped} events skipped (annotator choice unavailable)", flush=True)
    m0 = m0_costs(val_pool, 1000)

    tcfg = TrainConfig(hidden=tuple(args.hidden), dueling=True, seed=args.seed, device=args.device, lr=args.lr,
                       cuda_graph=False)
    agent = Agent(ecfg.obs_dim, ecfg.n_actions, tcfg)
    opt = torch.optim.Adam(agent.q.parameters(), lr=args.lr)
    dev = agent.dev
    Ot, Mt, At = torch.as_tensor(O, device=dev), torch.as_tensor(M, device=dev), torch.as_tensor(A, device=dev)
    rng = np.random.default_rng(args.seed)
    meta = dict(env=ecfg.__dict__, prep=prep.__dict__, mode=args.mode, method="S1", seed=args.seed, args=vars(args),
                v_max=ctx.v_max, samples=used, skipped=skipped)
    (out / "config.json").write_text(json.dumps(meta, indent=1, default=str))
    log = open(out / "eval_log.csv", "w", newline="")
    w = None
    best = -np.inf
    for epoch in range(1, args.epochs + 1):
        perm = rng.permutation(len(A))
        tot = 0.0
        agent.q.train()
        for i in range(0, len(perm), args.batch):
            j = torch.as_tensor(perm[i:i + args.batch], device=dev)
            logits = agent.q(Ot[j]).masked_fill(~Mt[j], NEG)
            loss = F.cross_entropy(logits, At[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item()) * len(j)
        agent.q.eval()
        m = {"epoch": epoch, "train_ce": tot / len(A), **summarize(rollout(val_pool, agent.greedy()), m0)}
        if w is None:
            w = csv.DictWriter(log, fieldnames=list(m.keys()))
            w.writeheader()
        w.writerow(m)
        log.flush()
        tag = ""
        if m.get("Mgen", -1) > best:
            best = m["Mgen"]
            agent.save(out / "best.pt", {**meta, "epoch": epoch, "eval": m})
            tag = "  <- best"
        print(f"epoch {epoch:>2} ce {m['train_ce']:.4f} val Mgen {m.get('Mgen', 0):.4f} difficulty/note "
              f"{m['difficulty_per_note']:.3f} infeasible {m['infeasible_rate']:.4f}{tag}", flush=True)
    agent.save(out / "last.pt", meta)
    (out / "done.json").write_text(json.dumps(dict(best_Mgen=best, seconds=time.time() - t0), indent=1))
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
