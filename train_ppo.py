"""MaskablePPO benchmark (README §C.9.5, §C.11.5): sb3-contrib, same features, masks and steps.

    python train_ppo.py --pig-dir data/PianoFingeringDataset_v1.2 --mode given --seed 0 --out runs/ppo_s0

16 environments, rollout 2,048 steps per environment, 10 epochs, batch 256, lr 3e-4, clip 0.2,
entropy 0.01, gamma 0.99, GAE lambda 0.95, 1M steps, MLP 256-256. Validation every 25k steps
(greedy, no bonus); the best model by Mgen (lam > 0) or difficulty (lam = 0) is best.zip.
ckpt.zip is written every 50k steps and the run resumes from it.
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

from fingerly.synth import twohand_segments
from fingerly.twohand.env import SegmentPool, TwoHandEnv
from fingerly.twohand.human import Context
from fingerly.twohand.metrics import rollout, summarize
from fingerly.twohand.prepare import PrepConfig
from fingerly.twohand.train_cli import env_config, m0_costs


def ppo_policy(model):
    def pol(obs, masks):
        a, _ = model.predict(obs, action_masks=masks, deterministic=True)
        return np.asarray(a)
    return pol


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pig-dir", required=True)
    ap.add_argument("--mode", choices=["lh", "given", "assign"], default="given")
    ap.add_argument("--cost", choices=["parncutt12", "balliauw15"], default="parncutt12")
    ap.add_argument("--chord-horizontal", choices=["all", "extremes"], default="all")
    ap.add_argument("--chords", choices=["full", "reduce"], default="full")
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--n-steps", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae", type=float, default=0.95)
    ap.add_argument("--synthetic", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=25_000)
    ap.add_argument("--checkpoint-every", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", required=True)
    for flag, kw in (("--span-scale", dict(type=float, default=1.0)), ("--w-time", dict(type=float, default=1.0)),
                     ("--w-cross", dict(type=float, default=1.0)), ("--lookahead", dict(type=int, default=4)),
                     ("--chord-tol", dict(type=float, default=0.03)), ("--overlap-tol", dict(type=float, default=0.05))):
        ap.add_argument(flag, **kw)
    for flag in ("--cross", "--no-mask", "--no-hand-pos"):
        ap.add_argument(flag, action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "done.json").exists():
        print(f"{out} already finished")
        return
    t0 = time.time()
    prep = PrepConfig(chord_tol=args.chord_tol, overlap_tol=args.overlap_tol, chords=args.chords)
    ctx = Context(args.pig_dir, prep, verbose=True)
    ecfg = env_config(args, ctx.v_max)
    segs = list(ctx.split.train) + (twohand_segments(args.synthetic, 1000 + args.seed, prep=prep) if args.synthetic else [])
    pool = SegmentPool(segs, ecfg, bonus=True)
    val_pool = SegmentPool(ctx.split.val, ecfg, bonus=False)
    m0 = m0_costs(val_pool, 1000)
    venv = DummyVecEnv([lambda k=k: TwoHandEnv(pool, seed=args.seed * 100 + k) for k in range(args.envs)])
    select_match = args.lam > 0
    meta = dict(env=ecfg.__dict__, prep=prep.__dict__, mode=args.mode, method="MaskablePPO", seed=args.seed,
                args=vars(args), v_max=ctx.v_max, select="match" if select_match else "difficulty")
    (out / "config.json").write_text(json.dumps(meta, indent=1, default=str))

    ck = out / "ckpt.zip"
    if ck.exists():
        model = MaskablePPO.load(ck, env=venv, device=args.device)
        print(f"resumed at {model.num_timesteps} steps", flush=True)
    else:
        model = MaskablePPO("MlpPolicy", venv, learning_rate=args.lr, n_steps=args.n_steps, batch_size=args.batch,
                            n_epochs=args.epochs, gamma=args.gamma, gae_lambda=args.gae, clip_range=args.clip,
                            ent_coef=args.ent, policy_kwargs=dict(net_arch=[256, 256]), seed=args.seed,
                            device=args.device, verbose=0)
    state_f = out / "state.json"
    state = json.loads(state_f.read_text()) if state_f.exists() else dict(best=-np.inf, next_eval=args.eval_every,
                                                                          next_ck=args.checkpoint_every)
    log = open(out / "eval_log.csv", "a", newline="")

    class Cb(BaseCallback):
        writer = None

        def _on_step(self) -> bool:
            n = self.model.num_timesteps
            if n >= state["next_eval"]:
                state["next_eval"] += args.eval_every
                m = {"step": n, **summarize(rollout(val_pool, ppo_policy(self.model)), m0)}
                if Cb.writer is None:
                    Cb.writer = csv.DictWriter(log, fieldnames=list(m.keys()), extrasaction="ignore")
                    if log.tell() == 0:
                        Cb.writer.writeheader()
                Cb.writer.writerow(m)
                log.flush()
                score = m.get("Mgen", -np.inf) if select_match else -m["difficulty_per_note"]
                tag = ""
                if score > state["best"]:
                    state["best"] = score
                    state["best_step"] = n
                    self.model.save(out / "best.zip")
                    tag = "  <- best"
                print(f"step {n:>8} return/event {m['return_per_event']:.3f} difficulty/note {m['difficulty_per_note']:.3f} "
                      f"Mgen {m.get('Mgen', 0):.4f} gap {m.get('gap', float('nan')):.3f}{tag}", flush=True)
            if n >= state["next_ck"]:
                state["next_ck"] += args.checkpoint_every
                self.model.save(ck)
                state_f.write_text(json.dumps(state))
            return True

    remaining = max(0, args.steps - model.num_timesteps)
    model.learn(total_timesteps=remaining, callback=Cb(), reset_num_timesteps=False)
    model.save(out / "last.zip")
    model.save(ck)
    state_f.write_text(json.dumps(state))
    (out / "done.json").write_text(json.dumps(dict(steps=model.num_timesteps, seconds=time.time() - t0,
                                                   best=state["best"]), indent=1, default=float))
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
