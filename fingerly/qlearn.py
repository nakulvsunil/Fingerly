"""Tabular Q-learning benchmark (README §C.9.5, §C.11.5).

Per piece, 500 episodes; an episode is one pass through all of the piece's segments, with the rule
reward only (no imitation bonus). Discrete state, per hand: the last two keys relative to the
current event (clipped to +-12 semitones), the last two finger sets; plus the current event's
size, colour and given hand. A dictionary holds the Q-table; masks are applied. alpha 0.1,
gamma 0.99, epsilon linear from 1.0 to 0.05 over the episodes. The final greedy fingering of each
segment is saved for evaluation.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .twohand.env import EnvConfig, Episode, SegmentPool


def _clip(x: int) -> int:
    return int(max(-12, min(12, x)))


def state_key(ep: Episode) -> tuple:
    e = ep.event
    key = [e.size, int(e.pitches[0] % 12 in (1, 3, 6, 8, 10)), e.hand if ep.cfg.mode != "assign" else ""]
    for h in ("L", "R"):
        hs = ep.hands[h]
        cur = e.pitches[0] if h == "R" else e.pitches[-1]
        k1 = _clip(cur - hs.keys[-1]) if hs.keys else 99
        k2 = _clip(cur - hs.keys[-2]) if len(hs.keys) == 2 else 99
        s1 = hs.sets[-1] if hs.sets else -1
        s2 = hs.sets[-2] if len(hs.sets) == 2 else -1
        key += [k1, k2, s1, s2]
    return tuple(key)


def train_piece(pool: SegmentPool, episodes: int = 500, seed: int = 0, alpha: float = 0.1, gamma: float = 0.99,
                eps_start: float = 1.0, eps_end: float = 0.05):
    rng = np.random.default_rng(seed)
    Q: Dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(pool.cfg.n_actions))
    curve = []
    ep = Episode(pool.cfg, pool.model)

    def run(eps: float, learn: bool):
        total, fings = 0.0, []
        for info in pool.infos:
            ep.reset(info)
            s, m = state_key(ep), ep.mask()
            while not ep.done:
                q = Q[s]
                valid = np.flatnonzero(m)
                a = int(rng.choice(valid)) if rng.random() < eps else int(valid[np.argmax(q[valid])])
                r, done = ep.step(a)
                total += r
                if done:
                    target = r
                    s2 = m2 = None
                else:
                    s2, m2 = state_key(ep), ep.mask()
                    target = r + gamma * Q[s2][np.flatnonzero(m2)].max()
                if learn:
                    q[a] += alpha * (target - q[a])
                s, m = s2, m2
            fings.append(dict(piece=info.seg.piece, index=info.seg.index,
                              fingering=[[x.hand, x.set] for x in ep.records]))
        return total, fings

    for k in range(episodes):
        eps = eps_start + (eps_end - eps_start) * min(1.0, k / max(1, episodes - 1))
        ret, _ = run(eps, True)
        curve.append(ret)
    greedy_ret, fings = run(0.0, False)
    return dict(curve=curve, greedy_return=greedy_ret, states=len(Q), segments=fings)


def run_qlearning(segments: Sequence, env_cfg: dict, out: str | Path, episodes: int = 500, seed: int = 0):
    """One Q-table per piece; results in <out>/<piece>.json (resumable: done pieces are skipped)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = EnvConfig(**env_cfg)
    by: Dict[str, List] = defaultdict(list)
    for s in segments:
        by[s.piece].append(s)
    (out / "config.json").write_text(json.dumps(dict(env=env_cfg, episodes=episodes, seed=seed), indent=1, default=str))
    for piece, segs in sorted(by.items()):
        f = out / f"{piece}.json"
        if f.exists():
            continue
        t0 = time.time()
        pool = SegmentPool(segs, cfg, bonus=False)
        res = train_piece(pool, episodes, seed)
        res["seconds"] = time.time() - t0
        f.write_text(json.dumps(res))
        print(f"q-learning {piece}: greedy return {res['greedy_return']:.1f}, {res['states']} states, "
              f"{res['seconds']:.0f}s", flush=True)
