"""Two-hand evaluation core (README §C.11.1-C.11.3): greedy rollouts and per-piece metrics.

A note matches an annotator only if hand **and** finger agree. Chord notes are matched one by one.
Rates are computed per piece and averaged over pieces. Notes are coded +finger (right hand) and
-finger (left hand); a note an annotator did not finger is 0 and never matches. Evaluation always
uses the frozen greedy policy and no imitation bonus.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ..metrics import match_rates
from .actions import fingers_of
from .env import Episode, SegInfo, SegmentPool, StepRecord

BatchPolicy = Callable[[np.ndarray, np.ndarray], np.ndarray]  # (obs[B, D], mask[B, A]) -> actions[B]


def rollout(pool: SegmentPool, policy: BatchPolicy, infos: Optional[Sequence[SegInfo]] = None) -> List[Episode]:
    """Run the greedy policy on every segment of the pool at once (one batched forward per step)."""
    infos = list(infos if infos is not None else pool.infos)
    eps = [Episode(pool.cfg, pool.model) for _ in infos]
    obs = np.zeros((len(infos), pool.cfg.obs_dim), np.float32)
    masks = np.zeros((len(infos), pool.cfg.n_actions), bool)
    for k, (ep, info) in enumerate(zip(eps, infos)):
        obs[k], masks[k] = ep.reset(info)
    live = [k for k, ep in enumerate(eps) if not ep.done]
    while live:
        acts = policy(obs[live], masks[live])
        nxt = []
        for k, a in zip(live, acts):
            ep = eps[k]
            ep.step(int(a))
            if not ep.done:
                obs[k], masks[k] = ep.obs(), ep.mask()
                nxt.append(k)
        live = nxt
    return eps


def note_codes(ep: Episode):
    """Per visited note: predicted code, per-annotator codes (K x N), chord flag, hand."""
    seg = ep.info.seg
    pred, labs, chord, hands = [], [], [], []
    K = seg.n_annotators
    for r in ep.records:
        e = seg.events[r.event]
        fs = fingers_of(r.set, r.hand)
        for j, f in enumerate(fs):
            pred.append(f if r.hand == "R" else -f)
            row = []
            for a in range(K):
                h, g = e.labels[a][j]
                row.append(0 if not g or h is None else (g if h == "R" else -g))
            labs.append(row)
            chord.append(e.size > 1)
            hands.append(r.hand)
    lab = np.array(labs, int).T if labs else np.zeros((K, 0), int)
    return np.array(pred, int), lab, np.array(chord, bool), np.array(hands)


def piece_match(eps: Sequence[Episode]) -> Dict[str, Dict[str, float]]:
    """{piece: {Mgen, Mhigh, Msoft, Mrec, notes}} over the episodes of each piece."""
    by = defaultdict(list)
    for ep in eps:
        if ep.info.seg.n_annotators:
            by[ep.info.seg.piece].append(note_codes(ep))
    out = {}
    for piece, parts in by.items():
        pred = np.concatenate([p[0] for p in parts])
        lab = np.concatenate([p[1] for p in parts], axis=1)
        if not len(pred):
            continue
        m = match_rates(pred, lab)
        m["notes"] = float(len(pred))
        out[piece] = m
    return out


def summarize(eps: Sequence[Episode], m0: Optional[Dict[tuple, float]] = None, success_gap: float = 0.10) -> Dict[str, float]:
    """Validation/test summary: per-event return (rule reward only), difficulty, match rates,
    infeasible rate, and, with M0 costs {(piece, index): cost}, the gap and success rates."""
    events = sum(len(ep.records) for ep in eps)
    notes = sum(ep.info.seg.events[r.event].size for ep in eps for r in ep.records)
    rules = sum(r.rules for ep in eps for r in ep.records)
    extra = sum(r.extra for ep in eps for r in ep.records)
    infeas = sum(r.infeasible or r.forced for ep in eps for r in ep.records)
    ret = -(rules + extra)
    out = dict(events=float(events), notes=float(notes), return_per_event=ret / max(events, 1),
               difficulty_per_note=rules / max(notes, 1), cost=rules + extra,
               infeasible_rate=infeas / max(events, 1))
    pm = piece_match(eps)
    if pm:
        for k in ("Mgen", "Mhigh", "Msoft", "Mrec"):
            out[k] = float(np.mean([v[k] for v in pm.values()]))
    if m0:
        gaps, succ, play = [], [], []
        agent_tot = opt_tot = 0.0
        for ep in eps:
            key = (ep.info.seg.piece, ep.info.seg.index)
            if key not in m0:
                continue
            c = sum(r.rules + r.extra for r in ep.records)
            opt = m0[key]
            agent_tot += c
            opt_tot += opt
            gap = (c - opt) / opt if opt > 0 else (0.0 if c <= 1e-9 else np.inf)
            ok_play = not any(r.infeasible or r.forced for r in ep.records)
            gaps.append(gap)
            play.append(ok_play)
            succ.append(ok_play and gap <= success_gap)
        if gaps:
            out["gap"] = (agent_tot - opt_tot) / max(opt_tot, 1e-9)
            out["SR"] = float(np.mean(succ))
            out["SR_play"] = float(np.mean(play))
    return out


def replay(pool: SegmentPool, fingerings, infos: Optional[Sequence[SegInfo]] = None) -> List[Episode]:
    """Replay fixed fingerings {(piece, index): [(hand, global set), ...]} (or a function
    info -> list) through unmasked environments: moves the masks would forbid are kept and counted
    as infeasible. Used for S2, Q-learning, M0 and the pianists."""
    import dataclasses

    from .actions import N_SETS

    cfg = dataclasses.replace(pool.cfg, use_mask=False, lam=0.0)
    out = []
    for info in (infos if infos is not None else pool.infos):
        fing = fingerings(info) if callable(fingerings) else fingerings.get((info.seg.piece, info.seg.index))
        if fing is None:
            continue
        ep = Episode(cfg, pool.model)
        ep.reset(info)
        for h, s in fing:
            ep.step(int(s) + (N_SETS if cfg.mode == "assign" and h == "R" else 0))
        out.append(ep)
    return out
