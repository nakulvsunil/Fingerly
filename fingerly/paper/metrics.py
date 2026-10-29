"""Match rates (Nakamura et al., 2020, as used by Iman & Ahn, section III-D) and the exact
minimum-difficulty fingering under the 12 rules.

Mgen   mean over annotators of the match rate with that annotator
Mhigh  match rate with the closest annotator
Msoft  fraction of notes whose finger at least one annotator chose
Mrec   match rate with the best recombination of the annotators' fingerings. The ground truth
       may switch annotator between notes; each switch costs one match. The source defining the
       exact switching cost could not be checked here, so treat Mrec as approximate.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .difficulty import step_difficulty


def match_rates(pred: Sequence[int], labels: Sequence[Sequence[int]]) -> Dict[str, float]:
    lab = np.array(labels)  # K x N, 0 = not fingered by the right hand
    hit = lab == np.array(pred)[None, :]
    n = lab.shape[1]
    per = hit.sum(1) / n
    out = {"Mgen": float(per.mean()), "Mhigh": float(per.max()), "Msoft": float(hit.any(0).sum() / n)}
    # Mrec: best path through annotators, -1 per switch
    score = hit[:, 0].astype(float)
    for t in range(1, n):
        stay = score
        switch = score.max() - 1.0
        score = np.maximum(stay, switch) + hit[:, t]
    out["Mrec"] = float(score.max() / n)
    return out


def optimal_fingering(pitches: Sequence[int]) -> List[int]:
    """Exact minimum of D(s, a) by dynamic programming over (previous finger, current finger):
    every rule looks back at most two notes, and forward only at the next note's colour."""
    n = len(pitches)
    nxt = lambda t: pitches[t + 1] if t + 1 < n else None  # noqa: E731
    fingers = range(1, 6)
    if n == 1:
        return [min(fingers, key=lambda f: step_difficulty(pitches, [f], 0, nxt(0)))]
    # cost[(f0, f1)] after note 1
    cost, back = {}, []
    for f0 in fingers:
        c0 = step_difficulty(pitches, [f0], 0, nxt(0))
        for f1 in fingers:
            cost[(f0, f1)] = c0 + step_difficulty(pitches, [f0, f1], 1, nxt(1))
    for t in range(2, n):
        new, bk = {}, {}
        for (fa, fb), c in cost.items():
            for fc in fingers:
                v = c + step_difficulty(pitches[t - 2: t + 1], [fa, fb, fc], 2, nxt(t))
                if v < new.get((fb, fc), float("inf")):
                    new[(fb, fc)], bk[(fb, fc)] = v, fa
        cost = new
        back.append(bk)
    fb, fc = min(cost, key=cost.get)
    out = [fc, fb]
    for bk in reversed(back):
        fa = bk[(fb, fc)]
        out.append(fa)
        fb, fc = fa, fb
    return out[::-1]
