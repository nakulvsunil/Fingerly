"""Exact minimum-difficulty fingering by dynamic programming (README §A.4, check 3).

The cost charged at note t depends only on (a_{t-2}, a_{t-1}, a_t) and fixed pitches, so a DP
over the 25 pair states (a_{t-1}, a_t) is exact.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .rules import RuleConfig, note_rules

F = (1, 2, 3, 4, 5)


def _p(pitches, k):
    return pitches[k] if 0 <= k < len(pitches) else None


def optimal_fingering(pitches: Sequence[int], cfg: RuleConfig = RuleConfig()) -> Tuple[float, List[int]]:
    """(minimum difficulty D, one fingering that reaches it)."""
    n = len(pitches)
    c0 = lambda a0: note_rules(None, None, pitches[0], _p(pitches, 1), None, None, a0, cfg).sum()  # noqa: E731
    if n == 1:
        best = min(F, key=c0)
        return float(c0(best)), [best]
    # V[a1, a0] = cheapest cost of notes 0..t ending with fingers (a1, a0) on notes (t-1, t)
    V = np.full((6, 6), np.inf)
    for a1 in F:
        for a0 in F:
            V[a1, a0] = c0(a1) + note_rules(None, pitches[0], pitches[1], _p(pitches, 2), None, a1, a0, cfg).sum()
    back = []
    for t in range(2, n):
        p2, p1, p0, pn = pitches[t - 2], pitches[t - 1], pitches[t], _p(pitches, t + 1)
        W = np.full((6, 6), np.inf)
        B = np.zeros((6, 6), dtype=int)
        for a1 in F:
            for a0 in F:
                for a2 in F:
                    v = V[a2, a1] + note_rules(p2, p1, p0, pn, a2, a1, a0, cfg).sum()
                    if v < W[a1, a0]:
                        W[a1, a0], B[a1, a0] = v, a2
        V = W
        back.append(B)
    a1, a0 = np.unravel_index(np.argmin(V), V.shape)
    best = float(V[a1, a0])
    seq = [int(a0), int(a1)]
    for B in reversed(back):
        a2 = B[seq[-1], seq[-2]]
        seq.append(int(a2))
    return best, seq[::-1]
