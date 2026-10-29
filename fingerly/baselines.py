"""Baselines for one-hand monophonic fingering.

* RandomValid   - uniform over fingers the mask allows (sanity floor).
* Greedy        - lowest immediate effort, no look-ahead.
* OptimalCost   - Viterbi over the cost model: the exact minimum-total-effort fingering.
                  The RL agent optimises the same cost (gamma ~ 1), so this is the ceiling
                  for reward; how far the agent is from it measures learning, not the reward.
* HMM1          - supervised first-order HMM trained on PIG labels:
                  P(f_t | f_{t-1}) * P(interval_t | f_{t-1}, f_t), decoded with Viterbi.
                  A simplified (semitone, first-order) version of Nakamura et al. (2020).
                  It is not masked, so it can produce moves the cost model calls impossible.
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence

import numpy as np

from .cost import FINGERS, CostConfig, mirrored_interval, step_cost, valid_fingers
from .pig import Passage, is_black


def _iv_list(p: Passage) -> List[int]:
    return [0] + [mirrored_interval(a, b, p.hand) for a, b in zip(p.pitches, p.pitches[1:])]


class RandomValid:
    name = "random-valid"

    def __init__(self, cfg: Optional[CostConfig] = None, seed: int = 0):
        self.cfg = cfg or CostConfig()
        self.rng = random.Random(seed)

    def predict(self, p: Passage) -> List[int]:
        out, prev = [], None
        for iv in _iv_list(p):
            ok = [f for f, m in zip(FINGERS, valid_fingers(prev, iv, self.cfg)) if m]
            prev = self.rng.choice(ok)
            out.append(prev)
        return out


class Greedy:
    name = "greedy-cost"

    def __init__(self, cfg: Optional[CostConfig] = None):
        self.cfg = cfg or CostConfig()

    def predict(self, p: Passage) -> List[int]:
        out, prev = [], None
        for iv, pitch in zip(_iv_list(p), p.pitches):
            mask = valid_fingers(prev, iv, self.cfg)
            best = min(
                (f for f, m in zip(FINGERS, mask) if m),
                key=lambda f: (step_cost(prev, f, iv, is_black(pitch), self.cfg).effort, abs(f - 3)),
            )
            out.append(best)
            prev = best
        return out


class OptimalCost:
    name = "optimal-cost (DP)"

    def __init__(self, cfg: Optional[CostConfig] = None):
        self.cfg = cfg or CostConfig()

    def predict(self, p: Passage) -> List[int]:
        ivs, n = _iv_list(p), len(p)
        INF = float("inf")
        dp = np.full((n, 5), INF)
        back = np.zeros((n, 5), dtype=int)
        for f in FINGERS:
            dp[0, f - 1] = step_cost(None, f, 0, is_black(p.pitches[0]), self.cfg).effort
        for t in range(1, n):
            black = is_black(p.pitches[t])
            for prev in FINGERS:
                if dp[t - 1, prev - 1] == INF:
                    continue
                mask = valid_fingers(prev, ivs[t], self.cfg)
                for f in FINGERS:
                    if not mask[f - 1]:
                        continue
                    c = dp[t - 1, prev - 1] + step_cost(prev, f, ivs[t], black, self.cfg).effort
                    if c < dp[t, f - 1]:
                        dp[t, f - 1], back[t, f - 1] = c, prev
        f = int(dp[-1].argmin()) + 1
        out = [f]
        for t in range(n - 1, 0, -1):
            f = int(back[t, f - 1])
            out.append(f)
        return out[::-1]


class HMM1:
    name = "hmm-1st-order"

    def __init__(self, max_iv: int = 24, alpha: float = 0.5):
        self.max_iv = max_iv
        self.alpha = alpha
        self.log_init = np.zeros(5)
        self.log_trans = np.zeros((5, 5))
        self.log_emit = np.zeros((5, 5, 2 * max_iv + 1))

    def _bin(self, iv: int) -> int:
        return int(np.clip(iv, -self.max_iv, self.max_iv)) + self.max_iv

    def fit(self, passages: Sequence[Passage]) -> "HMM1":
        init = np.full(5, self.alpha)
        trans = np.full((5, 5), self.alpha)
        emit = np.full((5, 5, 2 * self.max_iv + 1), self.alpha)
        for p in passages:
            for labels in p.labels:
                ivs = _iv_list(p)
                if labels and labels[0]:
                    init[labels[0] - 1] += 1
                for t in range(1, len(p)):
                    a, b = labels[t - 1], labels[t]
                    if a and b:
                        trans[a - 1, b - 1] += 1
                        emit[a - 1, b - 1, self._bin(ivs[t])] += 1
        self.log_init = np.log(init / init.sum())
        self.log_trans = np.log(trans / trans.sum(1, keepdims=True))
        self.log_emit = np.log(emit / emit.sum(2, keepdims=True))
        return self

    def predict(self, p: Passage) -> List[int]:
        ivs, n = _iv_list(p), len(p)
        dp = np.zeros((n, 5))
        back = np.zeros((n, 5), dtype=int)
        dp[0] = self.log_init
        for t in range(1, n):
            # score[prev, f]
            score = dp[t - 1][:, None] + self.log_trans + self.log_emit[:, :, self._bin(ivs[t])]
            back[t] = score.argmax(0)
            dp[t] = score.max(0)
        f = int(dp[-1].argmax())
        out = [f + 1]
        for t in range(n - 1, 0, -1):
            f = int(back[t, f])
            out.append(f + 1)
        return out[::-1]


class HMMSets:
    """S2 (README §C.9.1): Part B's first-order HMM extended to finger sets, one hand stream at a
    time. Hidden state = the finger set (31); P(s_t | s_{t-1}) renormalised over the sets of the
    current chord size; emission P(interval | s_{t-1}, s_t), the mirrored interval between the
    events' thumb-side notes, clipped to +-24. Trained on the pianists' fingerings of the training
    pieces (each annotator's own pitch-order chord fingerings); decoded with Viterbi. Unmasked."""

    name = "S2 hmm-sets"

    def __init__(self, max_iv: int = 24, alpha: float = 0.5):
        from .twohand.actions import N_SETS

        self.n = N_SETS
        self.max_iv, self.alpha = max_iv, alpha

    def _iv(self, prev_e, e, hand):
        from .twohand.features import key_of

        d = key_of(e.pitches, hand) - key_of(prev_e.pitches, hand)
        d = d if hand == "R" else -d
        return int(np.clip(d, -self.max_iv, self.max_iv)) + self.max_iv

    def fit(self, segments) -> "HMMSets":
        from .twohand.actions import N_SETS, SETS_OF_SIZE, fingers_of

        n, B = N_SETS, 2 * self.max_iv + 1
        init = np.full(n, self.alpha)
        trans = np.full((n, n), self.alpha)
        emit = np.full((n, n, B), self.alpha)
        for seg in segments:
            for hand in ("L", "R"):
                idx = seg.hand_events(hand)
                for a in range(seg.n_annotators):
                    prev = None
                    for i in idx:
                        e = seg.events[i]
                        row = e.labels[a]
                        s = None
                        if all(h == hand and f for h, f in row):
                            fs = tuple(f for _, f in row)
                            s = next((int(x) for x in SETS_OF_SIZE[e.size] if fingers_of(int(x), hand) == fs), None)
                        if s is not None:
                            if prev is None:
                                init[s] += 1
                            else:
                                trans[prev[0], s] += 1
                                emit[prev[0], s, self._iv(prev[1], e, hand)] += 1
                        prev = (s, e) if s is not None else None
        self.log_init = np.log(init / init.sum())
        self.log_trans = np.log(trans / trans.sum(1, keepdims=True))
        self.log_emit = np.log(emit / emit.sum(2, keepdims=True))
        return self

    def predict_hand(self, seg, hand):
        """[(event index, global set)] for one hand's events of a segment."""
        from .twohand.actions import SETS_OF_SIZE

        idx = seg.hand_events(hand)
        if not idx:
            return []
        cands = [SETS_OF_SIZE[seg.events[i].size] for i in idx]
        V = self.log_init[cands[0]] - np.logaddexp.reduce(self.log_init[cands[0]])
        backs = []
        for k in range(1, len(idx)):
            c0, c1 = cands[k], cands[k - 1]
            lt = self.log_trans[np.ix_(c1, c0)]
            lt = lt - np.logaddexp.reduce(lt, axis=1, keepdims=True)  # renormalise over this size
            b = self._iv(seg.events[idx[k - 1]], seg.events[idx[k]], hand)
            score = V[:, None] + lt + self.log_emit[np.ix_(c1, c0)][:, :, b]
            backs.append(score.argmax(0))
            V = score.max(0)
        j = int(V.argmax())
        loc = [j]
        for k in range(len(idx) - 1, 0, -1):
            j = int(backs[k - 1][j])
            loc.append(j)
        loc = loc[::-1]
        return [(i, int(cands[k][loc[k]])) for k, i in enumerate(idx)]

    def predict_segment(self, seg, order):
        """Per visited event (in `order`): (hand, global set), hands as annotated."""
        got = {}
        for hand in ("L", "R"):
            for i, s in self.predict_hand(seg, hand):
                got[i] = (hand, s)
        return [got[i] for i in order]
