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
