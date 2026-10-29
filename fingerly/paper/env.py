"""PianoFingering environment as in Iman & Ahn (2024), section III-B / III-C.

State   s_t = (n_{t-2}, a_{t-2}, n_{t-1}, a_{t-1}, n_t), one-hot encoded: 88 keys (A0..C8) for
        each note and 5 fingers for each action, concatenated. That is 3 * 88 + 2 * 5 = 274
        numbers; the paper says 1x273, which does not add up, so we keep 274. Missing history at
        the start of the piece is all zeros.
Action  a_t in {0..4} -> finger a_t + 1. No action masking: every finger is always allowed.
Reward  R_{t+1} = -D_t(s_t, a_t), the 12-rule difficulty (Algorithm 1).
One episode is the whole piece.
"""

from __future__ import annotations

from typing import List, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .difficulty import step_difficulty
from .pieces import Piece

N_KEYS = 88
LOWEST = 21  # A0
OBS_DIM = 3 * N_KEYS + 2 * 5


class PianoFingeringEnv(gym.Env):
    def __init__(self, piece: Piece):
        super().__init__()
        self.piece = piece
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(OBS_DIM,), dtype=np.float32)
        self.t = 0
        self.fingers: List[int] = []

    def _obs(self) -> np.ndarray:
        o = np.zeros(OBS_DIM, np.float32)
        if self.t >= len(self.piece):
            return o
        P = self.piece.pitches
        for slot, back in enumerate((2, 1, 0)):
            i = self.t - back
            if i >= 0:
                o[slot * N_KEYS + P[i] - LOWEST] = 1.0
        base = 3 * N_KEYS
        for slot, back in enumerate((2, 1)):
            if len(self.fingers) >= back:
                o[base + slot * 5 + self.fingers[-back] - 1] = 1.0
        return o

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.t = 0
        self.fingers = []
        return self._obs(), {}

    def step(self, action):
        f = int(action) + 1
        P = self.piece.pitches
        self.fingers.append(f)
        nxt = P[self.t + 1] if self.t + 1 < len(P) else None
        d = step_difficulty(P[: self.t + 1], self.fingers, self.t, nxt)
        self.t += 1
        done = self.t >= len(P)
        return self._obs(), -d, done, False, {"difficulty": d}
