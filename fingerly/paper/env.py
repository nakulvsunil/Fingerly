"""The paper's environment `PianoFingering-V*` (README §A.1).

One episode = one pass through the piece. State s_t = (n_{t-2}, a_{t-2}, n_{t-1}, a_{t-1}, n_t),
one-hot: each note over 88 keys (A0..C8), each finger over 5 -> 3*88 + 2*5 = 274 inputs
(the paper prints 1x273; README §A.7 item 1). Notes or fingers before the start of the piece
are all-zero blocks. No action mask. Reward = minus the 12-rule difficulty charged at note t.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import gymnasium as gym
import numpy as np

from .rules import RuleConfig, step_breakdown

N_KEYS, N_FINGERS = 88, 5
OBS_DIM = 3 * N_KEYS + 2 * N_FINGERS  # 274
LOWEST = 21  # A0


class PaperFingeringEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, pitches: Sequence[int], cfg: RuleConfig = RuleConfig()):
        self.pitches = list(pitches)
        self.cfg = cfg
        self.observation_space = gym.spaces.Box(0.0, 1.0, (OBS_DIM,), np.float32)
        self.action_space = gym.spaces.Discrete(N_FINGERS)
        self.t = 0
        self.fingers: List[int] = []

    def _note(self, out: np.ndarray, offset: int, t: int) -> None:
        if 0 <= t < len(self.pitches):
            out[offset + self.pitches[t] - LOWEST] = 1.0

    def _finger(self, out: np.ndarray, offset: int, t: int) -> None:
        if 0 <= t < len(self.fingers):
            out[offset + self.fingers[t] - 1] = 1.0

    def obs(self) -> np.ndarray:
        o = np.zeros(OBS_DIM, dtype=np.float32)
        t = self.t
        self._note(o, 0, t - 2)
        self._finger(o, N_KEYS, t - 2)
        self._note(o, N_KEYS + N_FINGERS, t - 1)
        self._finger(o, 2 * N_KEYS + N_FINGERS, t - 1)
        self._note(o, 2 * N_KEYS + 2 * N_FINGERS, t)
        return o

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.fingers = []
        return self.obs(), {}

    def step(self, action: int):
        finger = int(action) + 1
        self.fingers.append(finger)
        rules = step_breakdown(self.pitches, self.fingers, self.t, self.cfg)
        reward = -float(rules.sum())
        self.t += 1
        done = self.t >= len(self.pitches)
        obs = self.obs() if not done else np.zeros(OBS_DIM, dtype=np.float32)
        return obs, reward, done, False, {"rules": rules}
