"""Gymnasium environment: the score is the environment, one note per step, one hand.

Action  a in {0..4}  ->  finger a + 1.
Reward  R_t = -alpha * C_effort - beta * P_impossible + lambda * B_expert
State   (lookahead k = 3 gives 24 numbers)

    current note        (pitch - 60) / 24, black?                                   2
    next k notes        interval from the current note / 12, black?, exists?       3k
    the hand            played yet?                                                 1
                        interval from its last key to this note / 12                1
                        its last finger, one-hot                                    5
                        interval from its second-to-last key to this note / 12      1
                        its second-to-last finger, one-hot                          5

Intervals are mirrored for the left hand (see cost.py). Transitions are deterministic:
the next note always comes from the passage. The episode ends after the last note.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .cost import CostConfig, mirrored_interval, step_cost, valid_fingers
from .pig import Passage

INTERVAL_CLIP = 3.0  # in octaves after dividing by 12


def obs_size(lookahead: int) -> int:
    return 2 + 3 * lookahead + 13


class FingeringEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        passages: Sequence[Passage],
        hand: str = "R",
        lookahead: int = 3,
        cost_cfg: Optional[CostConfig] = None,
        alpha: float = 1.0,
        beta: float = 1.0,
        lam: float = 0.0,
        use_mask: bool = True,
        reward_scale: float = 1.0,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        if not passages:
            raise ValueError("FingeringEnv needs at least one passage")
        self.passages = list(passages)
        self.hand = hand
        self.k = lookahead
        self.cfg = cost_cfg or CostConfig()
        self.alpha, self.beta, self.lam = alpha, beta, lam
        self.use_mask = use_mask
        self.reward_scale = reward_scale
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_size(lookahead),), dtype=np.float32)

        self.passage: Passage = self.passages[0]
        self.t = 0
        self.fingers: List[int] = []
        self.total_effort = 0.0
        self.n_impossible = 0

    # ------------------------------------------------------------------ helpers
    def _iv(self, from_pitch: int, to_pitch: int) -> int:
        return mirrored_interval(from_pitch, to_pitch, self.hand)

    def _last(self, back: int):
        """(pitch, finger) of the hand's note `back` steps ago (1 = previous), or (None, None)."""
        if len(self.fingers) < back:
            return None, None
        i = self.t - back
        return self.passage.pitches[i], self.fingers[i]

    def _current_interval(self) -> int:
        prev_p, _ = self._last(1)
        return 0 if prev_p is None else self._iv(prev_p, self.passage.pitches[self.t])

    def action_masks(self) -> np.ndarray:
        """Boolean mask of allowed fingers for the current note (sb3-contrib compatible)."""
        if self.t >= len(self.passage):
            return np.ones(5, dtype=bool)
        if not self.use_mask:
            return np.ones(5, dtype=bool)
        _, prev_f = self._last(1)
        return np.array(valid_fingers(prev_f, self._current_interval(), self.cfg), dtype=bool)

    def _obs(self) -> np.ndarray:
        o = np.zeros(self.observation_space.shape, dtype=np.float32)
        if self.t >= len(self.passage):
            return o
        P = self.passage.pitches
        cur = P[self.t]
        o[0] = (cur - 60) / 24.0
        o[1] = float(cur % 12 in {1, 3, 6, 8, 10})
        j = 2
        for d in range(1, self.k + 1):
            if self.t + d < len(P):
                nxt = P[self.t + d]
                o[j] = np.clip(self._iv(cur, nxt) / 12.0, -INTERVAL_CLIP, INTERVAL_CLIP)
                o[j + 1] = float(nxt % 12 in {1, 3, 6, 8, 10})
                o[j + 2] = 1.0
            j += 3
        o[j] = float(len(self.fingers) > 0)
        j += 1
        for back in (1, 2):
            p, f = self._last(back)
            if p is not None:
                o[j] = np.clip(self._iv(p, cur) / 12.0, -INTERVAL_CLIP, INTERVAL_CLIP)
                o[j + f] = 1.0  # one-hot at j+1 .. j+5
            j += 6
        return o

    def _info(self) -> dict:
        return {"action_mask": self.action_masks(), "t": self.t}

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        options = options or {}
        if "passage" in options:
            self.passage = options["passage"]
        elif "passage_index" in options:
            self.passage = self.passages[options["passage_index"]]
        else:
            self.passage = self.passages[int(self.np_random.integers(len(self.passages)))]
        self.t = 0
        self.fingers = []
        self.total_effort = 0.0
        self.n_impossible = 0
        return self._obs(), self._info()

    def step(self, action):
        a = int(action)
        if not 0 <= a < 5:
            raise ValueError(f"action must be in 0..4, got {action}")
        if self.t >= len(self.passage):
            raise RuntimeError("step() called after the episode ended; call reset()")
        f = a + 1
        _, prev_f = self._last(1)
        cur = self.passage.pitches[self.t]
        sc = step_cost(prev_f, f, self._current_interval(), cur % 12 in {1, 3, 6, 8, 10}, self.cfg)

        bonus = 0.0
        if self.lam and self.passage.labels:
            bonus = float(self.passage.labels[0][self.t] == f)
        penalty = self.cfg.impossible if sc.impossible else 0.0
        reward = -self.alpha * sc.effort - self.beta * penalty + self.lam * bonus

        self.total_effort += sc.effort + penalty
        self.n_impossible += int(sc.impossible)
        self.fingers.append(f)
        self.t += 1
        terminated = self.t >= len(self.passage)
        info = self._info()
        info.update(effort=sc.effort, impossible=sc.impossible, finger=f, expert_match=bonus)
        if self.render_mode == "ansi":
            print(self.render())
        return self._obs(), reward * self.reward_scale, terminated, False, info

    def render(self):
        names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        notes = [f"{names[p % 12]}{p // 12 - 1}" for p in self.passage.pitches[: self.t]]
        return " ".join(f"{n}:{f}" for n, f in zip(notes, self.fingers))


# ---------------------------------------------------------------------- rollouts
PolicyFn = Callable[[np.ndarray, np.ndarray], int]  # (obs, mask) -> action 0..4


def run_policy(env: FingeringEnv, passage: Passage, policy: PolicyFn) -> List[int]:
    """Finger one passage with a policy; returns fingers 1..5."""
    obs, info = env.reset(options={"passage": passage})
    done = False
    while not done:
        a = policy(obs, info["action_mask"])
        obs, _, done, _, info = env.step(a)
    return list(env.fingers)
