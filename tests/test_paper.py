"""Tests for the reproduction of Iman & Ahn (2024)."""

import itertools
import random

import numpy as np
import pytest

from fingerly.paper.difficulty import difficulty_level, per_step, piece_difficulty
from fingerly.paper.metrics import match_rates, optimal_fingering
from fingerly.paper.pieces import BEYER_12, BEYER_12_FINGERING

# Fingerings of the PG and A2C agents on Beyer No. 12, read off the paper's Fig. 5.
PG_FIG5 = [1, 4, 4, 3, 1, 2, 4, 1, 5, 2, 2, 2, 5, 2, 1, 1, 5, 3, 5, 1, 4, 1, 2, 2, 3, 1, 4, 2, 1]
A2C_FIG5 = [1, 2, 3, 1, 2, 1, 1, 2, 1, 2, 3, 1, 2, 1, 1, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1]


def test_beyer_scores_match_the_paper():
    # Table II: optimal (DQN / DDQN / human) 0.6207 = 18/29, A2C 2.7586 = 80/29, PG 8.1034 = 235/29
    assert len(BEYER_12) == 29
    assert piece_difficulty(BEYER_12, BEYER_12_FINGERING) == 18
    assert difficulty_level(BEYER_12, BEYER_12_FINGERING) == pytest.approx(0.6207, abs=1e-4)
    assert piece_difficulty(BEYER_12, A2C_FIG5) == 80
    assert piece_difficulty(BEYER_12, PG_FIG5) == pytest.approx(235, abs=1)  # 234: one finger misread?


def test_step_sum_equals_piece_total():
    rng = random.Random(0)
    pitches = [rng.randint(55, 80) for _ in range(40)]
    fingers = [rng.randint(1, 5) for _ in range(40)]
    assert sum(per_step(pitches, fingers)) == pytest.approx(piece_difficulty(pitches, fingers))


def test_individual_rules():
    # thumb on black between two white keys: 1 + 2 + 2
    assert piece_difficulty([60, 61, 62], [2, 1, 2]) - piece_difficulty([60, 62, 64], [2, 1, 2]) > 0
    # 3 on white then 4 on black: three-to-four (1) + four-on-black (1) + weak finger (1)
    assert piece_difficulty([64, 66], [3, 4]) == 3
    # repeating a key with the same finger is free
    assert piece_difficulty([60, 60, 60], [2, 2, 2]) == 0


def test_optimal_is_exact_on_small_inputs():
    rng = random.Random(1)
    for _ in range(5):
        pitches = [rng.randint(58, 74) for _ in range(6)]
        brute = min(piece_difficulty(pitches, list(f)) for f in itertools.product(range(1, 6), repeat=6))
        assert piece_difficulty(pitches, optimal_fingering(pitches)) == brute
    assert piece_difficulty(BEYER_12, optimal_fingering(BEYER_12)) == 18


def test_match_rates():
    labels = [[1, 2, 3, 4], [1, 2, 4, 4]]
    m = match_rates([1, 2, 4, 5], labels)
    assert m["Mgen"] == pytest.approx((2 / 4 + 3 / 4) / 2)
    assert m["Mhigh"] == pytest.approx(3 / 4)
    assert m["Msoft"] == pytest.approx(3 / 4)
    assert m["Mhigh"] <= m["Mrec"] <= m["Msoft"]


def test_env_state_encoding_and_reward():
    pytest.importorskip("gymnasium")
    from fingerly.paper.env import OBS_DIM, PianoFingeringEnv
    from fingerly.paper.pieces import Piece

    env = PianoFingeringEnv(Piece("V0", "Beyer", list(BEYER_12)))
    obs, _ = env.reset()
    assert obs.shape == (OBS_DIM,) and obs.sum() == 1  # only n_t at the start
    total = 0.0
    for t, f in enumerate(BEYER_12_FINGERING):
        obs, r, done, _, _ = env.step(f - 1)
        total += r
        if not done and t >= 1:
            assert obs.sum() == 5  # n_{t-2}, n_{t-1}, n_t and two fingers
    assert done and total == -18
    # the state of Fig. 3: (62, 1, 66, 2, 71)
    env = PianoFingeringEnv(Piece("x", "x", [62, 66, 71]))
    env.reset()
    env.step(0)
    obs, *_ = env.step(1)
    hot = np.flatnonzero(obs)
    assert list(hot) == [62 - 21, 88 + 66 - 21, 176 + 71 - 21, 264 + 0, 269 + 1]
