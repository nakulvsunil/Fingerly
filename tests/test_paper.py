"""Part A (Iman & Ahn 2024 reproduction): reward, DP, environment, Mrec."""

import itertools
import random

import numpy as np
import pytest

from fingerly.metrics import match_rates, recombination_match
from fingerly.paper.dp import optimal_fingering
from fingerly.paper.env import OBS_DIM, PaperFingeringEnv
from fingerly.paper.pieces import load_piece
from fingerly.paper.rules import RULE_NAMES, RuleConfig, breakdown, difficulty

C, D, E, F, G = 60, 62, 64, 65, 67


def test_v0_composer_fingering_scores_18():
    v0 = load_piece("V0")
    assert len(v0) == 29
    b = breakdown(v0.pitches, v0.labels[0])
    assert b.sum() == 18
    assert b[RULE_NAMES.index("weak-finger")] == 9
    assert b[RULE_NAMES.index("three-four-five")] == 6
    assert b[RULE_NAMES.index("three-to-four")] == 3


def test_rule7_counts_only_3_4_5_triples():
    k = RULE_NAMES.index("three-four-five")
    assert breakdown([E, F, G], [3, 4, 5])[k] == 1
    assert breakdown([G, F, E], [5, 4, 3])[k] == 1
    assert breakdown([F, G, F], [4, 5, 4])[k] == 0


def test_v0_optimum_is_18_and_free_same_finger_collapses():
    v0 = load_piece("V0")
    assert optimal_fingering(v0.pitches)[0] == 18
    assert optimal_fingering(v0.pitches, RuleConfig(same_finger="free"))[0] == 0


@pytest.mark.parametrize("same_finger", ["zero-span", "free"])
@pytest.mark.parametrize("table1", ["paper", "parncutt"])
def test_dp_equals_brute_force(same_finger, table1):
    cfg = RuleConfig(table1=table1, same_finger=same_finger)
    rng = random.Random(0)
    for _ in range(8):
        n = rng.randint(1, 5)
        pitches = [rng.randint(55, 80) for _ in range(n)]
        brute = min(difficulty(pitches, f, cfg) for f in itertools.product(range(1, 6), repeat=n))
        D, fing = optimal_fingering(pitches, cfg)
        assert D == brute
        assert difficulty(pitches, fing, cfg) == D


def test_env_shape_and_episode():
    v0 = load_piece("V0")
    env = PaperFingeringEnv(v0.pitches)
    obs, _ = env.reset()
    assert obs.shape == (OBS_DIM,) == (274,)
    assert obs.sum() == 1  # only the current note at t = 0
    total, steps, done = 0.0, 0, False
    for f in v0.labels[0]:
        obs, r, done, _, _ = env.step(f - 1)
        total += r
        steps += 1
    assert done and steps == 29 and total == -18


def test_match_rates_ordering():
    labels = [[1, 2, 3, 4, 5, 1], [1, 3, 3, 2, 5, 2]]
    pred = [1, 3, 3, 4, 5, 2]
    m = match_rates(pred, labels)
    assert m["Mgen"] <= m["Mhigh"] <= m["Mrec"] <= m["Msoft"]
    assert recombination_match(labels[0], labels) == 1.0
