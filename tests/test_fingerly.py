import random

import numpy as np
import pytest

from fingerly.baselines import HMM1, Greedy, OptimalCost, RandomValid
from fingerly.cost import CostConfig, sequence_cost, step_cost, valid_fingers
from fingerly.env import FingeringEnv, obs_size, run_policy
from fingerly.fake_pig import write_fake_pig
from fingerly.metrics import evaluate_predictor, human_agreement
from fingerly.pig import (Note, Passage, is_black, load_pig, make_split, monophonic_runs,
                          parse_finger, pitch_to_midi)
from fingerly.synth import synthetic_passages

C_MAJOR = [60, 62, 64, 65, 67, 69, 71, 72]


def P(pitches, hand="R", labels=None, piece="x"):
    on = [float(i) for i in range(len(pitches))]
    return Passage(piece, hand, list(pitches), on, [o + 1 for o in on],
                   [(int(o * 1000), p) for o, p in zip(on, pitches)], labels or [])


# ----------------------------------------------------------------------------- parsing
@pytest.mark.parametrize("name,midi", [("C4", 60), ("F#4", 66), ("Bb2", 46), ("E-4", 63),
                                       ("A0", 21), ("C8", 108), ("B#3", 60), ("Fx4", 67)])
def test_pitch_to_midi(name, midi):
    assert pitch_to_midi(name) == midi


def test_black_keys():
    assert [is_black(p) for p in range(60, 72)] == [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]


@pytest.mark.parametrize("tok,out", [("3", ("R", 3)), ("-5", ("L", 5)), ("3_1", ("R", 3)),
                                     ("-3_-1", ("L", 3)), ("0", (None, None)), ("x", (None, None))])
def test_parse_finger(tok, out):
    assert parse_finger(tok) == out


def _n(i, on, off, pitch):
    return Note(i, on, off, pitch, 0, "R", 1)


def test_monophonic_filter_drops_chords_and_held_notes():
    notes = [_n(i, i * 0.5, i * 0.5 + 0.5, 60 + i) for i in range(10)]          # run 1: 10 notes
    notes += [_n(20, 5.0, 5.5, 60), _n(21, 5.0, 5.5, 64)]                      # chord
    notes += [_n(30 + i, 6 + i * 0.5, 6.5 + i * 0.5, 70 - i) for i in range(9)]  # run 2: 9 notes
    notes += [_n(40, 11.0, 14.0, 50)]                                          # long note ends run 2...
    notes += [_n(41 + i, 11.5 + i * 0.25, 11.75 + i * 0.25, 60 + i) for i in range(9)]  # ...notes under it are dropped
    runs = monophonic_runs(notes, min_len=8)
    assert [len(r) for r in runs] == [10, 10]
    assert runs[1][-1].note_id == 40
    assert all(n.note_id < 41 for r in runs for n in r)
    assert monophonic_runs(notes, min_len=11) == []


def test_fake_pig_roundtrip(tmp_path):
    write_fake_pig(tmp_path, pieces=20, multi=5, seed=1)
    pig = load_pig(tmp_path)
    assert len(pig) == 20
    assert sum(len(v) >= 2 for v in pig.values()) == 5
    split = make_split(pig, "R", val_pieces=3, seed=0)
    assert len(split.test_pieces) == 5 and len(split.val_pieces) == 3 and len(split.train_pieces) == 12
    assert not set(split.train_pieces) & set(split.test_pieces)
    for p in split.test:
        assert len(p.labels) >= 4 and all(len(l) == len(p) for l in p.labels)
        assert len(p) >= 8
    # the first annotator of fake data is the optimal-cost fingering
    dp = OptimalCost()
    p = split.test[0]
    assert sequence_cost(p.pitches, p.labels[0])[0] == pytest.approx(sequence_cost(p.pitches, dp.predict(p))[0])


# ----------------------------------------------------------------------------- cost model
def test_c_major_scale_standard_fingering():
    total, imp, d = sequence_cost(C_MAJOR, [1, 2, 3, 1, 2, 3, 4, 5])
    # thumb-under F (crossing 1) + finger 4 (weak 1) + finger 5 (weak 1)
    assert (total, imp) == (3.0, 0)
    assert [x.crossing for x in d] == [False, False, False, True, False, False, False, False]


def test_same_finger_scale_is_expensive():
    total, imp, d = sequence_cost(C_MAJOR, [2] * 8)
    assert total == 7 * 2.0 and imp == 0 and all(x.shift for x in d[1:])


def test_thumb_on_black_and_weak_finger():
    assert step_cost(None, 1, 0, True, CostConfig()).effort == 1
    assert step_cost(None, 5, 0, False, CostConfig()).effort == 1
    assert step_cost(None, 4, 0, True, CostConfig()).effort == 1
    assert step_cost(None, 3, 0, True, CostConfig()).effort == 0


def test_stretch_and_cramp():
    cfg = CostConfig()
    # 1 -> 5 over a 10th (16 semitones) is out of reach -> hand shift, cost 2 + weak 1
    sc = step_cost(1, 5, 16, False, cfg)
    assert sc.shift and sc.effort == 3
    # 1 -> 5 over 14 semitones: MaxComf 13 -> 1 too wide, + weak finger
    assert step_cost(1, 5, 14, False, cfg).effort == 2
    # 2 -> 3 on a repeated note: MinComf 1 -> 1 too close
    assert step_cost(2, 3, 0, False, cfg).effort == 1


def test_impossible_move_and_mask():
    cfg = CostConfig()
    sc = step_cost(2, 3, 6, False, cfg)  # 2->3 a tritone apart: beyond MaxPrac 5
    assert sc.impossible
    assert valid_fingers(2, 6, cfg) == [False, True, False, True, True]
    assert valid_fingers(None, 0, cfg) == [True] * 5
    assert valid_fingers(2, 14, cfg) == [True] * 5  # out of reach -> hand shifts, any finger


def test_mask_never_empty_and_matches_cost():
    cfg = CostConfig()
    for prev in range(1, 6):
        for iv in range(-30, 31):
            m = valid_fingers(prev, iv, cfg)
            assert any(m)
            for f in range(1, 6):
                assert m[f - 1] == (not step_cost(prev, f, iv, False, cfg).impossible)


def test_left_hand_is_mirror_of_right_hand():
    rng = random.Random(0)
    for _ in range(50):
        lh = [rng.randint(36, 60) for _ in range(12)]
        rh = [124 - p for p in lh]  # mirror around D4 keeps black keys black
        fingers = [rng.randint(1, 5) for _ in range(12)]
        assert sequence_cost(lh, fingers, "L")[0] == sequence_cost(rh, fingers, "R")[0]


def test_hand_size_scaling():
    small, large = CostConfig(span_scale=0.8), CostConfig(span_scale=1.2)
    assert step_cost(1, 5, 14, False, large).effort < step_cost(1, 5, 14, False, small).effort


# ----------------------------------------------------------------------------- environment
def test_env_api():
    from gymnasium.utils.env_checker import check_env
    env = FingeringEnv(synthetic_passages(10))
    check_env(env, skip_render_check=True)
    obs, info = env.reset(seed=0)
    assert obs.shape == (obs_size(3),) == (24,)
    assert info["action_mask"].dtype == bool


def test_env_reward_equals_cost_model():
    ps = synthetic_passages(30, seed=3)
    env = FingeringEnv(ps)
    dp = OptimalCost()
    for p in ps:
        fingers = dp.predict(p)
        env.reset(options={"passage": p})
        ret = sum(env.step(f - 1)[1] for f in fingers)
        assert -ret == pytest.approx(sequence_cost(p.pitches, fingers, p.hand)[0])
        assert env.fingers == fingers


def test_random_masked_agent_never_impossible():
    rng = np.random.default_rng(0)
    for hand in "RL":
        ps = synthetic_passages(100, hand, seed=4)
        env = FingeringEnv(ps, hand=hand)
        for p in ps:
            fingers = run_policy(env, p, lambda o, m: int(rng.choice(np.flatnonzero(m))))
            assert env.n_impossible == 0 and len(fingers) == len(p)


def test_unmasked_env_charges_penalty():
    env = FingeringEnv([P([60, 66, 60] + C_MAJOR)], use_mask=False)
    env.reset(options={"passage_index": 0})
    env.step(1)                       # finger 2 on C4
    _, r, _, _, info = env.step(2)    # finger 3 on F#4: 2->3 over 6 semitones
    assert info["impossible"] and r == pytest.approx(-10.0)


def test_expert_bonus():
    p = P(C_MAJOR, labels=[[1, 2, 3, 1, 2, 3, 4, 5]])
    env = FingeringEnv([p], lam=1.0)
    env.reset(options={"passage_index": 0})
    _, r, _, _, info = env.step(0)
    assert info["expert_match"] == 1.0 and r == pytest.approx(1.0)


# ----------------------------------------------------------------------------- baselines & metrics
def test_dp_is_optimal_and_beats_greedy_and_random():
    cfg = CostConfig()
    ps = synthetic_passages(60, seed=5)
    cost = lambda b: sum(sequence_cost(p.pitches, b.predict(p))[0] for p in ps)  # noqa: E731
    assert cost(OptimalCost(cfg)) <= cost(Greedy(cfg)) <= cost(RandomValid(cfg))
    # brute force on short passages
    import itertools
    for p in synthetic_passages(15, seed=6):
        q = P(p.pitches[:6])
        best = min(sequence_cost(q.pitches, fs)[0] for fs in itertools.product(range(1, 6), repeat=6)
                   if sequence_cost(q.pitches, fs)[1] == 0)
        assert sequence_cost(q.pitches, OptimalCost(cfg).predict(q))[0] == best


def test_hmm_learns_its_training_fingering():
    ps = synthetic_passages(80, seed=7)
    dp = OptimalCost()
    for p in ps:
        p.labels = [dp.predict(p)]
    hmm = HMM1().fit(ps)
    m = evaluate_predictor(hmm.predict, ps)
    assert m["Mgen"] > 0.6


def test_match_metrics():
    p = P([60, 62, 64], labels=[[1, 2, 3], [1, 2, 4]], piece="a")
    m = evaluate_predictor(lambda q: [1, 2, 3], [p])
    assert m["Mgen"] == pytest.approx(5 / 6)
    assert m["Mhigh"] == pytest.approx(1.0)
    assert m["Msoft"] == pytest.approx(1.0)
    m = evaluate_predictor(lambda q: [1, 2, 5], [p])
    assert m["Msoft"] == pytest.approx(2 / 3)
    h = human_agreement([p])
    assert h["Mgen"] == pytest.approx(2 / 3)


# ----------------------------------------------------------------------------- export
def test_cost_breakdown_adds_up_to_step_cost():
    from fingerly.cost import cost_breakdown

    cfg, rng = CostConfig(), random.Random(0)
    for _ in range(5000):
        pf, f, iv, b = rng.choice([None, 1, 2, 3, 4, 5]), rng.randint(1, 5), rng.randint(-20, 20), rng.random() < 0.4
        sc = step_cost(pf, f, iv, b, cfg)
        assert sum(cost_breakdown(pf, f, iv, b, cfg).values()) == pytest.approx(
            sc.effort + (cfg.impossible if sc.impossible else 0.0))


def test_export_fingering_matches_sequence_cost(tmp_path):
    from fingerly.export import export_fingering, passages_of
    from fingerly.pig import find_fingering_dir, read_fingering_file

    write_fake_pig(tmp_path, pieces=3, multi=1, seed=2)
    path = sorted(find_fingering_dir(tmp_path).glob("*_fingering.txt"))[0]
    dp = OptimalCost()
    data = export_fingering(path, dp.predict, "dp", hand="R", max_gap=0.5)
    _, passages = passages_of(read_fingering_file(path), "R")
    expected = sum(sequence_cost(p.pitches, dp.predict(p), "R")[0] for p in passages)
    assert data["total_cost"] == pytest.approx(expected)
    assert len(data["notes"]) == sum(len(p) for p in passages)
    on = [n["onset"] for n in data["notes"]]
    assert on == sorted(on)
    assert all(b["onset"] - a["offset"] <= 0.5 + 1e-6 for a, b in zip(data["notes"], data["notes"][1:]))
