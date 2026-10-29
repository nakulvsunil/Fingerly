"""Part C tests (README §C.15). Phase 1: data pipeline, finger-set actions, cost models."""

import random

import numpy as np
import pytest

from fingerly.pig import Note
from fingerly.twohand.actions import FINGER_SETS, N_SETS, SET_INDEX, SETS_OF_SIZE, fingers_of, set_of, sizes_mask
from fingerly.twohand.prepare import PrepConfig, piece_events
from fingerly.twohand.rules import (Balliauw15, Parncutt12, balliauw15_swap_pair, make_model, merged_cost, position,
                                    thumb_position)

C4, D4, E4, F4, G4, A4, B4, C5 = 60, 62, 64, 65, 67, 69, 71, 72
MIRROR = 124  # reflection around D4 keeps key colours and invisible-key positions


def note(i, on, off, pitch, hand="R", finger=1):
    return Note(i, on, off, pitch, 0 if hand == "R" else 1, hand, finger)


def mirror_stream(stream):
    """Right-hand [(pitches, fingers)] -> the left-hand line mirrored around D4, same fingers."""
    out = []
    for ps, fs in stream:
        pairs = sorted(zip((MIRROR - p for p in ps), fs))
        out.append((tuple(p for p, _ in pairs), tuple(f for _, f in pairs)))
    return out


def random_stream(rng, n=40, chords=True):
    out = []
    for _ in range(n):
        size = rng.choice([1, 1, 1, 2, 3]) if chords else 1
        base = rng.randint(55, 80)
        ps = sorted({base + rng.randint(0, 9) for _ in range(size)})
        fs = sorted(rng.sample(range(1, 6), len(ps)))
        out.append((tuple(ps), tuple(fs)))
    return out


# ---------------------------------------------------------------- invisible black key
def test_invisible_black_key():
    assert position(F4) - position(E4) == 2
    assert position(C5) - position(C4) == 14
    assert position(C5) - position(B4) == 2
    assert position(61) - position(C4) == 1  # C#4


# ---------------------------------------------------------------- mirror
@pytest.mark.parametrize("name", ["parncutt12", "balliauw15"])
@pytest.mark.parametrize("hz", ["all", "extremes"])
def test_mirror_left_equals_right(name, hz):
    m = make_model(name, hz)
    rng = random.Random(1)
    for _ in range(20):
        right = random_stream(rng)
        left = mirror_stream(right)
        for deferred in (False, True):
            assert np.allclose(m.sequence("R", right, deferred), m.sequence("L", left, deferred))


def test_balliauw_swap_reading_equals_mirroring():
    m = Balliauw15()
    for f in range(1, 6):
        for g in range(1, 6):
            for pa, pb in [(60, 64), (64, 65), (67, 55), (71, 72), (62, 62), (60, 79)]:
                mir = m._span_rules(f, m.hand_event("L", [pa], [f]).vals[0], g, m.hand_event("L", [pb], [g]).vals[0])
                assert np.allclose(mir, balliauw15_swap_pair("L", f, pa, g, pb))


# ---------------------------------------------------------------- overlap tolerance and chords
def test_overlap_tolerance():
    cfg = PrepConfig(chord_tol=0.03, overlap_tol=0.05)
    notes = [note(0, 0.0, 0.54, C4), note(1, 0.5, 1.0, D4),  # 40 ms overlap -> sequential
             note(2, 1.0, 1.8, E4), note(3, 1.5, 2.0, F4)]  # 300 ms -> held
    ev = piece_events([notes], cfg)
    assert [e.pitches for e in ev] == [(C4,), (D4,), (E4,), (F4,)]
    assert ev[0].offset == pytest.approx(0.5) and ev[0].hold_until[0] == -np.inf
    assert ev[2].hold_until[0] == pytest.approx(1.8) and ev[2].offset == pytest.approx(1.5)


def test_chord_grouping():
    cfg = PrepConfig(chord_tol=0.03)
    notes = [note(0, 0.0, 1.0, C4), note(1, 0.01, 1.0, E4), note(2, 0.02, 1.0, G4),
             note(3, 1.0, 2.0, C5), note(4, 1.05, 2.0, E4)]
    ev = piece_events([notes], cfg)
    assert [e.size for e in ev] == [3, 1, 1]
    assert ev[0].pitches == (C4, E4, G4)


def test_merged_stream_ties_left_first():
    cfg = PrepConfig()
    notes = [note(0, 0.0, 0.5, C5, "R"), note(1, 0.0, 0.5, C4, "L", 1), note(2, 0.5, 1.0, D4, "L", 2)]
    ev = piece_events([notes], cfg)
    assert [e.hand for e in ev] == ["L", "R", "L"]


def test_labels_keep_every_annotator():
    a1 = [note(0, 0.0, 0.5, C4, "R", 1), note(1, 0.5, 1.0, D4, "R", 2)]
    a2 = [note(0, 0.0, 0.5, C4, "R", 1), note(1, 0.5, 1.0, D4, "L", 3)]
    ev = piece_events([a1, a2], PrepConfig())
    assert ev[1].labels == [(("R", 2),), (("L", 3),)]


# ---------------------------------------------------------------- finger sets
def test_finger_set_table():
    assert N_SETS == 31
    assert [len(SETS_OF_SIZE[k]) for k in range(1, 6)] == [5, 10, 10, 5, 1]
    assert all(len(FINGER_SETS[i]) == k for k in range(1, 6) for i in SETS_OF_SIZE[k])
    assert sizes_mask(3).sum() == 10


def test_finger_set_pitch_order():
    s = SET_INDEX[(1, 3, 5)]
    # pitches C-E-G in ascending order
    assert fingers_of(s, "R") == (1, 3, 5)  # C=1, E=3, G=5
    assert fingers_of(s, "L") == (5, 3, 1)  # C=5, E=3, G=1
    assert set_of((5, 3, 1)) == s and set_of((1, 1)) == -1


def test_chord_reach_mask():
    m = Parncutt12()
    assert m.chord_reach_ok(m.hand_event("R", [C4, E4, G4], [1, 3, 5]))
    assert not m.chord_reach_ok(m.hand_event("R", [C4, C5 + 7], [1, 2]))  # 19 semitones with 1-2
    assert m.chord_reach_ok(m.hand_event("L", [C4, E4, G4], [5, 3, 1]))
    assert not m.chord_reach_ok(m.hand_event("L", [C4, C5 + 7], [2, 1]))


# ---------------------------------------------------------------- per-hand history and charges
@pytest.mark.parametrize("name", ["parncutt12", "balliauw15"])
def test_per_hand_history(name):
    """In an L-R-L stream, the left hand's t-1 is its own previous event, not the right hand's."""
    m = make_model(name)
    stream = [("L", (48,), (1,)), ("R", (84,), (5,)), ("L", (50,), (2,))]
    left_alone = m.sequence("L", [((48,), (1,)), ((50,), (2,))])
    right_alone = m.sequence("R", [((84,), (5,))])
    assert np.allclose(merged_cost(m, stream), left_alone + right_alone)


@pytest.mark.parametrize("name", ["parncutt12", "balliauw15"])
def test_retroactive_equals_immediate(name):
    m = make_model(name)
    rng = random.Random(3)
    for _ in range(30):
        s = random_stream(rng, 30)
        # make some black-key thumbs and fives likely
        s += [((61,), (1,)), ((62,), (2,)), ((66,), (5,)), ((64,), (3,))]
        assert np.isclose(m.sequence("R", s).sum(), m.sequence("R", s, deferred=True).sum())


def test_thumb_position():
    assert thumb_position("R", [G4], [5]) == G4 - 7
    assert thumb_position("L", [C4], [5]) == C4 + 7
    assert thumb_position("R", [C4, E4, G4], [1, 3, 5]) == C4


# ---------------------------------------------------------------- Part A regression
def test_part_a_regression_v0():
    from fingerly.paper.pieces import load_piece
    from fingerly.paper.rules import breakdown

    v0 = load_piece("V0")
    stream = [((p,), (f,)) for p, f in zip(v0.pitches, v0.labels[0])]
    assert Parncutt12().sequence("R", stream).sum() == 18
    assert np.allclose(Parncutt12().sequence("R", stream), breakdown(v0.pitches, v0.labels[0]))


def test_part_a_regression_random_monophonic():
    from fingerly.paper.rules import difficulty

    rng = random.Random(5)
    m = Parncutt12()
    for _ in range(50):
        ps = [rng.randint(55, 80) for _ in range(25)]
        fs = [rng.randint(1, 5) for _ in ps]
        assert m.sequence("R", [((p,), (f,)) for p, f in zip(ps, fs)]).sum() == difficulty(ps, fs)


# ================================================================ phase 2: env, masks, search
from fingerly.twohand.actions import N_SETS as _NS  # noqa: E402
from fingerly.twohand.env import EnvConfig, Episode, SegInfo, SegmentPool  # noqa: E402
from fingerly.twohand.prepare import Segment  # noqa: E402
from fingerly.twohand.search import beam_search, dp_segment, enumerate_segment, search_segment  # noqa: E402


def make_segment(notes, n_ann=1):
    ev = piece_events([notes] * n_ann, PrepConfig())
    return Segment("t", 0, ev)


def two_hand_notes(rng, n=10):
    """Random two-hand material: mostly single notes, some chords, alternating hands."""
    notes, t, i = [], 0.0, 0
    for _ in range(n):
        hand = rng.choice("LR")
        base = 48 if hand == "L" else 67
        size = rng.choice([1, 1, 1, 2])
        ps = sorted({base + rng.randint(-5, 7) for _ in range(size)})
        for p in ps:
            notes.append(note(i, t, t + 0.2, p, hand, rng.randint(1, 5)))
            i += 1
        t += rng.choice([0.25, 0.5])
    return notes


def test_mask_only_chord_size():
    notes = [note(0, 0, 0.5, C4), note(1, 0, 0.5, E4), note(2, 0, 0.5, G4), note(3, 0.5, 1.0, A4)]
    info = SegInfo(make_segment(notes), EnvConfig(mode="given"), Parncutt12(), bonus=False)
    ep = Episode(EnvConfig(mode="given"))
    ep.reset(info)
    m = ep.mask()
    assert m.any() and all(len(FINGER_SETS[s]) == 3 for s in np.flatnonzero(m))


def test_occupied_finger_masked():
    # C4 held for 1 s under D4, E4: whatever finger holds C4 is not available for D4
    notes = [note(0, 0.0, 1.0, C4), note(1, 0.25, 0.5, D4), note(2, 0.5, 0.75, E4)]
    cfg = EnvConfig(mode="given")
    info = SegInfo(make_segment(notes), cfg, Parncutt12(), bonus=False)
    ep = Episode(cfg)
    ep.reset(info)
    ep.step(SET_INDEX[(1,)])  # thumb on C4, held
    assert not ep.mask()[SET_INDEX[(1,)]]


def test_assign_simultaneous_events_take_different_hands():
    notes = [note(0, 0.0, 0.5, C4, "L"), note(1, 0.0, 0.5, C5, "R"), note(2, 0.5, 1.0, D4, "L")]
    cfg = EnvConfig(mode="assign")
    info = SegInfo(make_segment(notes), cfg, Parncutt12(), bonus=False)
    ep = Episode(cfg)
    ep.reset(info)
    ep.step(SET_INDEX[(2,)] + _NS)  # first event to the right hand
    m = ep.mask()
    assert not m[_NS:].any() and m[:_NS].any()  # the simultaneous one must take the left hand


def test_assign_chords_keep_annotated_hand():
    notes = [note(0, 0.0, 0.5, C4, "L"), note(1, 0.0, 0.5, E4, "L"), note(2, 0.5, 1.0, D4, "R")]
    cfg = EnvConfig(mode="assign")
    info = SegInfo(make_segment(notes), cfg, Parncutt12(), bonus=False)
    ep = Episode(cfg)
    ep.reset(info)
    m = ep.mask()
    assert m[:_NS].any() and not m[_NS:].any()


@pytest.mark.parametrize("seed", range(4))
def test_decomposition_dp_equals_unpruned_search(seed):
    """2b without inter-hand terms: the per-hand DPs equal the full two-hand search."""
    seg = make_segment(two_hand_notes(random.Random(seed), 10))
    cfg = EnvConfig(mode="given", occupied=False)
    info = SegInfo(seg, cfg, Parncutt12(), bonus=False)
    assert dp_segment(info).cost == pytest.approx(beam_search(info, cfg, None).cost)


@pytest.mark.parametrize("mode", ["given", "assign"])
def test_search_equals_enumeration(mode):
    for seed in range(3):
        seg = make_segment(two_hand_notes(random.Random(10 + seed), 4))
        cfg = EnvConfig(mode=mode, cross=True, occupied=False)
        info = SegInfo(seg, cfg, Parncutt12(), bonus=False)
        assert beam_search(info, cfg, None).cost == pytest.approx(enumerate_segment(info, cfg))


def test_beam_matches_unpruned_on_10_events():
    agree = 0
    for seed in range(10):
        seg = make_segment(two_hand_notes(random.Random(100 + seed), 10))
        cfg = EnvConfig(mode="assign", occupied=False)
        info = SegInfo(seg, cfg, Parncutt12(), bonus=False)
        agree += beam_search(info, cfg, 10_000).cost == pytest.approx(beam_search(info, cfg, None).cost)
    assert agree >= 10 * 0.99


@pytest.mark.parametrize("mode", ["given", "assign"])
def test_m0_replays_at_the_same_cost(mode):
    seg = make_segment(two_hand_notes(random.Random(7), 12))
    cfg = EnvConfig(mode=mode, occupied=False)
    info = SegInfo(seg, cfg, Parncutt12(), bonus=False)
    r = search_segment(info, cfg, 10_000)
    ep = Episode(cfg)
    ep.reset(info)
    for h, s in r.fingering:
        ep.step(s + (_NS if mode == "assign" and h == "R" else 0))
    assert sum(x.rules + x.extra for x in ep.records) == pytest.approx(r.cost)


def test_hand_mismatch_is_a_miss():
    from fingerly.metrics import piece_match, rollout

    notes = [note(0, 0.0, 0.5, C4, "L", 1), note(1, 0.5, 1.0, D4, "L", 2)]
    seg = make_segment(notes)
    cfg = EnvConfig(mode="assign")
    pool = SegmentPool([seg], cfg, bonus=False)
    # always the right hand, with the annotator's fingers 1 then 2
    acts = iter([SET_INDEX[(1,)] + _NS, SET_INDEX[(2,)] + _NS])
    eps = rollout(pool, lambda o, m: [next(acts)])
    assert piece_match(eps)["t"]["Mgen"] == 0.0
    acts = iter([SET_INDEX[(1,)], SET_INDEX[(2,)]])
    eps = rollout(pool, lambda o, m: [next(acts)])
    assert piece_match(eps)["t"]["Mgen"] == 1.0
