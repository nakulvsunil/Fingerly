"""Synthetic, label-free exercises: scales, arpeggios, broken thirds and random melodies.

They give the agent extra practice on the patterns that dominate real fingering decisions
(thumb-under, finger-over, position shifts) without needing expert labels.
"""

from __future__ import annotations

import random
from typing import List

from .pig import Passage

MAJOR = [0, 2, 4, 5, 7, 9, 11]
MINOR_HARMONIC = [0, 2, 3, 5, 7, 8, 11]


def _scale_pitches(tonic: int, steps: List[int], octaves: int) -> List[int]:
    out = []
    for o in range(octaves):
        out += [tonic + 12 * o + s for s in steps]
    out.append(tonic + 12 * octaves)
    return out


def _make(pitches: List[int], hand: str, name: str, dur: float) -> Passage:
    onsets = [i * dur for i in range(len(pitches))]
    return Passage(
        piece=f"synth:{name}",
        hand=hand,
        pitches=pitches,
        onsets=onsets,
        offsets=[o + dur for o in onsets],
        keys=[(int(o * 1000), p) for o, p in zip(onsets, pitches)],
    )


def scale(rng: random.Random, hand: str) -> Passage:
    base = 60 if hand == "R" else 48
    tonic = base + rng.randrange(12) - (12 if hand == "L" and rng.random() < 0.5 else 0)
    steps = rng.choice([MAJOR, MINOR_HARMONIC])
    up = _scale_pitches(tonic, steps, rng.choice([1, 2]))
    pitches = up + up[-2::-1] if rng.random() < 0.6 else (up[::-1] if rng.random() < 0.5 else up)
    return _make(pitches, hand, "scale", rng.choice([0.125, 0.25, 0.5]))


def arpeggio(rng: random.Random, hand: str) -> Passage:
    base = 60 if hand == "R" else 48
    root = base + rng.randrange(12)
    chord = rng.choice([[0, 4, 7], [0, 3, 7], [0, 4, 7, 10], [0, 3, 6]])
    octs = rng.choice([1, 2])
    up = [root + 12 * o + c for o in range(octs) for c in chord] + [root + 12 * octs]
    pitches = up + up[-2::-1] if rng.random() < 0.6 else up
    return _make(pitches, hand, "arpeggio", rng.choice([0.125, 0.25]))


def broken_thirds(rng: random.Random, hand: str) -> Passage:
    base = 60 if hand == "R" else 48
    tonic = base + rng.randrange(12)
    sc = _scale_pitches(tonic, MAJOR, 2)
    pitches = []
    for i in range(min(8, len(sc) - 2)):
        pitches += [sc[i], sc[i + 2]]
    return _make(pitches, hand, "thirds", 0.25)


def melody(rng: random.Random, hand: str, length: int = 24) -> Passage:
    """A random walk mostly by steps, with some leaps and repeated notes."""
    base = 67 if hand == "R" else 50
    p = base + rng.randrange(-5, 6)
    pitches = [p]
    for _ in range(length - 1):
        r = rng.random()
        if r < 0.55:
            d = rng.choice([-2, -1, 1, 2])
        elif r < 0.80:
            d = rng.choice([-4, -3, 3, 4, -5, 5])
        elif r < 0.90:
            d = rng.choice([-12, -7, 7, 12, -9, 9])
        else:
            d = 0
        p = max(base - 15, min(base + 15, p + d))
        pitches.append(p)
    return _make(pitches, hand, "melody", rng.choice([0.25, 0.5]))


def synthetic_passages(n: int, hand: str = "R", seed: int = 0) -> List[Passage]:
    rng = random.Random(seed)
    makers = [scale, scale, arpeggio, broken_thirds, melody, melody, melody]
    return [rng.choice(makers)(rng, hand) for _ in range(n)]


# --------------------------------------------------------------------------------------
# Two-hand exercises (README §C.4.7): label-free, each event's hand set by the generator
# --------------------------------------------------------------------------------------
def _tn(notes, i, t, dur, pitch, hand):
    from .pig import Note

    notes.append(Note(i, round(t, 4), round(t + dur, 4), int(pitch), 0 if hand == "R" else 1, hand, None))


def contrary_scales(rng: random.Random):
    tonic = rng.randrange(12)
    steps = rng.choice([MAJOR, MINOR_HARMONIC])
    octs = rng.choice([1, 2])
    deg = [12 * o + s for o in range(octs) for s in steps] + [12 * octs]
    dur = rng.choice([0.125, 0.25])
    notes, t = [], 0.0
    for k, d in enumerate(deg + deg[-2::-1]):
        _tn(notes, 2 * k, t, dur, 60 + tonic + d, "R")
        _tn(notes, 2 * k + 1, t, dur, 48 + tonic - d + 12 * octs, "L")
        t += dur
    return notes


def alberti(rng: random.Random):
    """Left-hand Alberti bass or broken chords under a right-hand melody."""
    root = 48 + rng.randrange(12)
    chords = [[0, 4, 7], [5, 9, 12], [7, 11, 14], [0, 4, 7]]
    notes, t, i = [], 0.0, 0
    mel = 72 + rng.randrange(-3, 4)
    for c in chords * rng.choice([1, 2]):
        pattern = [c[0], c[2], c[1], c[2]] if rng.random() < 0.6 else [c[0], c[1], c[2], c[1]]
        for k, iv in enumerate(pattern):
            _tn(notes, i, t + 0.25 * k, 0.25, root + iv, "L")
            i += 1
        for k in range(2):
            mel = max(65, min(84, mel + rng.choice([-2, -1, 1, 2, 3, -3])))
            _tn(notes, i, t + 0.5 * k, 0.5, mel, "R")
            i += 1
        t += 1.0
    return notes


def split_arpeggio(rng: random.Random):
    """An arpeggio over more than 2 octaves, left hand low half, right hand high half."""
    root = 36 + rng.randrange(12)
    chord = rng.choice([[0, 4, 7], [0, 3, 7], [0, 4, 7, 10]])
    up = [root + 12 * o + c for o in range(4) for c in chord] + [root + 48]
    seq = up + up[-2::-1] if rng.random() < 0.6 else up
    mid = root + 24
    notes = []
    for k, p in enumerate(seq):
        _tn(notes, k, 0.125 * k, 0.125, p, "L" if p < mid else "R")
    return notes


def passed_melody(rng: random.Random, length: int = 32):
    """A melody passed between the hands: low phrases left, high phrases right."""
    p = 60 + rng.randrange(-4, 5)
    notes = []
    for k in range(length):
        p = max(43, min(84, p + rng.choice([-2, -1, 1, 2, -4, 4, -7, 7])))
        _tn(notes, k, 0.25 * k, 0.25, p, "L" if p < 60 else "R")
    return notes


def block_chords(rng: random.Random, length: int = 16):
    notes, i = [], 0
    for k in range(length):
        hand = rng.choice("LR")
        base = (48 if hand == "L" else 67) + rng.randrange(-5, 6)
        size = rng.choice([2, 3, 3, 4])
        ivs = sorted(rng.sample([0, 3, 4, 5, 7, 8, 9, 12], size - 1))
        for iv in [0] + ivs:
            _tn(notes, i, 0.5 * k, 0.45, base + iv, hand)
            i += 1
        other = "R" if hand == "L" else "L"
        _tn(notes, i, 0.5 * k + 0.25, 0.2, (72 if other == "R" else 45) + rng.randrange(-4, 5), other)
        i += 1
    return notes


TWOHAND_MAKERS = [contrary_scales, alberti, split_arpeggio, passed_melody, block_chords]


def twohand_segments(n: int, seed: int = 0, max_len: int = 0, prep=None):
    """n synthetic two-hand segments (README §C.4.7). No labels: no imitation bonus, no metrics.
    max_len > 0 keeps only the first max_len events (2c curriculum: short, easy segments)."""
    from .twohand.prepare import PrepConfig, Segment, piece_events

    prep = prep or PrepConfig()
    rng = random.Random(seed)
    out = []
    for k in range(n):
        maker = rng.choice(TWOHAND_MAKERS)
        ev = piece_events([maker(rng)], prep)
        for e in ev:
            e.labels = []
        if max_len:
            ev = ev[:max_len]
        out.append(Segment(f"synth:{maker.__name__}", k, ev, synthetic=True))
    return out
