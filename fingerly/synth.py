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
