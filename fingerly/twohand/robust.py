"""Robustness perturbations (README §C.11.7). Evaluation only: the policy stays frozen.

Tempo, transposition and onset jitter are applied to the raw PIG notes of the test pieces, then
events, chords, held notes and segments are rebuilt, so jitter stresses chord grouping and the
overlap tolerance as the README asks. Each note gets the same change for every annotator, so the
annotations still line up. Hand size is an environment setting (`span_scale`); eras group the clean
results by composer; the stress set is synthetic (large leaps, fast hand alternation).
"""

from __future__ import annotations

import dataclasses
import hashlib
import random
from typing import List, Sequence

from ..pig import Note
from .prepare import PrepConfig, Segment, piece_events, segment_stream


def _key_rng(seed: int, key) -> random.Random:
    h = hashlib.sha1(f"{seed}/{key}".encode()).hexdigest()
    return random.Random(int(h[:12], 16))


def perturb_notes(notes: Sequence[Note], kind: str, level: float, seed: int = 0) -> List[Note]:
    out = []
    for n in notes:
        on, off, p = n.onset, n.offset, n.pitch
        if kind == "tempo":
            on, off = on * level, off * level
        elif kind == "transpose":
            p = int(min(108, max(21, p + int(level))))
        elif kind == "jitter":
            d = _key_rng(seed, n.key).gauss(0.0, level)
            on, off = max(0.0, on + d), max(0.0, off + d)
        else:
            raise ValueError(kind)
        out.append(dataclasses.replace(n, onset=on, offset=off, pitch=p))
    out.sort(key=lambda n: (n.onset, n.pitch, n.note_id))
    return out


def perturbed_segments(pig, piece_ids: Sequence[str], kind: str, level: float, prep: PrepConfig,
                       seed: int = 0, min_len: int = 2) -> List[Segment]:
    """Test segments rebuilt from perturbed notes (annotations stay matched by note identity)."""
    segs = []
    for k in piece_ids:
        # each change is a function of the note's (onset, pitch) key, which the annotators share,
        # so the annotators' perturbed notes still match each other
        anns = [perturb_notes(a, kind, level, seed) for a in pig[k]]
        segs += segment_stream(k, piece_events(anns, prep), prep, min_len)
    return segs


def stress_segments(n: int = 40, seed: int = 0, prep: PrepConfig = PrepConfig()) -> List[Segment]:
    """Synthetic stress set: (a) large leaps of 2-3 octaves within a hand, fast; (b) single notes
    alternating quickly between the hands. No labels."""
    rng = random.Random(seed)
    segs = []
    for k in range(n):
        notes, t, i = [], 0.0, 0
        if k % 2 == 0:  # large leaps
            for _ in range(32):
                hand = rng.choice("LR")
                lo, hi = (28, 60) if hand == "L" else (60, 96)
                p = rng.choice([lo + rng.randrange(6), hi - rng.randrange(6)])
                notes.append(Note(i, t, t + 0.12, p, 0 if hand == "R" else 1, hand, None))
                i += 1
                t += rng.choice([0.12, 0.18, 0.25])
            name = "stress:leaps"
        else:  # fast alternation between the hands
            pL, pR = 52, 67
            for j in range(48):
                hand = "L" if j % 2 == 0 else "R"
                if hand == "L":
                    pL = max(36, min(60, pL + rng.choice([-2, -1, 1, 2, 5, -5])))
                    p = pL
                else:
                    pR = max(60, min(88, pR + rng.choice([-2, -1, 1, 2, 5, -5])))
                    p = pR
                notes.append(Note(i, t, t + 0.08, p, 0 if hand == "R" else 1, hand, None))
                i += 1
                t += 0.08
            name = "stress:alternation"
        ev = piece_events([notes], prep)
        for e in ev:
            e.labels = []
        segs.append(Segment(name, k, ev, synthetic=True))
    return segs


def parse_spec(specs: Sequence[str]) -> List[tuple]:
    """['tempo=0.75,1.5', 'transpose=-5,5', 'jitter=0.01', 'span=0.85,1.15', 'era', 'stress']
    -> [(kind, level), ...]"""
    out = []
    for s in specs:
        if "=" in s:
            k, vals = s.split("=", 1)
            out += [(k, float(v)) for v in vals.split(",") if v]
        else:
            out.append((s, None))
    return out
