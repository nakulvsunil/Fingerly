"""Ergonomic effort cost for one hand (cost table of the Fingerly proposal, section 9).

Distances are in semitones. All rules are written for the RIGHT hand; the left hand is
handled by mirroring the keyboard (an interval of +3 for the left hand is fed in as -3), so
finger 1 of either hand is always on the "inner" side.

Span limits per finger pair come from Table 1 of Parncutt et al. (1997). For an ordered
pair (a, b) with a < b they bound the interval from finger a's key to finger b's key:

    MinPrac  MinComf  MinRel  MaxRel  MaxComf  MaxPrac

Only Prac (hard limit -> action mask) and Comf (soft limit -> per-semitone cost) are used here.
Check these numbers against the paper before you report them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

FINGERS = (1, 2, 3, 4, 5)

# (MinPrac, MinComf, MinRel, MaxRel, MaxComf, MaxPrac), right hand, semitones
PARNCUTT_SPANS: Dict[Tuple[int, int], Tuple[int, int, int, int, int, int]] = {
    (1, 2): (-5, -3, 1, 5, 8, 10),
    (1, 3): (-4, -2, 3, 7, 10, 12),
    (1, 4): (-3, -1, 5, 9, 12, 14),
    (1, 5): (-1, 1, 7, 10, 13, 15),
    (2, 3): (1, 1, 1, 2, 3, 5),
    (2, 4): (1, 1, 3, 4, 5, 7),
    (2, 5): (2, 2, 5, 6, 8, 10),
    (3, 4): (1, 1, 1, 2, 2, 4),
    (3, 5): (1, 1, 3, 4, 5, 7),
    (4, 5): (1, 1, 1, 2, 3, 5),
}


@dataclass
class CostConfig:
    """Weights of the cost table. Defaults follow the proposal exactly."""

    stretch_per_semitone: float = 1.0  # wider than comfortable
    cramp_per_semitone: float = 1.0  # closer than comfortable
    weak_finger: float = 1.0  # finger 4 or 5 used
    thumb_on_black: float = 1.0
    crossing: float = 1.0  # finger over thumb / thumb under
    same_finger_shift: float = 2.0  # same finger on a different key straight away (also any hand shift)
    hand_switch: float = 2.0  # unused with one hand; kept for stage 2
    impossible: float = 10.0  # wider than the maximum stretch
    span_scale: float = 1.0  # hand-size personalisation: scales the stretch extents (1.0 = Parncutt)
    spans: Dict[Tuple[int, int], Tuple[int, ...]] = field(default_factory=lambda: dict(PARNCUTT_SPANS))

    def limits(self, a: int, b: int) -> Tuple[float, float, float, float]:
        """(MinPrac, MinComf, MaxComf, MaxPrac) for ordered pair a < b, scaled by hand size."""
        min_prac, min_comf, _, _, max_comf, max_prac = self.spans[(a, b)]
        s = self.span_scale
        # A bigger hand reaches further: maxima grow, and negative minima (thumb crossings)
        # grow too. Positive minima (the natural spacing of adjacent fingers) stay as they are.
        lo = lambda v: v * s if v < 0 else v  # noqa: E731
        return lo(min_prac), lo(min_comf), max_comf * s, max_prac * s


@dataclass
class StepCost:
    effort: float  # C_effort for this note
    impossible: bool  # the move breaks a hard span limit (only reachable without the mask)
    crossing: bool = False
    shift: bool = False  # the hand had to move (same finger to new key, or out-of-reach jump)
    stretch: float = 0.0  # semitones beyond comfortable (wide or cramped)


def _rel(prev_f: int, f: int, interval: float) -> Tuple[int, int, float]:
    """Order the pair as (low finger, high finger) and give the interval from the low-numbered
    finger's key to the high-numbered finger's key."""
    if prev_f < f:
        return prev_f, f, interval
    return f, prev_f, -interval


def pair_feasible(prev_f: int, f: int, interval: float, cfg: CostConfig) -> bool:
    """Can finger f play a key `interval` semitones from the key under prev_f without shifting the hand?"""
    if f == prev_f or interval == 0:
        return True
    a, b, rel = _rel(prev_f, f, interval)
    min_prac, _, _, max_prac = cfg.limits(a, b)
    return min_prac <= rel <= max_prac


def is_out_of_reach(prev_f: int, interval: float, cfg: CostConfig) -> bool:
    """True when no *other* finger can reach the next key from prev_f: the hand must jump,
    and then any finger may land on the key."""
    if interval == 0:
        return False
    return not any(pair_feasible(prev_f, f, interval, cfg) for f in FINGERS if f != prev_f)


def valid_fingers(prev_f: Optional[int], interval: float, cfg: CostConfig) -> List[bool]:
    """Action mask for fingers 1..5 (index 0..4). Never all False."""
    if prev_f is None or is_out_of_reach(prev_f, interval, cfg):
        return [True] * 5
    return [pair_feasible(prev_f, f, interval, cfg) for f in FINGERS]


def note_cost(f: int, black: bool, cfg: CostConfig) -> float:
    c = 0.0
    if f in (4, 5):
        c += cfg.weak_finger
    if f == 1 and black:
        c += cfg.thumb_on_black
    return c


def step_cost(prev_f: Optional[int], f: int, interval: float, black: bool, cfg: CostConfig) -> StepCost:
    """Effort of playing the current note (black or white key) with finger f, when the same hand's
    previous note was played by prev_f and the current note is `interval` semitones above it
    (already mirrored for the left hand). prev_f=None means the first note of the hand."""
    effort = note_cost(f, black, cfg)
    if prev_f is None:
        return StepCost(effort=effort, impossible=False)

    if f == prev_f:
        if interval == 0:
            return StepCost(effort=effort, impossible=False)
        return StepCost(effort=effort + cfg.same_finger_shift, impossible=False, shift=True)

    if is_out_of_reach(prev_f, interval, cfg):
        return StepCost(effort=effort + cfg.same_finger_shift, impossible=False, shift=True)

    a, b, rel = _rel(prev_f, f, interval)
    min_prac, min_comf, max_comf, max_prac = cfg.limits(a, b)
    if interval != 0 and not (min_prac <= rel <= max_prac):
        return StepCost(effort=effort, impossible=True)

    wide = max(0.0, rel - max_comf)
    cramped = max(0.0, min_comf - rel)
    effort += cfg.stretch_per_semitone * wide + cfg.cramp_per_semitone * cramped
    crossing = rel < 0  # only thumb pairs allow rel < 0: thumb under / finger over thumb
    if crossing:
        effort += cfg.crossing
    return StepCost(effort=effort, impossible=False, crossing=crossing, stretch=wide + cramped)


def cost_breakdown(prev_f: Optional[int], f: int, interval: float, black: bool, cfg: CostConfig) -> Dict[str, float]:
    """The rules behind step_cost, by name, with their cost (only the rules that fired).
    The values add up to step_cost(...).effort, plus cfg.impossible for an impossible move."""
    out: Dict[str, float] = {}
    if f in (4, 5) and cfg.weak_finger:
        out["weak finger"] = cfg.weak_finger
    if f == 1 and black and cfg.thumb_on_black:
        out["thumb on black"] = cfg.thumb_on_black
    if prev_f is None or (f == prev_f and interval == 0):
        return out
    if f == prev_f:
        out["same finger shift"] = cfg.same_finger_shift
        return out
    if is_out_of_reach(prev_f, interval, cfg):
        out["hand shift"] = cfg.same_finger_shift
        return out
    a, b, rel = _rel(prev_f, f, interval)
    min_prac, min_comf, max_comf, max_prac = cfg.limits(a, b)
    if interval != 0 and not (min_prac <= rel <= max_prac):
        out["impossible"] = cfg.impossible
        return out
    if rel > max_comf:
        out["stretch"] = cfg.stretch_per_semitone * (rel - max_comf)
    if rel < min_comf:
        out["cramp"] = cfg.cramp_per_semitone * (min_comf - rel)
    if rel < 0:
        out["crossing"] = cfg.crossing
    return out


def mirrored_interval(prev_pitch: int, pitch: int, hand: str) -> int:
    """Semitone interval seen from the hand: left-hand intervals are mirrored."""
    d = pitch - prev_pitch
    return d if hand == "R" else -d


def sequence_cost(
    pitches: Sequence[int], fingers: Sequence[int], hand: str = "R", cfg: Optional[CostConfig] = None
) -> Tuple[float, int, List[StepCost]]:
    """Total effort, number of impossible moves, and per-note details of a full fingering."""
    cfg = cfg or CostConfig()
    total, impossible, details = 0.0, 0, []
    prev_f, prev_p = None, None
    for p, f in zip(pitches, fingers):
        iv = 0 if prev_p is None else mirrored_interval(prev_p, p, hand)
        sc = step_cost(prev_f, int(f), iv, p % 12 in {1, 3, 6, 8, 10}, cfg)
        total += sc.effort + (cfg.impossible if sc.impossible else 0.0)
        impossible += int(sc.impossible)
        details.append(sc)
        prev_f, prev_p = int(f), p
    return total, impossible, details
