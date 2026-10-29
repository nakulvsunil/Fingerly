"""Two-hand reward rules (README §C.6): cost models `parncutt12` and `balliauw15`.

Both models work in **mirrored space**: for the left hand every key value is negated (key colour
is unchanged), so all rules are written once, for the right hand. With the pitch-order finger
assignment (actions.py), a left-hand chord in mirrored space is an ascending right-hand chord with
ascending fingers. `balliauw15_swap_pair` implements the README's other left-hand reading
(X_L(f, g) = X_R(g, f) on unmirrored distances) for the mirror test.

An event in a hand's history is a `HandEvent`: key values (ascending, mirrored), key colours and
fingers aligned with them. Every rule is charged at the event that completes it; the per-hand
history (t-1, t-2) is that hand's own previous events (§C.6.1). Rules that look at the *next*
note of the hand can be charged immediately (2a, 2b) or when that next event arrives (2c,
§C.6.4); `next_terms` holds exactly those parts, so both ways give the same total.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..paper.rules import TABLES, RULE_NAMES as PARNCUTT_NAMES
from ..pig import is_black

# --------------------------------------------------------------------------------------
# Hand events
# --------------------------------------------------------------------------------------
_POS = (0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12)  # pitch class -> position, 14 per octave


def position(midi: int) -> int:
    """Key position with an invisible black key between E-F and B-C (14 per octave, §C.6.2)."""
    return 14 * (midi // 12) + _POS[midi % 12]


@dataclass(frozen=True)
class HandEvent:
    vals: Tuple[int, ...]  # mirrored key values, ascending
    blacks: Tuple[bool, ...]
    fingers: Tuple[int, ...]
    pitches: Tuple[int, ...]  # real MIDI pitches, same order as vals

    @property
    def single(self) -> bool:
        return len(self.vals) == 1


def make_hand_event(hand: str, pitches: Sequence[int], fingers: Sequence[int], unit: str) -> HandEvent:
    """pitches ascending (real), fingers aligned with them. unit: 'semitone' or 'position'."""
    val = (lambda p: p) if unit == "semitone" else position
    sign = 1 if hand == "R" else -1
    items = sorted(((sign * val(p), is_black(p), f, p) for p, f in zip(pitches, fingers)), key=lambda x: x[0])
    return HandEvent(tuple(x[0] for x in items), tuple(x[1] for x in items), tuple(x[2] for x in items),
                     tuple(x[3] for x in items))


def _between_pairs(ev1: HandEvent, ev0: HandEvent, mode: str) -> List[Tuple[int, int]]:
    """Note pairs (i in ev1, j in ev0) scored by rules 1-3 between events (§C.6.1)."""
    if mode == "all":
        return [(i, j) for i in range(len(ev1.vals)) for j in range(len(ev0.vals))]
    if mode == "extremes":  # lowest to lowest and highest to highest
        return sorted({(0, 0), (len(ev1.vals) - 1, len(ev0.vals) - 1)})
    raise ValueError(f"chord_horizontal must be 'all' or 'extremes', got {mode!r}")


# --------------------------------------------------------------------------------------
# Cost model base
# --------------------------------------------------------------------------------------
class CostModel:
    name = ""
    unit = "semitone"
    rule_names: Tuple[str, ...] = ()

    def __init__(self, chord_horizontal: str = "all", span_scale: float = 1.0):
        self.chord_horizontal = chord_horizontal
        self.span_scale = span_scale

    @property
    def n_rules(self) -> int:
        return len(self.rule_names)

    def hand_event(self, hand: str, pitches: Sequence[int], fingers: Sequence[int]) -> HandEvent:
        return make_hand_event(hand, pitches, fingers, self.unit)

    # to be provided by subclasses
    def event_rules(self, ev2: Optional[HandEvent], ev1: Optional[HandEvent], ev0: HandEvent) -> np.ndarray:
        """Points charged at event t, without the parts that depend on the next event."""
        raise NotImplementedError

    def next_terms(self, ev: HandEvent, nxt: Optional[HandEvent]) -> np.ndarray:
        """Parts of event `ev`'s rules that depend on the hand's next event `nxt`."""
        raise NotImplementedError

    def prac_ok(self, fa: int, va: int, fb: int, vb: int) -> bool:
        """Is the move from (finger fa, value va) to (finger fb, value vb) within the practical
        limits? Same finger always passes (a hand shift; its cost comes from the rules)."""
        raise NotImplementedError

    def chord_reach_ok(self, ev: HandEvent) -> bool:
        """Every adjacent pair of the chord within [MinPrac, MaxPrac] (§C.5.3)."""
        return all(self.prac_ok(ev.fingers[k], ev.vals[k], ev.fingers[k + 1], ev.vals[k + 1])
                   for k in range(len(ev.vals) - 1))

    def transition_ok(self, ev1: Optional[HandEvent], ev0: HandEvent) -> bool:
        """Single note to single note within the practical limits (chord transitions are not
        masked; their cost comes from the rules)."""
        if ev1 is None or not (ev1.single and ev0.single):
            return True
        return self.prac_ok(ev1.fingers[0], ev1.vals[0], ev0.fingers[0], ev0.vals[0])

    def outside_comf(self, ev1: HandEvent, ev0: HandEvent) -> bool:
        """Large-stretch indicator for the metrics: any scored pair outside [MinComf, MaxComf]."""
        raise NotImplementedError

    def sequence(self, hand: str, events: Sequence[Tuple[Sequence[int], Sequence[int]]],
                 deferred: bool = False) -> np.ndarray:
        """Per-rule totals of one hand's stream [(pitches, fingers), ...]. deferred=True charges the
        next-note parts when the next event arrives (§C.6.4)."""
        hev = [self.hand_event(hand, p, f) for p, f in events]
        tot = np.zeros(self.n_rules)
        for t, ev in enumerate(hev):
            ev1 = hev[t - 1] if t >= 1 else None
            ev2 = hev[t - 2] if t >= 2 else None
            tot += self.event_rules(ev2, ev1, ev)
            if deferred:
                if ev1 is not None:
                    tot += self.next_terms(ev1, ev)
            else:
                tot += self.next_terms(ev, hev[t + 1] if t + 1 < len(hev) else None)
        return tot


# --------------------------------------------------------------------------------------
# parncutt12: Part A's 12 rules, extended to the left hand and chords (§C.6.1)
# --------------------------------------------------------------------------------------
class Parncutt12(CostModel):
    name = "parncutt12"
    unit = "semitone"
    rule_names = PARNCUTT_NAMES

    def __init__(self, chord_horizontal: str = "all", span_scale: float = 1.0, table1: str = "paper"):
        super().__init__(chord_horizontal, span_scale)
        self.table = {k: tuple(v * span_scale for v in lim) for k, lim in TABLES[table1].items()}

    def _limits(self, i: int, j: int):
        return (0, 0, 0, 0, 0, 0) if i == j else self.table[(i, j)]

    @staticmethod
    def _pair(fa: int, va: int, fb: int, vb: int):
        """(i, j, d): i <= j, d = value(j) - value(i); same finger: the move from a to b."""
        if fa < fb:
            return fa, fb, vb - va
        if fa > fb:
            return fb, fa, va - vb
        return fa, fb, vb - va

    def _span_rules(self, fa, va, fb, vb) -> np.ndarray:
        i, j, d = self._pair(fa, va, fb, vb)
        _, min_comf, min_rel, max_rel, max_comf, _ = self._limits(i, j)
        k = 1 if 1 in (i, j) else 2
        return np.array([2 * max(0, min_comf - d) + 2 * max(0, d - max_comf),
                         k * max(0, min_rel - d), k * max(0, d - max_rel)], float)

    def prac_ok(self, fa, va, fb, vb) -> bool:
        if fa == fb:
            return True
        i, j, d = self._pair(fa, va, fb, vb)
        lim = self._limits(i, j)
        return lim[0] <= d <= lim[5]

    def outside_comf(self, ev1, ev0) -> bool:
        for i, j in _between_pairs(ev1, ev0, self.chord_horizontal):
            fa, va, fb, vb = ev1.fingers[i], ev1.vals[i], ev0.fingers[j], ev0.vals[j]
            if fa == fb:
                continue
            a, b, d = self._pair(fa, va, fb, vb)
            lim = self._limits(a, b)
            if d < lim[1] or d > lim[4]:
                return True
        return False

    def event_rules(self, ev2, ev1, ev0) -> np.ndarray:
        r = np.zeros(12)
        # within the chord: rules 1-3 on adjacent pairs, double score
        for k in range(len(ev0.vals) - 1):
            r[0:3] += 2 * self._span_rules(ev0.fingers[k], ev0.vals[k], ev0.fingers[k + 1], ev0.vals[k + 1])
        if ev1 is not None:
            for i, j in _between_pairs(ev1, ev0, self.chord_horizontal):
                r[0:3] += self._span_rules(ev1.fingers[i], ev1.vals[i], ev0.fingers[j], ev0.vals[j])
        if not ev0.single:
            return r
        a0, p0, b0 = ev0.fingers[0], ev0.vals[0], ev0.blacks[0]
        # single note: rules 6, 10, 11 (previous-note parts)
        if a0 in (4, 5):
            r[5] = 1
        prev_white = ev1 is not None and ev1.single and not ev1.blacks[0]
        if a0 == 1 and b0:
            r[9] = 1 + (2 if prev_white else 0)
        if a0 == 5 and b0 and prev_white:
            r[10] = 2
        if ev1 is None or not ev1.single:
            return r
        a1, p1, b1 = ev1.fingers[0], ev1.vals[0], ev1.blacks[0]
        # consecutive single notes: rules 8, 9, 12
        if a1 == 3 and a0 == 4:
            r[7] = 1
        if {a1, a0} == {3, 4}:
            b3, b4 = (b1, b0) if a1 == 3 else (b0, b1)
            if not b3 and b4:
                r[8] = 1
        if (a1 == 1) != (a0 == 1):
            thumb_p, thumb_b, other_p, other_b = (p1, b1, p0, b0) if a1 == 1 else (p0, b0, p1, b1)
            if other_p < thumb_p:
                if thumb_b == other_b:
                    r[11] = 1
                elif not other_b and thumb_b:
                    r[11] = 3
        if ev2 is None or not ev2.single:
            return r
        a2, p2 = ev2.fingers[0], ev2.vals[0]
        # triples of single notes: rules 4, 5, 7
        i, j, d = self._pair(a2, p2, a0, p0)
        min_prac, min_comf, _, _, max_comf, max_prac = self._limits(i, j)
        if d < min_comf or d > max_comf:
            r[3] = 1
            if a1 == 1 and min(p2, p0) < p1 < max(p2, p0) and (d < min_prac or d > max_prac):
                r[3] += 1
        r[4] = max(0, min_comf - d) + max(0, d - max_comf)
        if sorted((a2, a1, a0)) == [3, 4, 5]:
            r[6] = 1
        return r

    def next_terms(self, ev, nxt) -> np.ndarray:
        r = np.zeros(12)
        if nxt is None or not ev.single or not nxt.single or nxt.blacks[0] or not ev.blacks[0]:
            return r
        if ev.fingers[0] == 1:
            r[9] = 2
        elif ev.fingers[0] == 5:
            r[10] = 2
        return r


# --------------------------------------------------------------------------------------
# balliauw15: the 2025 paper's rules (§C.6.2)
# --------------------------------------------------------------------------------------
_MAX_PRAC_R = np.array([[0, 11, 15, 16, 18], [10, 0, 7, 8, 12], [8, -1, 0, 4, 8], [6, -1, -1, 0, 6], [2, -2, -1, -1, 0]])
_MAX_COMF_R = np.array([[0, 9, 13, 14, 16], [8, 0, 5, 6, 10], [6, -1, 0, 2, 6], [4, -1, -1, 0, 4], [0, -2, -1, -1, 0]])
_MAX_REL_R = np.array([[0, 6, 9, 11, 12], [-1, 0, 2, 4, 6], [-3, -1, 0, 2, 4], [-5, -3, -1, 0, 2], [-7, -5, -3, -1, 0]])

BALLIAUW_NAMES = (
    "1 comfort", "2 relaxed", "3 practical", "4 same key new finger", "5 finger 4",
    "6 within chord", "7 position change", "8 position change size", "9 same finger skip",
    "10 three-four", "11 three black four white", "12 white non-thumb to black thumb",
    "13 non-thumb to thumb", "14 thumb on black", "15 five on black",
)


def balliauw_limits(hand: str = "R", swap: bool = False):
    """(MinPrac, MinComf, MinRel, MaxRel, MaxComf, MaxPrac) as 5x5 arrays indexed [from-1, to-1].
    Min*(f, g) = -Max*(g, f). swap=True gives the left-hand tables X_L(f, g) = X_R(g, f)."""
    mx = [_MAX_PRAC_R, _MAX_COMF_R, _MAX_REL_R]
    mn = [-m.T for m in mx]
    tabs = (mn[0], mn[1], mn[2], mx[2], mx[1], mx[0])
    if swap:
        tabs = tuple(t.T for t in tabs)
    return tabs


class Balliauw15(CostModel):
    name = "balliauw15"
    unit = "position"
    rule_names = BALLIAUW_NAMES

    def __init__(self, chord_horizontal: str = "all", span_scale: float = 1.0, rule7b: str = "middle-thumb"):
        super().__init__(chord_horizontal, span_scale)
        self.lim = tuple(t * span_scale for t in balliauw_limits("R"))
        self.rule7b = rule7b  # "middle-thumb" | "any-thumb"

    def _out(self, f, g, d, lo, hi) -> float:
        return max(0.0, self.lim[lo][f - 1, g - 1] - d) + max(0.0, d - self.lim[hi][f - 1, g - 1])

    def _span_rules(self, f, va, g, vb) -> np.ndarray:
        """Rules 1-3 for a move from (f, va) to (g, vb)."""
        d = vb - va
        return np.array([2 * self._out(f, g, d, 1, 4), self._out(f, g, d, 2, 3), 10 * self._out(f, g, d, 0, 5)])

    def prac_ok(self, fa, va, fb, vb) -> bool:
        if fa == fb:
            return True
        return self._out(fa, fb, vb - va, 0, 5) == 0

    def outside_comf(self, ev1, ev0) -> bool:
        for i, j in _between_pairs(ev1, ev0, self.chord_horizontal):
            f, g = ev1.fingers[i], ev0.fingers[j]
            if f != g and self._out(f, g, ev0.vals[j] - ev1.vals[i], 1, 4) > 0:
                return True
        return False

    def event_rules(self, ev2, ev1, ev0) -> np.ndarray:
        r = np.zeros(15)
        # rule 6: rules 1-3 within the chord at double score (lower note -> higher note)
        for k in range(len(ev0.vals) - 1):
            r[5] += 2 * self._span_rules(ev0.fingers[k], ev0.vals[k], ev0.fingers[k + 1], ev0.vals[k + 1]).sum()
        if ev1 is not None:
            for i, j in _between_pairs(ev1, ev0, self.chord_horizontal):
                r[0:3] += self._span_rules(ev1.fingers[i], ev1.vals[i], ev0.fingers[j], ev0.vals[j])
            # rule 4: the same key replayed with a different finger
            prev = dict(zip(ev1.vals, ev1.fingers))
            r[3] = sum(1 for v, f in zip(ev0.vals, ev0.fingers) if v in prev and prev[v] != f)
        if not ev0.single:
            return r
        a0, p0, b0 = ev0.fingers[0], ev0.vals[0], ev0.blacks[0]
        if a0 == 4:
            r[4] = 1
        prev_white = ev1 is not None and ev1.single and not ev1.blacks[0]
        if a0 == 1 and b0:
            r[13] = 0.5 + (1 if prev_white else 0)
        if a0 == 5 and b0 and prev_white:
            r[14] = 1
        if ev1 is None or not ev1.single:
            return r
        a1, p1, b1 = ev1.fingers[0], ev1.vals[0], ev1.blacks[0]
        if {a1, a0} == {3, 4}:
            r[9] = 1
            b3, b4 = (b1, b0) if a1 == 3 else (b0, b1)
            if b3 and not b4:
                r[10] = 1
        if a1 != 1 and not b1 and a0 == 1 and b0:
            r[11] = 2
        if a1 != 1 and a0 == 1:
            r[12] = 1
        if ev2 is None or not ev2.single:
            return r
        a2, p2 = ev2.fingers[0], ev2.vals[0]
        d = p0 - p2
        out_comf = self._out(a2, a0, d, 1, 4)
        if out_comf > 0:
            r[6] = 1
        thumb = a1 == 1 if self.rule7b == "middle-thumb" else 1 in (a2, a1, a0)
        if p2 < p1 < p0 and thumb and self._out(a2, a0, d, 0, 5) > 0:
            r[6] += 1
        if p2 == p0 and a2 != a0:
            r[6] += 1
        r[7] = out_comf
        if p2 != p0 and a2 == a0 and p2 < p1 < p0:
            r[8] = 1
        return r

    def next_terms(self, ev, nxt) -> np.ndarray:
        r = np.zeros(15)
        if nxt is None or not ev.single or not nxt.single or nxt.blacks[0] or not ev.blacks[0]:
            return r
        if ev.fingers[0] == 1:
            r[13] = 1
        elif ev.fingers[0] == 5:
            r[14] = 1
        return r


def balliauw15_swap_pair(hand: str, f: int, pa: int, g: int, pb: int) -> np.ndarray:
    """Rules 1-3 for a move from (f, pitch pa) to (g, pitch pb) on *unmirrored* positions, using
    the left-hand reading X_L(f, g) = X_R(g, f) for hand 'L'. Used by the mirror test."""
    lim = balliauw_limits(hand, swap=(hand == "L"))
    d = position(pb) - position(pa)

    def out(lo, hi):
        return max(0.0, lim[lo][f - 1, g - 1] - d) + max(0.0, d - lim[hi][f - 1, g - 1])

    return np.array([2 * out(1, 4), out(2, 3), 10 * out(0, 5)])


def merged_cost(model: CostModel, stream: Sequence[Tuple[str, Sequence[int], Sequence[int]]],
                deferred: bool = False) -> np.ndarray:
    """Per-rule totals of a merged two-hand stream [(hand, pitches, fingers), ...]: each hand's
    history is its own previous events (§C.6.1)."""
    tot = np.zeros(model.n_rules)
    for hand in ("L", "R"):
        tot += model.sequence(hand, [(p, f) for h, p, f in stream if h == hand], deferred)
    return tot


MODELS = {"parncutt12": Parncutt12, "balliauw15": Balliauw15}


def make_model(name: str, chord_horizontal: str = "all", span_scale: float = 1.0, **kw) -> CostModel:
    return MODELS[name](chord_horizontal=chord_horizontal, span_scale=span_scale, **kw)


# --------------------------------------------------------------------------------------
# Inter-hand and timing terms (§C.6.3)
# --------------------------------------------------------------------------------------
FIVE_FINGER = (0, 2, 4, 5, 7)  # semitones from the thumb for fingers 1-5


def thumb_position(hand: str, pitches: Sequence[int], fingers: Sequence[int]) -> float:
    """Thumb key from an event: the thumb-side note (right: lowest, left: highest) and its finger."""
    if hand == "R":
        k = int(np.argmin(pitches))
        return pitches[k] - FIVE_FINGER[fingers[k] - 1]
    k = int(np.argmax(pitches))
    return pitches[k] + FIVE_FINGER[fingers[k] - 1]


def time_leap(shift: float, dt: float, v_max: float, w_time: float = 1.0) -> float:
    return w_time * max(0.0, abs(shift) - v_max * dt)


def crossing(thumb_l: Optional[float], thumb_r: Optional[float], margin: float = 2.0, w_cross: float = 1.0) -> float:
    if thumb_l is None or thumb_r is None:
        return 0.0
    return w_cross * max(0.0, thumb_l - thumb_r - margin)
