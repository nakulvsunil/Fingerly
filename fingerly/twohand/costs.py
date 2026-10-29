"""Precomputed cost tables (README §C.10.1).

For one hand's events in a segment (modes 2a and 2b, where every event's hand is given) the cost of
event i depends only on the finger sets chosen at the hand's events i-2, i-1 and i. So each event
gets a table over those sets, computed once:

    C[s2, s1, s0]   reward cost = rule points + time-leap + forced-shift cost
    D[s2, s1, s0]   rule points only (the difficulty)
    allowed[s1, s0] static mask: chord reach, single-note transition within the practical limits,
                    and the out-of-reach shift (§C.5.3)

Indices are local: s0 runs over the sets of event i's chord size (`sets`), s1 over event i-1's,
s2 over event i-2's. A missing previous event has one dummy index. When rules can't see event i-2
(it doesn't exist, or one of the three events is a chord) the first axis has length 1 and
broadcasts. Next-note rule parts are charged immediately (the next event of the hand is known).
Occupied fingers depend on earlier choices beyond two events, so they are applied by the
environment, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .actions import SETS_OF_SIZE, fingers_of
from .prepare import Event, Segment
from .rules import CostModel, HandEvent, thumb_position, time_leap

SHIFT_COST = 2.0


@dataclass
class EventTable:
    event: int  # index in the segment
    sets: np.ndarray  # global set ids for s0
    C: np.ndarray  # (n2 or 1, n1, n0) float32
    D: np.ndarray
    allowed: np.ndarray  # (n1, n0) bool
    shift: np.ndarray  # (n1,) bool: no reachable set from s1, so the hand jumps
    reach: np.ndarray  # (n0,) bool
    infeasible: np.ndarray  # (n1, n0) bool: transition outside MinPrac/MaxPrac or forced fallback


@dataclass
class HandTables:
    hand: str
    events: List[int]  # segment indices of this hand's events, in order
    tables: List[EventTable] = field(default_factory=list)


def _hand_event_cache(model: CostModel, hand: str, ev: Event) -> List[HandEvent]:
    return [model.hand_event(hand, ev.pitches, fingers_of(int(s), hand)) for s in SETS_OF_SIZE[ev.size]]


def build_hand_tables(seg: Segment, hand: str, model: CostModel, v_max: float, w_time: float = 1.0,
                      events: Optional[Sequence[int]] = None) -> HandTables:
    idx = list(events) if events is not None else seg.hand_events(hand)
    hevs = [_hand_event_cache(model, hand, seg.events[i]) for i in idx]
    thumbs = [np.array([thumb_position(hand, seg.events[i].pitches, fingers_of(int(s), hand))
                        for s in SETS_OF_SIZE[seg.events[i].size]]) for i in idx]
    out = HandTables(hand, idx)
    for k, i in enumerate(idx):
        ev0 = seg.events[i]
        H0 = hevs[k]
        n0 = len(H0)
        H1 = hevs[k - 1] if k >= 1 else [None]
        H2 = hevs[k - 2] if k >= 2 else [None]
        nxt = hevs[k + 1] if k + 1 < len(idx) else None
        triple = k >= 2 and ev0.single and seg.events[idx[k - 1]].single and seg.events[idx[k - 2]].single
        n1 = len(H1)
        n2 = len(H2) if triple else 1
        D = np.zeros((n2, n1, n0), np.float32)
        for b, h1 in enumerate(H1):
            for c, h0 in enumerate(H0):
                if triple:
                    for a, h2 in enumerate(H2):
                        D[a, b, c] = model.event_rules(h2, h1, h0).sum()
                else:
                    D[0, b, c] = model.event_rules(None, h1, h0).sum()
        # immediate next-note terms: the next event's colours do not depend on its fingers
        if nxt is not None:
            D += np.array([model.next_terms(h, nxt[0]).sum() for h in H0], np.float32)[None, None, :]

        reach = np.array([model.chord_reach_ok(h) for h in H0])
        trans = np.ones((n1, n0), bool)
        if k >= 1:
            for b, h1 in enumerate(H1):
                for c, h0 in enumerate(H0):
                    trans[b, c] = model.transition_ok(h1, h0)
        base = reach[None, :] & trans
        shift = ~base.any(1)
        allowed = base.copy()
        allowed[shift] = reach[None, :].repeat(shift.sum(), 0)
        unreachable = ~allowed.any(1)
        allowed[unreachable] = True  # no set reaches the chord: all sets of the size (logged as unreachable)
        infeasible = ~base & allowed

        X = np.zeros((1, n1, n0), np.float32)
        if k >= 1:
            dt = ev0.onset - seg.events[idx[k - 1]].onset
            th1, th0 = thumbs[k - 1], thumbs[k]
            for b in range(n1):
                for c in range(n0):
                    X[0, b, c] = time_leap(th0[c] - th1[b], dt, v_max, w_time)
        X[0, shift, :] += SHIFT_COST
        out.tables.append(EventTable(i, SETS_OF_SIZE[ev0.size].copy(), (D + X).astype(np.float32),
                                     D.astype(np.float32), allowed, shift, reach, infeasible))
    return out


@dataclass
class SegmentTables:
    """Both hands' tables for one segment (modes 2a/2b), plus the event -> (hand, position) map."""

    hands: Dict[str, HandTables]
    pos: List[tuple]  # per segment event: (hand, k) with k the index in that hand's tables


def build_segment_tables(seg: Segment, model: CostModel, v_max: float, w_time: float = 1.0,
                         hands: Sequence[str] = ("L", "R")) -> SegmentTables:
    ht = {h: build_hand_tables(seg, h, model, v_max, w_time) for h in hands}
    pos: List[tuple] = [None] * len(seg)
    for h, t in ht.items():
        for k, i in enumerate(t.events):
            pos[i] = (h, k)
    return SegmentTables(ht, pos)


def lookup(t: EventTable, s2: Optional[int], s1: Optional[int], s0: int, which: str = "C") -> float:
    """Cost of local sets (s2, s1, s0); None for a missing previous event."""
    arr = t.C if which == "C" else t.D
    a = 0 if arr.shape[0] == 1 or s2 is None else s2
    b = 0 if s1 is None else s1
    return float(arr[a, b, s0])
