"""Scoring a two-hand fingering of a segment under any cost model."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .actions import fingers_of
from .prepare import Segment
from .rules import CostModel, merged_cost


def as_stream(seg: Segment, order: Sequence[int], fing: Sequence[Tuple[str, int]]):
    """[(hand, pitches, fingers)] for the visited events, from per-event (hand, global set)."""
    return [(h, seg.events[i].pitches, fingers_of(int(s), h)) for i, (h, s) in zip(order, fing)]


def rule_points(seg: Segment, order: Sequence[int], fing: Sequence[Tuple[str, int]], model: CostModel) -> float:
    """Total rule points (each hand's own history, next-note parts charged immediately)."""
    return float(merged_cost(model, as_stream(seg, order, fing)).sum())


def rule_breakdown(seg: Segment, order, fing, model: CostModel) -> np.ndarray:
    return merged_cost(model, as_stream(seg, order, fing))
