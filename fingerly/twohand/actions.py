"""Finger-set actions (README §C.5.1).

An action is a set of C distinct fingers for an event of C notes: C(5, C) sets per size,
5 + 10 + 10 + 5 + 1 = 31 in all, in a fixed global table. The set is given to the chord's notes in
pitch order: from the lowest note upward the right hand uses its fingers in ascending order and the
left hand in descending order. In mode 2c the action also picks the hand: index = hand * 31 + set,
with hand 0 = left, 1 = right.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import numpy as np

FINGER_SETS: Tuple[Tuple[int, ...], ...] = tuple(
    s for k in range(1, 6) for s in combinations(range(1, 6), k)
)
N_SETS = len(FINGER_SETS)  # 31
SET_INDEX: Dict[Tuple[int, ...], int] = {s: i for i, s in enumerate(FINGER_SETS)}
SETS_OF_SIZE: Dict[int, np.ndarray] = {
    k: np.array([i for i, s in enumerate(FINGER_SETS) if len(s) == k]) for k in range(1, 6)
}
SET_BITS = np.zeros((N_SETS, 5), np.float32)
for _i, _s in enumerate(FINGER_SETS):
    SET_BITS[_i, [f - 1 for f in _s]] = 1.0

HANDS = ("L", "R")


def fingers_of(set_index: int, hand: str) -> Tuple[int, ...]:
    """Fingers for the event's notes in ascending pitch order."""
    s = FINGER_SETS[set_index]
    return s if hand == "R" else tuple(reversed(s))


def set_of(fingers: Sequence[int]) -> int:
    """Global index of the set used by a fingering (any order); -1 if fingers repeat or are missing."""
    if any(f < 1 or f > 5 for f in fingers) or len(set(fingers)) != len(fingers):
        return -1
    return SET_INDEX[tuple(sorted(fingers))]


def canonical(fingers: Sequence[int], hand: str) -> bool:
    """True if the fingers (ascending pitch order) follow the pitch-order assignment."""
    f = list(fingers)
    return f == sorted(f) if hand == "R" else f == sorted(f, reverse=True)


def sizes_mask(size: int) -> np.ndarray:
    m = np.zeros(N_SETS, bool)
    m[SETS_OF_SIZE[size]] = True
    return m


def local_sets(size: int) -> List[int]:
    return list(SETS_OF_SIZE[size])
