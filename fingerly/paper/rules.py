"""The paper's reward: 12 difficulty rules after Parncutt et al. (1997), right hand (README §A.3).

Distance convention: for a finger pair (i, j) with i < j, d = pitch(finger j) - pitch(finger i)
in semitones (negative when the fingers are crossed). Rules 1-3 use consecutive notes
(t-1, t); rules 4-5 use notes (t-2, t) with fingers (a_{t-2}, a_t).

Every rule is charged at the note that completes it, so the difficulty of a fingering is
D = sum_t sum_k d_k(t), and the reward of step t is -sum_k d_k(t). The rules depend on the
fingers (a_{t-2}, a_{t-1}, a_t) and on fixed pitches (including the next note's), which is what
makes the exact DP in dp.py possible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..pig import is_black

N_RULES = 12
RULE_NAMES = (
    "stretch",
    "small-span",
    "large-span",
    "position-change-count",
    "position-change-size",
    "weak-finger",
    "three-four-five",
    "three-to-four",
    "four-on-black",
    "thumb-on-black",
    "five-on-black",
    "thumb-passing",
)

# (MinPrac, MinComf, MinRel, MaxRel, MaxComf, MaxPrac), right hand, semitones.
# Table I of the paper, as printed.
TABLE1_PAPER: Dict[Tuple[int, int], Tuple[int, ...]] = {
    (1, 2): (-5, -3, 1, 5, 8, 10),
    (1, 3): (-4, -2, 3, 7, 10, 12),
    (1, 4): (-3, -1, 5, 9, 12, 14),
    (1, 5): (-1, 1, 7, 10, 13, 15),
    (2, 3): (1, 1, 1, 2, 3, 5),
    (2, 4): (1, 1, 3, 4, 5, 7),
    (2, 5): (2, 2, 5, 6, 8, 10),
    (3, 4): (1, 1, 1, 2, 2, 4),
    (3, 5): (1, 1, 3, 4, 5, 7),
    (4, 5): (1, 1, 1, 4, 3, 5),  # MaxRel 4 > MaxComf 3: likely a typo (README §A.7, item 3)
}
# The same table with row 4-5 as in fingerly/cost.py (Parncutt's MaxRel = 2 for 4-5).
TABLE1_PARNCUTT: Dict[Tuple[int, int], Tuple[int, ...]] = {**TABLE1_PAPER, (4, 5): (1, 1, 1, 2, 3, 5)}
TABLES = {"paper": TABLE1_PAPER, "parncutt": TABLE1_PARNCUTT}
SAME_FINGER_MODES = ("zero-span", "free")


@dataclass(frozen=True)
class RuleConfig:
    table1: str = "paper"  # "paper" | "parncutt"
    same_finger: str = "zero-span"  # (i, i) pairs: "zero-span" = all six limits 0; "free" = no cost

    def __post_init__(self):
        if self.table1 not in TABLES:
            raise ValueError(f"table1 must be one of {list(TABLES)}")
        if self.same_finger not in SAME_FINGER_MODES:
            raise ValueError(f"same_finger must be one of {SAME_FINGER_MODES}")


def pair_limits(i: int, j: int, cfg: RuleConfig) -> Optional[Tuple[int, ...]]:
    """Limits for an ordered pair i <= j, or None if the pair is not charged at all."""
    if i == j:
        return (0, 0, 0, 0, 0, 0) if cfg.same_finger == "zero-span" else None
    return TABLES[cfg.table1][(i, j)]


def pair_distance(f_a: int, p_a: int, f_b: int, p_b: int) -> Tuple[int, int, int]:
    """(i, j, d) for two fingered notes: i <= j and d = pitch(j) - pitch(i).
    For the same finger, d is the move from the earlier note (a) to the later one (b)."""
    if f_a < f_b:
        return f_a, f_b, p_b - p_a
    if f_a > f_b:
        return f_b, f_a, p_a - p_b
    return f_a, f_b, p_b - p_a


def note_rules(
    p2: Optional[int],
    p1: Optional[int],
    p0: int,
    pn: Optional[int],
    a2: Optional[int],
    a1: Optional[int],
    a0: int,
    cfg: RuleConfig,
) -> np.ndarray:
    """Points of each of the 12 rules charged at note t.

    p2, p1, p0, pn = pitches of notes t-2, t-1, t, t+1 (None if they do not exist);
    a2, a1, a0 = fingers of notes t-2, t-1, t."""
    r = np.zeros(N_RULES)
    black0 = is_black(p0)

    # ---- consecutive pair (t-1, t): rules 1-3, 8, 9, 12 ----
    if p1 is not None:
        i, j, d = pair_distance(a1, p1, a0, p0)
        lim = pair_limits(i, j, cfg)
        if lim is not None:
            _, min_comf, min_rel, max_rel, max_comf, _ = lim
            r[0] = 2 * max(0, min_comf - d) + 2 * max(0, d - max_comf)
            k = 1 if 1 in (i, j) else 2
            r[1] = k * max(0, min_rel - d)
            r[2] = k * max(0, d - max_rel)
        if a1 == 3 and a0 == 4:
            r[7] = 1
        if {a1, a0} == {3, 4}:
            p3, p4 = (p1, p0) if a1 == 3 else (p0, p1)
            if not is_black(p3) and is_black(p4):
                r[8] = 1
        if (a1 == 1) != (a0 == 1):  # exactly one of the two is the thumb
            thumb_p, other_p = (p1, p0) if a1 == 1 else (p0, p1)
            if other_p < thumb_p:  # crossed: the thumb plays above another finger
                if is_black(thumb_p) == is_black(other_p):
                    r[11] = 1
                elif not is_black(other_p) and is_black(thumb_p):
                    r[11] = 3

    # ---- triple (t-2, t-1, t): rules 4, 5, 7 ----
    if p2 is not None:
        i, j, d = pair_distance(a2, p2, a0, p0)
        lim = pair_limits(i, j, cfg)
        if lim is not None:
            min_prac, min_comf, _, _, max_comf, max_prac = lim
            if d < min_comf or d > max_comf:
                r[3] = 1
                between = min(p2, p0) < p1 < max(p2, p0)
                if a1 == 1 and between and (d < min_prac or d > max_prac):
                    r[3] += 1
            r[4] = max(0, min_comf - d) + max(0, d - max_comf)
        if sorted((a2, a1, a0)) == [3, 4, 5]:
            r[6] = 1

    # ---- single note t: rules 6, 10, 11 ----
    if a0 in (4, 5):
        r[5] = 1
    if a0 == 1 and black0:
        r[9] = 1
        if pn is not None and not is_black(pn):
            r[9] += 2
        if p1 is not None and not is_black(p1):
            r[9] += 2
    if a0 == 5 and black0:
        if p1 is not None and not is_black(p1):
            r[10] += 2
        if pn is not None and not is_black(pn):
            r[10] += 2
    return r


def _at(seq: Sequence, k: int):
    return seq[k] if 0 <= k < len(seq) else None


def step_breakdown(pitches: Sequence[int], fingers: Sequence[int], t: int, cfg: RuleConfig) -> np.ndarray:
    """Rule points charged at note t (fingers needs entries up to t)."""
    return note_rules(
        _at(pitches, t - 2), _at(pitches, t - 1), pitches[t], _at(pitches, t + 1),
        fingers[t - 2] if t >= 2 else None, fingers[t - 1] if t >= 1 else None, fingers[t],
        cfg,
    )


def breakdown(pitches: Sequence[int], fingers: Sequence[int], cfg: RuleConfig = RuleConfig()) -> np.ndarray:
    """Per-rule totals over the whole sequence (length-12 array)."""
    assert len(pitches) == len(fingers)
    total = np.zeros(N_RULES)
    for t in range(len(pitches)):
        total += step_breakdown(pitches, fingers, t, cfg)
    return total


def difficulty(pitches: Sequence[int], fingers: Sequence[int], cfg: RuleConfig = RuleConfig()) -> float:
    """D: total difficulty of a fingering."""
    return float(breakdown(pitches, fingers, cfg).sum())


def level(pitches: Sequence[int], fingers: Sequence[int], cfg: RuleConfig = RuleConfig()) -> float:
    """L = D / N, the paper's difficulty level."""
    return difficulty(pitches, fingers, cfg) / len(pitches)


def format_breakdown(b: np.ndarray) -> str:
    return ", ".join(f"{n}={v:g}" for n, v in zip(RULE_NAMES, b) if v)


def per_note(pitches: Sequence[int], fingers: Sequence[int], cfg: RuleConfig = RuleConfig()) -> List[np.ndarray]:
    return [step_breakdown(pitches, fingers, t, cfg) for t in range(len(pitches))]
