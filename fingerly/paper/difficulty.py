"""The 12-rule fingering difficulty of Iman & Ahn (2024), section III-B, after Parncutt et al. (1997).

Right hand, distances in semitones, span limits from the paper's Table I (the same numbers as
`fingerly.cost.PARNCUTT_SPANS`). For an ordered finger pair (a, b), a < b, the "distance" is
the pitch under finger b minus the pitch under finger a; a negative distance means the fingers
cross (only thumb pairs allow that).

The rules look at a note, the note before it and the note before that. `step_difficulty`
returns D_t, the points charged when note t is played (pair rules on (t-1, t), triple rules on
(t-2, t-1, t), single-note rules on t, which may look at the colour of the next note). The sum
over a piece is D(s, a) of eq. (3); L(s, a) = D / N is eq. (6).

Where the paper's wording leaves a choice open we follow Parncutt et al. (1997). Each choice is
marked "interpretation" below. They were checked against the three scores the paper gives
for Beyer Op. 101 No. 12 (Table II, fingerings read off Fig. 5): the optimal fingering scores
18 (L = 0.6207, exact), A2C 80 (L = 2.7586, exact) and PG 234 against the paper's 235
(L = 8.069 vs 8.1034; most likely one finger misread from the figure).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from ..cost import PARNCUTT_SPANS

BLACK = {1, 3, 6, 8, 10}

RULES = (
    "stretch", "small_span", "large_span", "position_change_count", "position_change_size",
    "weak_finger", "three_four_five", "three_to_four", "four_on_black", "thumb_on_black",
    "five_on_black", "thumb_passing",
)


def is_black(p: int) -> bool:
    return p % 12 in BLACK


SAME_FINGER = (0, 0, 0, 0, 0, 0)


def _limits(a: int, b: int):
    """(MinPrac, MinComf, MinRel, MaxRel, MaxComf, MaxPrac). Table I has no row for one finger
    on two keys; interpretation: every limit is 0, so each semitone moved breaks the comfortable
    and relaxed spans (this reproduces the paper's A2C score on Beyer exactly)."""
    return SAME_FINGER if a == b else PARNCUTT_SPANS[(a, b)]


def _pair(p1: int, f1: int, p2: int, f2: int):
    """((a, b), distance) with a <= b."""
    if f1 <= f2:
        return (f1, f2), p2 - p1
    return (f2, f1), p1 - p2


def step_rules(
    pitches: Sequence[int], fingers: Sequence[int], t: int, next_pitch: Optional[int] = None
) -> Dict[str, float]:
    """Points per rule charged at note t, given pitches/fingers of notes 0..t (later entries are
    ignored). `next_pitch` is the pitch of note t+1 (only its colour is used); None at the end."""
    d = {r: 0.0 for r in RULES}
    p, f = pitches[t], fingers[t]
    prev = (pitches[t - 1], fingers[t - 1]) if t >= 1 else None
    prev2 = (pitches[t - 2], fingers[t - 2]) if t >= 2 else None

    # --- single-note rules -----------------------------------------------------------
    if f in (4, 5):  # rule 6
        d["weak_finger"] += 1
    if f == 1 and is_black(p):  # rule 10
        d["thumb_on_black"] += 1
        if next_pitch is not None and not is_black(next_pitch):
            d["thumb_on_black"] += 2
        if prev is not None and not is_black(prev[0]):
            d["thumb_on_black"] += 2
    if f == 5 and is_black(p):  # rule 11
        if prev is not None and not is_black(prev[0]):
            d["five_on_black"] += 2
        if next_pitch is not None and not is_black(next_pitch):
            d["five_on_black"] += 2

    # --- pair rules (t-1, t) -----------------------------------------------------------
    if prev is not None:
        pp, pf = prev
        if not (pf == f and pp == p):  # a repeated key with the same finger costs nothing
            (a, b), dist = _pair(pp, pf, p, f)
            min_prac, min_comf, min_rel, max_rel, max_comf, max_prac = _limits(a, b)
            thumb = a == 1
            # rule 1
            d["stretch"] += 2 * (max(0, min_comf - dist) + max(0, dist - max_comf))
            # rules 2 and 3
            w = 1 if thumb else 2
            d["small_span"] += w * max(0, min_rel - dist)
            d["large_span"] += w * max(0, dist - max_rel)
            # rule 12: thumb passing = a thumb pair played crossed (distance < 0)
            if thumb and a != b and dist < 0:
                lo_p, lo_f = (pp, pf) if pp < p else (p, f)
                hi_p, hi_f = (p, f) if pp < p else (pp, pf)
                if is_black(lo_p) == is_black(hi_p):
                    d["thumb_passing"] += 1
                elif not is_black(lo_p) and lo_f != 1 and is_black(hi_p) and hi_f == 1:
                    d["thumb_passing"] += 3
        # rule 8: finger 4 right after finger 3
        if pf == 3 and f == 4:
            d["three_to_four"] += 1
        # rule 9: 3 and 4 consecutive, any order, 3 on white and 4 on black
        if {pf, f} == {3, 4}:
            p3, p4 = (pp, p) if pf == 3 else (p, pp)
            if not is_black(p3) and is_black(p4):
                d["four_on_black"] += 1

    # --- triple rules (t-2, t-1, t) ------------------------------------------------------
    if prev2 is not None:
        (p1, f1), (p2, f2), (p3, f3) = prev2, prev, (p, f)
        # rule 7: fingers 3, 4, 5 consecutively in any order (interpretation: a permutation)
        if sorted((f1, f2, f3)) == [3, 4, 5]:
            d["three_four_five"] += 1
        # interpretation: no position rule when notes 1 and 3 share a finger (counting them
        # overshoots the paper's A2C score on Beyer by 10 points)
        if f1 != f3:
            (a, b), dist = _pair(p1, f1, p3, f3)
            min_prac, min_comf, _, _, max_comf, max_prac = _limits(a, b)
            if dist < min_comf or dist > max_comf:
                # rule 4: a change of position; a full change scores one point more
                d["position_change_count"] += 1
                between = min(p1, p3) < p2 < max(p1, p3)
                if f2 == 1 and between and (dist < min_prac or dist > max_prac):
                    d["position_change_count"] += 1
                # rule 5
                d["position_change_size"] += max(0, min_comf - dist) + max(0, dist - max_comf)
    return d


def step_difficulty(pitches, fingers, t, next_pitch=None) -> float:
    """D_t(s_t, a_t) of eq. (2)."""
    return float(sum(step_rules(pitches, fingers, t, next_pitch).values()))


def piece_rules(pitches: Sequence[int], fingers: Sequence[int]) -> Dict[str, float]:
    """Points per rule over a whole piece."""
    tot = {r: 0.0 for r in RULES}
    for t in range(len(pitches)):
        nxt = pitches[t + 1] if t + 1 < len(pitches) else None
        for k, v in step_rules(pitches, fingers, t, nxt).items():
            tot[k] += v
    return tot


def piece_difficulty(pitches: Sequence[int], fingers: Sequence[int]) -> float:
    """D(s, a) of eq. (3)."""
    return float(sum(piece_rules(pitches, fingers).values()))


def difficulty_level(pitches: Sequence[int], fingers: Sequence[int]) -> float:
    """L(s, a) of eq. (6): average difficulty per note."""
    return piece_difficulty(pitches, fingers) / max(1, len(pitches))


def per_step(pitches: Sequence[int], fingers: Sequence[int]) -> List[float]:
    return [
        step_difficulty(pitches, fingers, t, pitches[t + 1] if t + 1 < len(pitches) else None)
        for t in range(len(pitches))
    ]
