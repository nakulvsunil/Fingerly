"""State features (README §C.7).

    current event   register; black?; chord size / 5; span / 12; onset gap to the previous event;
                    overlaps a held note?                                                       6
    look-ahead K    per event: interval to each hand's last key / 12 (2); black?; chord size / 5;
                    onset gap from the current event; exists?; its given hand (+1 R, -1 L; 0 in 2c) 7K
    each hand       played yet?; interval from its last key / 12; last finger set (5 bits); finger
                    set before that (5 bits); interval from the key before that / 12; time since
                    it last played; holding a key?; thumb position relative to the current event 16 x 2
    mode            hand flag of the current event (+1 R, -1 L in 2a/2b; 0 in 2c)              1

"Key" is the event's thumb-side note: lowest for the right hand, highest for the left. Left-hand
intervals are negated (mirrored) so one network fingers both hands. Times are in seconds, clipped
at 2 and divided by 2. Intervals are clipped to +-3 octaves. With K = 4 there are 67 numbers;
`hand_pos=False` drops the two thumb-position features (ablation).
"""

from __future__ import annotations

import numpy as np

CLIP_IV = 3.0
CLIP_T = 2.0


def obs_dim(lookahead: int = 4, hand_pos: bool = True) -> int:
    return 6 + 7 * lookahead + 2 * (16 if hand_pos else 15) + 1


def key_of(pitches, hand: str) -> int:
    return pitches[0] if hand == "R" else pitches[-1]


def iv(a: float, b: float, hand: str) -> float:
    """Mirrored interval from a to b, in octaves, clipped."""
    d = (b - a) if hand == "R" else (a - b)
    return float(np.clip(d / 12.0, -CLIP_IV, CLIP_IV))


def tclip(x: float) -> float:
    return min(max(x, 0.0), CLIP_T) / CLIP_T
