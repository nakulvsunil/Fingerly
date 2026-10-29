"""Finger the monophonic passages of a PIG file and export them for the pygame player
(visual/piano/fingering_player.py).

The JSON looks like

    {"piece": "001-1", "policy": "agent (best.pt)", "hand": "R", "total_cost": 41.0,
     "notes": [{"name": "E5", "pitch": 76, "onset": 0.0, "offset": 0.25, "hand": "R",
                "finger": 3, "human": 3, "passage": 0, "cost": 0.0, "breakdown": {}}, ...]}

`cost` is the effort of the note under the reward (plus the impossible penalty) and `breakdown`
names the rules that fired. `human` is annotator 1's finger, or null when the file has none.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from .cost import CostConfig, cost_breakdown, mirrored_interval
from .pig import Note, Passage, is_black, monophonic_runs, read_fingering_file

NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

Fingerer = Callable[[Passage], List[int]]  # passage -> fingers 1..5


def note_name(pitch: int) -> str:
    return f"{NAMES[pitch % 12]}{pitch // 12 - 1}"


def passages_of(notes: Sequence[Note], hand: str, min_len: int = 8):
    """(runs, passages) for the monophonic passages of one hand, as predict.py builds them."""
    runs = monophonic_runs([n for n in notes if n.hand == hand], min_len)
    passages = [Passage("input", hand, [n.pitch for n in r], [n.onset for n in r], [n.offset for n in r]) for r in runs]
    return runs, passages


def export_fingering(
    path: str | Path,
    fingerer: Fingerer,
    policy: str,
    hand: str = "R",
    min_len: int = 8,
    cfg: Optional[CostConfig] = None,
    passage: Optional[int] = None,
    max_gap: Optional[float] = 1.0,
) -> Dict:
    """Finger every passage of `hand` in a PIG file (or only passage number `passage`).

    Notes between passages (chords, the hand's polyphony) are not fingered and are left out, so
    playback is silent there; `max_gap` shortens such pauses to at most that many seconds.
    """
    cfg = cfg or CostConfig()
    notes = read_fingering_file(path)
    runs, passages = passages_of(notes, hand, min_len)
    if not passages:
        raise ValueError(f"no monophonic {hand}-hand passages of >= {min_len} notes in {path}")
    chosen = range(len(passages)) if passage is None else [passage]
    if passage is not None and not 0 <= passage < len(passages):
        raise ValueError(f"--passage must be in 0..{len(passages) - 1}, got {passage}")

    out: List[Dict] = []
    total, shift, last_end = 0.0, 0.0, None
    for k in chosen:
        run, p = runs[k], passages[k]
        fingers = fingerer(p)
        if last_end is not None and max_gap is not None:
            shift += max(0.0, p.onsets[0] - shift - last_end - max_gap)
        prev_f, prev_p = None, None
        for n, f in zip(run, fingers):
            iv = 0 if prev_p is None else mirrored_interval(prev_p, n.pitch, hand)
            br = cost_breakdown(prev_f, int(f), iv, is_black(n.pitch), cfg)
            cost = sum(br.values())
            total += cost
            out.append({
                "name": note_name(n.pitch), "pitch": n.pitch,
                "onset": round(n.onset - shift, 6), "offset": round(n.offset - shift, 6),
                "hand": hand, "finger": int(f), "human": n.finger, "passage": k,
                "cost": cost, "breakdown": br,
            })
            prev_f, prev_p = int(f), n.pitch
        last_end = run[-1].offset - shift

    return {"piece": Path(path).name.split("_")[0], "policy": policy, "hand": hand,
            "total_cost": total, "n_passages": len(passages), "notes": out}
