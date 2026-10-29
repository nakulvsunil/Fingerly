"""Pianists' fingerings as hand streams, human hand speed (v_max) and shared run context."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..pig import LEFT, RIGHT, Note, load_pig
from .prepare import Event, PrepConfig, TwoHandSplit, make_twohand_split, piece_events
from .rules import thumb_position


def annotator_streams(notes: Sequence[Note], cfg: PrepConfig) -> Dict[str, List[Tuple[Event, Tuple[int, ...]]]]:
    """One annotator's own hands and fingers: {hand: [(event, fingers aligned with event.pitches)]}.
    Events are built from this annotator's hand assignment; unfingered notes are dropped."""
    evs = piece_events([[n for n in notes if n.finger]], cfg)
    out: Dict[str, list] = {LEFT: [], RIGHT: []}
    for e in evs:
        out[e.hand].append((e, tuple(f for (_, f) in e.labels[0])))
    return out


def hand_speeds(pig, piece_ids, cfg: PrepConfig):
    """|thumb shift| (semitones) and onset gap (s) between consecutive events of each hand, over
    every annotator of the given pieces. Events with a repeated finger are skipped."""
    shifts, dts = [], []
    for k in piece_ids:
        for ann in pig[k]:
            st = annotator_streams(ann, cfg)
            for hand in (LEFT, RIGHT):
                prev = None
                for e, fs in st[hand]:
                    if len(set(fs)) != len(fs):
                        prev = None
                        continue
                    th = thumb_position(hand, e.pitches, fs)
                    if prev is not None:
                        shifts.append(abs(th - prev[0]))
                        dts.append(e.onset - prev[1])
                    prev = (th, e.onset)
    return np.array(shifts), np.array(dts)


def human_vmax(pig, train_ids, cfg: PrepConfig, pct: float = 95.0) -> float:
    """95th percentile of human hand speed |shift| / dt on the training pieces (§C.6.3)."""
    s, dt = hand_speeds(pig, train_ids, cfg)
    ok = dt > 0
    return float(np.percentile(s[ok] / dt[ok], pct))


class Context:
    """PIG, the split and v_max, built once per process. v_max is cached in results/twohand."""

    def __init__(self, pig_dir: str, prep: PrepConfig = PrepConfig(), vmax_pct: float = 95.0,
                 cache: str = "results/twohand/vmax.json", verbose: bool = False):
        self.pig_dir = pig_dir
        self.prep = prep
        self.pig = load_pig(pig_dir)
        self.split: TwoHandSplit = make_twohand_split(self.pig, prep)
        key = f"{prep.chord_tol}/{prep.overlap_tol}/{vmax_pct}"
        path = Path(cache)
        store = json.loads(path.read_text()) if path.exists() else {}
        if key not in store:
            store[key] = human_vmax(self.pig, self.split.train_pieces, prep, vmax_pct)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(store, indent=1))
        self.v_max = float(store[key])
        if verbose:
            print(self.split.summary())
            print(f"v_max = {self.v_max:.2f} semitones/s ({vmax_pct}th percentile of training hand speed)")

    def segments(self, name: str):
        return {"train": self.split.train, "val": self.split.val, "test": self.split.test}[name]
