"""Data pipeline for two hands (README §C.4).

PIG notes (annotator 1 gives notes, onsets, offsets and hands) are turned into **events**: a single
note, or a chord struck by one hand. Within each hand, legato overlaps up to `overlap_tol` are
made sequential and longer overlaps become held notes whose finger stays occupied. Events of both
hands are merged in onset order and cut into **segments** at long rests or at `max_len` events.

Every annotator's (hand, finger) for every note is kept on the event for the imitation bonus and
the metrics. A note an annotator did not finger, or that cannot be matched, gets (None, 0).
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..pig import LEFT, RIGHT, Note, is_black, load_pig

# The 30-piece subset of Iman & Ahn (2025), README §C.4.6.
TEST_IDS_2025 = (
    "002 017 025 031 033 034 038 042 052 053 066 067 069 074 077 081 085 092 094 "
    "101 102 107 110 112 114 119 123 136 141 148"
).split()

Label = Tuple[Optional[str], int]  # (hand, finger) given by one annotator to one note


@dataclass
class PrepConfig:
    chord_tol: float = 0.03
    overlap_tol: float = 0.05
    rest_gap: float = 1.5
    max_len: int = 256
    min_len: int = 16
    chords: str = "full"  # "full" | "reduce"
    max_chord: int = 5


@dataclass
class Event:
    hand: str  # annotated hand (annotator 1)
    onset: float
    offset: float  # end of sound after clipping (max over the chord's notes)
    pitches: Tuple[int, ...]  # ascending
    keys: Tuple[Tuple[int, int], ...]  # PIG note identities, aligned with pitches
    hold_until: Tuple[float, ...]  # per note: original offset if the note is held, else -inf
    orig_offset: Tuple[float, ...]  # per note, before clipping
    labels: List[Tuple[Label, ...]] = field(default_factory=list)  # per annotator, per note
    dropped: Tuple[int, ...] = ()  # pitches dropped by --chords reduce
    full_size: int = 1  # chord size before reduction
    full_span: int = 0

    @property
    def size(self) -> int:
        return len(self.pitches)

    @property
    def blacks(self) -> Tuple[bool, ...]:
        return tuple(is_black(p) for p in self.pitches)

    @property
    def single(self) -> bool:
        return len(self.pitches) == 1


@dataclass
class Segment:
    piece: str
    index: int
    events: List[Event]
    synthetic: bool = False

    def __len__(self) -> int:
        return len(self.events)

    @property
    def n_notes(self) -> int:
        return sum(e.size for e in self.events)

    def hand_events(self, hand: str) -> List[int]:
        return [i for i, e in enumerate(self.events) if e.hand == hand]

    @property
    def n_annotators(self) -> int:
        return len(self.events[0].labels) if self.events else 0


# --------------------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------------------
@dataclass
class _N:
    """A note being processed (mutable copy of pig.Note)."""

    onset: float
    offset: float
    orig_offset: float
    pitch: int
    key: Tuple[int, int]
    hold_until: float = -np.inf


def _hand_events(notes: Sequence[Note], hand: str, cfg: PrepConfig, log: Dict[str, int]) -> List[Event]:
    ns = sorted(
        (_N(n.onset, n.offset, n.offset, n.pitch, n.key) for n in notes if n.hand == hand),
        key=lambda n: (n.onset, n.pitch),
    )
    # overlap tolerance (§C.4.2): compare each note with the first later onset group of this hand
    for i, a in enumerate(ns):
        for b in ns[i + 1:]:
            if b.onset - a.onset > cfg.chord_tol:
                if a.offset > b.onset:
                    if a.offset - b.onset > cfg.overlap_tol:
                        a.hold_until = a.offset
                        log["held"] += 1
                    else:
                        log["legato"] += 1
                    a.offset = b.onset
                break
    # chord grouping (§C.4.3)
    groups: List[List[_N]] = []
    for n in ns:
        if groups and n.onset - groups[-1][0].onset <= cfg.chord_tol:
            groups[-1].append(n)
        else:
            groups.append([n])
    events = []
    for g in groups:
        g = sorted(g, key=lambda n: n.pitch)
        # two annotated notes on the same key at once: keep one
        uniq: Dict[int, _N] = {}
        for n in g:
            uniq.setdefault(n.pitch, n)
        g = list(uniq.values())
        full_size, full_span = len(g), g[-1].pitch - g[0].pitch
        if len(g) > cfg.max_chord:
            log["chord>5"] += 1
            g = g[: cfg.max_chord] if hand == LEFT else g[-cfg.max_chord:]
        dropped: Tuple[int, ...] = ()
        if cfg.chords == "reduce" and len(g) > 1:
            keep = g[0] if hand == LEFT else g[-1]
            dropped = tuple(n.pitch for n in g if n is not keep)
            g = [keep]
        events.append(
            Event(
                hand=hand,
                onset=g[0].onset,
                offset=max(n.offset for n in g),
                pitches=tuple(n.pitch for n in g),
                keys=tuple(n.key for n in g),
                hold_until=tuple(n.hold_until for n in g),
                orig_offset=tuple(n.orig_offset for n in g),
                dropped=dropped,
                full_size=full_size,
                full_span=full_span,
            )
        )
    return events


def merge_streams(left: List[Event], right: List[Event]) -> List[Event]:
    """Both hands in onset order; ties: left hand first, then lower pitch (§C.4.4)."""
    return sorted(left + right, key=lambda e: (round(e.onset, 6), 0 if e.hand == LEFT else 1, e.pitches[0]))


def attach_labels(events: Sequence[Event], annotations: Sequence[Sequence[Note]]) -> None:
    lookups = []
    for notes in annotations:
        lk: Dict[Tuple[int, int], Note] = {}
        for n in notes:
            lk.setdefault(n.key, n)
        lookups.append(lk)
    for e in events:
        e.labels = []
        for lk in lookups:
            row = []
            for k in e.keys:
                n = lk.get(k)
                row.append((n.hand, n.finger) if n is not None and n.finger else (None, 0))
            e.labels.append(tuple(row))


def segment_stream(piece: str, events: List[Event], cfg: PrepConfig, min_len: Optional[int] = None,
                   synthetic: bool = False) -> List[Segment]:
    """Cut where both hands rest longer than rest_gap, or at max_len events (§C.4.5)."""
    min_len = cfg.min_len if min_len is None else min_len
    segs: List[List[Event]] = []
    cur: List[Event] = []
    sounding_until = -np.inf
    for e in events:
        if cur and (e.onset - sounding_until > cfg.rest_gap or len(cur) >= cfg.max_len):
            segs.append(cur)
            cur = []
            sounding_until = -np.inf
        cur.append(e)
        sounding_until = max(sounding_until, max(e.orig_offset))
    if cur:
        segs.append(cur)
    out = [s for s in segs if len(s) >= min_len]
    return [Segment(piece, i, s, synthetic) for i, s in enumerate(out)]


def piece_events(annotations: Sequence[Sequence[Note]], cfg: PrepConfig,
                 log: Optional[Dict[str, int]] = None) -> List[Event]:
    log = log if log is not None else defaultdict(int)
    ref = annotations[0]
    ev = merge_streams(_hand_events(ref, LEFT, cfg, log), _hand_events(ref, RIGHT, cfg, log))
    attach_labels(ev, annotations)
    return ev


# --------------------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------------------
@dataclass
class TwoHandSplit:
    train: List[Segment]
    val: List[Segment]
    test: List[Segment]
    train_pieces: List[str]
    val_pieces: List[str]
    test_pieces: List[str]
    log: Dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        def row(name, segs, pieces):
            ev = sum(len(s) for s in segs)
            notes = sum(s.n_notes for s in segs)
            return f"{name:<6}{len(pieces):>4} pieces {len(segs):>5} segments {ev:>7} events {notes:>7} notes"

        return "\n".join([row("train", self.train, self.train_pieces), row("val", self.val, self.val_pieces),
                          row("test", self.test, self.test_pieces)])


def make_twohand_split(pig: Dict[str, List[List[Note]]], cfg: PrepConfig = PrepConfig(),
                       val_pieces: int = 12, seed: int = 0, eval_min_len: int = 2) -> TwoHandSplit:
    """Test = the 2025 paper's 30 pieces; validation = 12 seeded others; the rest train.
    Training segments need cfg.min_len events; validation and test keep every segment of at least
    eval_min_len events so the metrics cover (almost) all notes."""
    test_ids = [k for k in TEST_IDS_2025 if k in pig]
    rest = sorted(k for k in pig if k not in set(test_ids))
    rng = random.Random(seed)
    rng.shuffle(rest)
    val_ids, train_ids = sorted(rest[:val_pieces]), sorted(rest[val_pieces:])
    log: Dict[str, int] = defaultdict(int)

    def collect(ids, min_len):
        out = []
        for k in ids:
            out += segment_stream(k, piece_events(pig[k], cfg, log), cfg, min_len)
        return out

    return TwoHandSplit(collect(train_ids, cfg.min_len), collect(val_ids, eval_min_len),
                        collect(test_ids, eval_min_len), train_ids, val_ids, test_ids, dict(log))


def load_twohand(pig_dir: str, cfg: PrepConfig = PrepConfig(), val_pieces: int = 12, seed: int = 0,
                 verbose: bool = True) -> TwoHandSplit:
    split = make_twohand_split(load_pig(pig_dir), cfg, val_pieces, seed)
    if verbose:
        print(split.summary())
    return split


# --------------------------------------------------------------------------------------
# Coverage report (§C.4.8)
# --------------------------------------------------------------------------------------
def coverage_rows(pig: Dict[str, List[List[Note]]], cfg: PrepConfig) -> List[dict]:
    """Per hand and setting: fraction of notes fingered, segments, mean segment length, chord and
    held shares. Setting 1 = Part B's filter (monophonic runs >= 8), 2 = overlap tolerance only
    (chords dropped), 3 = overlap tolerance + chords."""
    from ..pig import monophonic_runs

    rows = []
    for hand in (RIGHT, LEFT):
        total = sum(sum(n.hand == hand for n in anns[0]) for anns in pig.values())
        # (1) Part B
        runs = [r for anns in pig.values() for r in monophonic_runs([n for n in anns[0] if n.hand == hand], 8)]
        rows.append(dict(hand=hand, setting="1: Part B monophonic filter", notes=sum(map(len, runs)), total=total,
                         segments=len(runs), mean_len=float(np.mean([len(r) for r in runs])) if runs else 0.0,
                         chord_share=0.0, held_share=0.0))
        for name, keep_chords in (("2: overlap tolerance only", False), ("3: overlap tolerance + chords", True)):
            log: Dict[str, int] = defaultdict(int)
            notes = segs = events = chords = held = 0
            lens = []
            for k, anns in pig.items():
                ev = [e for e in _hand_events(anns[0], hand, cfg, log)]
                if not keep_chords:
                    # chords end a run: split the hand stream at chords, keep single-note runs
                    runs2, cur = [], []
                    for e in ev:
                        if e.single:
                            cur.append(e)
                        else:
                            runs2.append(cur)
                            cur = []
                    runs2.append(cur)
                    parts = [segment_stream(k, r, cfg) for r in runs2 if r]
                    ss = [s for p in parts for s in p]
                else:
                    ss = segment_stream(k, ev, cfg)
                for s in ss:
                    segs += 1
                    lens.append(len(s))
                    events += len(s)
                    for e in s.events:
                        notes += e.size
                        chords += not e.single
                        held += sum(h > -np.inf for h in e.hold_until)
            rows.append(dict(hand=hand, setting=name, notes=notes, total=total, segments=segs,
                             mean_len=float(np.mean(lens)) if lens else 0.0,
                             chord_share=chords / max(events, 1), held_share=held / max(notes, 1)))
    return rows


def coverage_markdown(rows: List[dict], cfg: PrepConfig) -> str:
    out = [
        "# Two-hand data coverage (README §C.4.8)",
        "",
        f"`chord_tol` {cfg.chord_tol} s, `overlap_tol` {cfg.overlap_tol} s, `rest_gap` {cfg.rest_gap} s, "
        f"segments {cfg.min_len}–{cfg.max_len} events. Notes, hands and onsets from annotator 1; all 150 PIG pieces.",
        "",
        "| Hand | Setting | Notes fingered | Segments | Mean segment length | Chord events | Held notes |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        out.append(
            f"| {r['hand']} | {r['setting']} | {r['notes']}/{r['total']} ({100 * r['notes'] / max(r['total'], 1):.1f}%) "
            f"| {r['segments']} | {r['mean_len']:.1f} | {100 * r['chord_share']:.1f}% | {100 * r['held_share']:.1f}% |"
        )
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------------------
# Pianists' fingerings as hand streams, human hand speed (v_max), shared run context
# --------------------------------------------------------------------------------------
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
    from .rules import thumb_position  # rules does not import prepare, so this is safe

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
