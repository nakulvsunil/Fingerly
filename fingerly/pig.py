"""Reading the PIG dataset (Nakamura et al., 2020) and cutting it into one-hand monophonic passages.

A PIG fingering file has one header line starting with "//" and then one note per line:

    noteID  onset  offset  pitch  onsetVel  offsetVel  channel  finger
    0       0.000  0.250   E4     64        80         0        3
    1       0.250  0.500   F#4    64        80         0        3_1
    2       0.500  1.000   Bb2    64        80         1        -5

Channel 0 = right hand, 1 = left hand. Left-hand fingers are negative. "3_1" is a finger
substitution (3 presses, 1 then holds); we keep the first finger. Files are named
<piece>-<annotator>_fingering.txt, e.g. 001-1_fingering.txt.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

BLACK_PITCH_CLASSES = {1, 3, 6, 8, 10}
_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_PITCH_RE = re.compile(r"^([A-Ga-g])([#bx\-]*)(\d+)$")
_FILE_RE = re.compile(r"^(\d+)-(\d+)_fingering\.txt$")

RIGHT, LEFT = "R", "L"


# --------------------------------------------------------------------------------------
# Low-level parsing
# --------------------------------------------------------------------------------------
def pitch_to_midi(name: str) -> int:
    """'C4' -> 60, 'F#4' -> 66, 'Bb2' -> 46, 'E-4' -> 63 (music21 flat), 'Fx4' -> 67."""
    m = _PITCH_RE.match(name.strip())
    if not m:
        raise ValueError(f"Cannot parse pitch name: {name!r}")
    letter, acc, octave = m.groups()
    shift = 0
    for ch in acc:
        shift += {"#": 1, "x": 2, "b": -1, "-": -1}[ch]
    return 12 * (int(octave) + 1) + _PC[letter.upper()] + shift


def is_black(midi: int) -> bool:
    return midi % 12 in BLACK_PITCH_CLASSES


def parse_finger(token: str) -> Tuple[Optional[str], Optional[int]]:
    """'3' -> ('R', 3); '-5' -> ('L', 5); '3_1' -> ('R', 3); '-3_-1' -> ('L', 3).

    Returns (None, None) if the token is not a valid finger (e.g. '0' or 'x')."""
    first = token.strip().split("_")[0]
    try:
        value = int(first)
    except ValueError:
        return None, None
    if value == 0 or abs(value) > 5:
        return None, None
    return (RIGHT if value > 0 else LEFT), abs(value)


@dataclass
class Note:
    note_id: int
    onset: float
    offset: float
    pitch: int  # MIDI number
    channel: int  # 0 = right hand, 1 = left hand (as written in the file)
    hand: Optional[str]  # from the sign of the finger ("R"/"L")
    finger: Optional[int]  # 1..5

    @property
    def black(self) -> bool:
        return is_black(self.pitch)

    @property
    def key(self) -> Tuple[int, int]:
        """Identity of a note inside a piece, shared by all annotators of that piece."""
        return (int(round(self.onset * 1000)), self.pitch)


def read_fingering_file(path: str | Path) -> List[Note]:
    notes: List[Note] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            hand, finger = parse_finger(parts[7])
            channel = int(parts[6])
            if hand is None:  # fall back to the channel if the finger is missing
                hand = RIGHT if channel == 0 else LEFT
            notes.append(
                Note(
                    note_id=int(parts[0]),
                    onset=float(parts[1]),
                    offset=float(parts[2]),
                    pitch=pitch_to_midi(parts[3]),
                    channel=channel,
                    hand=hand,
                    finger=finger,
                )
            )
    notes.sort(key=lambda n: (n.onset, n.pitch, n.note_id))
    return notes


# --------------------------------------------------------------------------------------
# Passages
# --------------------------------------------------------------------------------------
@dataclass
class Passage:
    """A run of notes for one hand where exactly one note sounds at a time."""

    piece: str
    hand: str
    pitches: List[int]
    onsets: List[float]
    offsets: List[float]
    keys: List[Tuple[int, int]] = field(default_factory=list)
    # One list of fingers per annotator (0 = this annotator gave the note to the other hand
    # or left it unfingered). Empty for unlabeled/synthetic passages.
    labels: List[List[int]] = field(default_factory=list)
    start_index: int = 0  # position of the first note inside the hand's note list

    def __len__(self) -> int:
        return len(self.pitches)

    @property
    def blacks(self) -> List[bool]:
        return [is_black(p) for p in self.pitches]


def monophonic_runs(
    notes: Sequence[Note],
    min_len: int = 8,
    tol: float = 1e-3,
) -> List[List[Note]]:
    """Split one hand's notes into monophonic runs.

    A note can continue the current run only if (a) no other note of this hand starts at the
    same time and (b) every earlier note of this hand has already ended (so nothing is held
    under it). Chords and notes played over held notes end the run and are dropped.
    """
    notes = sorted(notes, key=lambda n: (n.onset, n.pitch))
    # group simultaneous onsets
    groups: List[List[Note]] = []
    for n in notes:
        if groups and abs(n.onset - groups[-1][0].onset) <= tol:
            groups[-1].append(n)
        else:
            groups.append([n])

    runs: List[List[Note]] = []
    current: List[Note] = []
    busy_until = float("-inf")
    for g in groups:
        onset = g[0].onset
        if len(g) > 1 or onset < busy_until - tol:
            if len(current) >= min_len:
                runs.append(current)
            # A chord, or a single note played while an earlier note is still held, cannot
            # start a clean run either: it is dropped.
            current = []
        else:
            current.append(g[0])
        busy_until = max(busy_until, max(n.offset for n in g))
    if len(current) >= min_len:
        runs.append(current)
    return runs


def _passage_from_run(piece: str, hand: str, run: List[Note]) -> Passage:
    return Passage(
        piece=piece,
        hand=hand,
        pitches=[n.pitch for n in run],
        onsets=[n.onset for n in run],
        offsets=[n.offset for n in run],
        keys=[n.key for n in run],
    )


def attach_labels(passage: Passage, annotations: Sequence[List[Note]]) -> None:
    """Fill passage.labels with each annotator's finger for every note (0 if the annotator
    gave that note to the other hand or it cannot be matched)."""
    passage.labels = []
    for notes in annotations:
        lookup: Dict[Tuple[int, int], List[Note]] = defaultdict(list)
        for n in notes:
            lookup[n.key].append(n)
        fingers = []
        for k in passage.keys:
            cands = lookup.get(k, [])
            same_hand = [n for n in cands if n.hand == passage.hand and n.finger]
            fingers.append(same_hand[0].finger if same_hand else 0)
        passage.labels.append(fingers)


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------
def find_fingering_dir(root: str | Path) -> Path:
    """Accept either the FingeringFiles folder itself or any parent of it."""
    root = Path(root)
    if any(_FILE_RE.match(p.name) for p in root.glob("*_fingering.txt")):
        return root
    for p in root.rglob("*_fingering.txt"):
        if _FILE_RE.match(p.name):
            return p.parent
    raise FileNotFoundError(
        f"No PIG fingering files (NNN-N_fingering.txt) found under {root}. "
        "Download PIG from http://beam.kisarazu.ac.jp/research/PianoFingeringDataset/ "
        "and point --pig-dir at it."
    )


def load_pig(root: str | Path) -> Dict[str, List[List[Note]]]:
    """Returns {piece_id: [annotation_1_notes, annotation_2_notes, ...]}."""
    folder = find_fingering_dir(root)
    pieces: Dict[str, Dict[int, List[Note]]] = defaultdict(dict)
    for p in sorted(folder.glob("*_fingering.txt")):
        m = _FILE_RE.match(p.name)
        if not m:
            continue
        piece, ann = m.group(1), int(m.group(2))
        pieces[piece][ann] = read_fingering_file(p)
    return {k: [v[a] for a in sorted(v)] for k, v in sorted(pieces.items())}


def piece_passages(
    piece: str,
    annotations: List[List[Note]],
    hand: str = RIGHT,
    min_len: int = 8,
) -> List[Passage]:
    """Monophonic passages of one piece for one hand.

    The note list (and the hand each note belongs to) comes from the first annotator; every
    annotator's fingering for those notes is attached as a label row."""
    reference = [n for n in annotations[0] if n.hand == hand]
    passages = [_passage_from_run(piece, hand, r) for r in monophonic_runs(reference, min_len)]
    for p in passages:
        attach_labels(p, annotations)
    return passages


@dataclass
class Split:
    train: List[Passage]
    val: List[Passage]
    test: List[Passage]
    train_pieces: List[str]
    val_pieces: List[str]
    test_pieces: List[str]

    def summary(self) -> str:
        def n(ps):
            return sum(len(p) for p in ps)

        return (
            f"train: {len(self.train_pieces)} pieces, {len(self.train)} passages, {n(self.train)} notes\n"
            f"val:   {len(self.val_pieces)} pieces, {len(self.val)} passages, {n(self.val)} notes\n"
            f"test:  {len(self.test_pieces)} pieces, {len(self.test)} passages, {n(self.test)} notes"
        )


def make_split(
    pig: Dict[str, List[List[Note]]],
    hand: str = RIGHT,
    min_len: int = 8,
    val_pieces: int = 12,
    seed: int = 0,
) -> Split:
    """Test = pieces fingered by several pianists (the 30 multi-annotator pieces of PIG).
    The rest is shuffled with a fixed seed into validation (val_pieces) and training.
    Pieces are never split across sets."""
    test_ids = [k for k, v in pig.items() if len(v) >= 2]
    rest = [k for k, v in pig.items() if len(v) < 2]
    rng = random.Random(seed)
    rng.shuffle(rest)
    val_ids = sorted(rest[:val_pieces])
    train_ids = sorted(rest[val_pieces:])

    def collect(ids):
        out = []
        for k in ids:
            out.extend(piece_passages(k, pig[k], hand=hand, min_len=min_len))
        return out

    return Split(
        train=collect(train_ids),
        val=collect(val_ids),
        test=collect(test_ids),
        train_pieces=train_ids,
        val_pieces=val_ids,
        test_pieces=sorted(test_ids),
    )


def coverage(pig: Dict[str, List[List[Note]]], hand: str = RIGHT, min_len: int = 8) -> Tuple[int, int]:
    """(notes kept in monophonic passages, all notes of that hand), using annotator 1."""
    kept = total = 0
    for piece, anns in pig.items():
        ref = [n for n in anns[0] if n.hand == hand]
        total += len(ref)
        kept += sum(len(r) for r in monophonic_runs(ref, min_len))
    return kept, total
