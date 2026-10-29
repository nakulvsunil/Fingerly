"""The paper's four environments (README §A.2): V0 = Beyer Op. 101 No. 12 (transcribed, not in
PIG), V1-V3 = three PIG pieces, right hand only.

The paper's L values times N give whole numbers for N = 215 (V1), 136 (V2) and 68 (V3), which are
the full right-hand note counts of PIG 025, 061 and 140. So the paper used each piece's whole
right-hand stream, including notes that overlap the previous one (legato); we do the same and
report chords and overlaps instead of dropping them.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from ..pig import RIGHT, find_fingering_dir, read_fingering_file

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "paper"
V0_FILE = DATA_DIR / "V0_beyer_op101_no12_fingering.txt"
PIECES_JSON = DATA_DIR / "pieces.json"
ENVS = ("V0", "V1", "V2", "V3")

# (composer, words that must all appear in the PIG title)
PAPER_PIECES = {
    "V1": ("Chopin", ["waltz", "64", "1"]),
    "V2": ("Beethoven", ["elise"]),
    "V3": ("Scarlatti", ["208"]),
}


@dataclass
class PaperPiece:
    env: str
    title: str
    pitches: List[int]
    labels: List[List[int]] = field(default_factory=list)  # one finger list per annotator
    annotators: List[str] = field(default_factory=list)
    pig_id: Optional[str] = None
    chords: int = 0
    overlaps: int = 0

    def __len__(self) -> int:
        return len(self.pitches)


def find_pig_ids(pig_dir: str | Path) -> Dict[str, dict]:
    """Search PIG's List.csv for the three pieces of the paper."""
    folder = find_fingering_dir(pig_dir)
    lists = list(folder.parent.glob("List.csv")) or list(Path(pig_dir).rglob("List.csv"))
    if not lists:
        raise FileNotFoundError("List.csv not found next to FingeringFiles/")
    with open(lists[0], encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    out = {}
    for env, (composer, words) in PAPER_PIECES.items():
        hits = []
        for row in rows[1:]:
            if len(row) < 3 or row[1].strip().lower() != composer.lower():
                continue
            title = row[2].lower()
            tokens = re.findall(r"[a-z]+|\d+", title)
            if all(w in tokens for w in words):
                hits.append(row)
        if len(hits) != 1:
            raise RuntimeError(f"{env}: expected one PIG match for {composer} {words}, got {len(hits)}")
        row = hits[0]
        out[env] = {"pig_id": row[0].strip(), "composer": row[1], "title": row[2]}
    return out


def _right_hand_stream(notes) -> tuple:
    """Right-hand notes in time order, with the number of chord onsets and of overlaps."""
    r = sorted([n for n in notes if n.hand == RIGHT], key=lambda n: (n.onset, n.pitch))
    chords = sum(1 for a, b in zip(r, r[1:]) if abs(a.onset - b.onset) <= 1e-3)
    overlaps = sum(1 for a, b in zip(r, r[1:]) if b.onset < a.offset - 1e-3 and abs(b.onset - a.onset) > 1e-3)
    return r, chords, overlaps


def load_pig_piece(env: str, pig_id: str, title: str, pig_dir: str | Path) -> PaperPiece:
    folder = find_fingering_dir(pig_dir)
    files = sorted(folder.glob(f"{pig_id}-*_fingering.txt"))
    if not files:
        raise FileNotFoundError(f"no PIG files for {pig_id}")
    ref, chords, overlaps = _right_hand_stream(read_fingering_file(files[0]))
    piece = PaperPiece(env, title, [n.pitch for n in ref], pig_id=pig_id, chords=chords, overlaps=overlaps)
    for path in files:
        notes, _, _ = _right_hand_stream(read_fingering_file(path))
        by_key = {n.key: n.finger for n in notes}
        piece.labels.append([by_key.get(n.key) or 0 for n in ref])
        piece.annotators.append(path.name.split("_")[0])
    return piece


def load_v0(path: str | Path = V0_FILE) -> PaperPiece:
    ref, chords, overlaps = _right_hand_stream(read_fingering_file(path))
    return PaperPiece(
        "V0", "Beyer, Vorschule im Klavierspiel Op. 101 No. 12 (right hand)",
        [n.pitch for n in ref], labels=[[n.finger for n in ref]], annotators=["composer"],
        chords=chords, overlaps=overlaps,
    )


def write_pieces_json(pig_dir: str | Path, path: str | Path = PIECES_JSON) -> Dict[str, dict]:
    ids = find_pig_ids(pig_dir)
    info = {}
    for env, meta in ids.items():
        p = load_pig_piece(env, meta["pig_id"], meta["title"], pig_dir)
        info[env] = {**meta, "right_hand_notes": len(p), "chord_onsets": p.chords,
                     "overlaps": p.overlaps, "annotators": p.annotators}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(info, indent=2))
    return info


def load_piece(env: str, pig_dir: Optional[str | Path] = None) -> PaperPiece:
    if env == "V0":
        return load_v0()
    if PIECES_JSON.exists():
        meta = json.loads(PIECES_JSON.read_text())[env]
    elif pig_dir is not None:
        meta = find_pig_ids(pig_dir)[env]
    else:
        raise FileNotFoundError("run `python reproduce.py pieces --pig-dir ...` first")
    if pig_dir is None:
        raise ValueError(f"{env} needs --pig-dir")
    return load_pig_piece(env, meta["pig_id"], meta["title"], pig_dir)
