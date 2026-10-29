"""The four PianoFingering environments of Iman & Ahn (2024), section III-B.

V0  Beyer, Vorschule im Klavierspiel Op. 101 No. 12   transcribed from the paper's Fig. 5 (29 notes)
V1  Chopin, Waltz Op. 64 No. 1                         PIG piece 025, right hand (215 notes, 5 annotators)
V2  Beethoven, Für Elise                               PIG piece 061, right hand (136 notes, 2 annotators)
V3  Scarlatti, Sonata in A major K. 208                PIG piece 140, right hand (68 notes, 2 annotators)

The note counts agree with the paper: multiplying its L(s, a) values (Table II) by 29 / 215 /
136 / 68 gives whole numbers of difficulty points. Notes are taken in onset order; none of the
three PIG right hands has two notes starting together.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from ..pig import load_pig

C4, D4, E4, F4, G4 = 60, 62, 64, 65, 67

BEYER_12 = [C4, D4, E4, F4, G4, F4, E4, D4] * 3 + [E4, C4, E4, D4, C4]
# Composer's fingering, identical to the paper's DQN / DDQN / human rows in Fig. 5.
BEYER_12_FINGERING = [1, 2, 3, 4, 5, 4, 3, 2] * 3 + [3, 1, 3, 2, 1]

PIG_PIECES = {"V1": "025", "V2": "061", "V3": "140"}
TITLES = {
    "V0": "Beyer Op. 101 No. 12",
    "V1": "Chopin Waltz Op. 64 No. 1",
    "V2": "Beethoven Für Elise",
    "V3": "Scarlatti Sonata K. 208",
}


@dataclass
class Piece:
    env_id: str
    title: str
    pitches: List[int]
    # One row of fingers per annotator (0 = that annotator gave the note to the left hand).
    labels: List[List[int]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.pitches)


def load_pieces(pig_dir: str) -> Dict[str, Piece]:
    pieces = {"V0": Piece("V0", TITLES["V0"], list(BEYER_12), [list(BEYER_12_FINGERING)])}
    pig = load_pig(pig_dir)
    for env_id, pid in PIG_PIECES.items():
        anns = pig[pid]
        ref = [n for n in anns[0] if n.hand == "R"]
        labels = []
        for ann in anns:
            by_key = {n.key: n for n in ann if n.hand == "R" and n.finger}
            labels.append([by_key[n.key].finger if n.key in by_key else 0 for n in ref])
        pieces[env_id] = Piece(env_id, TITLES[env_id], [n.pitch for n in ref], labels)
    return pieces
