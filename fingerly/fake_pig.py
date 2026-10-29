"""Write a small fake dataset in PIG format, for tests and pipeline checks without the real PIG.

Pieces 001..n_multi get several "annotators" (the optimal-cost fingering with random edits),
the rest get one. Both hands are written, with a few chords and held notes so the
monophonic filter has something to remove. The numbers it produces mean nothing musically.

    python -m fingerly.fake_pig --out data/fake_pig --pieces 40 --multi 8
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from .baselines import OptimalCost
from .synth import synthetic_passages

NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def _name(p: int) -> str:
    return f"{NAMES[p % 12]}{p // 12 - 1}"


def write_fake_pig(out: str | Path, pieces: int = 40, multi: int = 8, seed: int = 0) -> Path:
    rng = random.Random(seed)
    folder = Path(out) / "FingeringFiles"
    folder.mkdir(parents=True, exist_ok=True)
    dp = OptimalCost()
    for k in range(1, pieces + 1):
        rh = synthetic_passages(rng.randint(2, 4), "R", seed=seed * 1000 + k)
        lh = synthetic_passages(rng.randint(1, 2), "L", seed=seed * 1000 + 500 + k)
        events = []  # (onset, offset, pitch, channel, finger_list_source)
        for hand, plist, ch in (("R", rh, 0), ("L", lh, 1)):
            t = 0.0
            for p in plist:
                fingers = dp.predict(p)
                for i, (pitch, f) in enumerate(zip(p.pitches, fingers)):
                    on, off = t + p.onsets[i], t + p.offsets[i]
                    events.append([on, off, pitch, ch, f, hand])
                t += p.offsets[-1] + 1.0
                # a chord between passages (removed by the filter)
                events.append([t, t + 0.5, 60 if hand == "R" else 43, ch, 1, hand])
                events.append([t, t + 0.5, 64 if hand == "R" else 48, ch, 3, hand])
                events.append([t, t + 0.5, 67 if hand == "R" else 52, ch, 5, hand])
                t += 1.0
        events.sort(key=lambda e: (e[0], e[2]))
        n_ann = rng.randint(4, 6) if k <= multi else 1
        for a in range(1, n_ann + 1):
            with open(folder / f"{k:03d}-{a}_fingering.txt", "w") as fo:
                fo.write("//Version: PianoFingering_v170101\n")
                for i, (on, off, pitch, ch, f, hand) in enumerate(events):
                    if a > 1 and rng.random() < 0.2:
                        f = rng.randint(1, 5)
                    tok = str(f) if hand == "R" else str(-f)
                    if rng.random() < 0.03:
                        tok = f"{tok}_{1 if hand == 'R' else -1}"
                    fo.write(f"{i}\t{on:.6f}\t{off:.6f}\t{_name(pitch)}\t64\t80\t{ch}\t{tok}\n")
    return folder.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/fake_pig")
    ap.add_argument("--pieces", type=int, default=40)
    ap.add_argument("--multi", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(f"wrote fake PIG data to {write_fake_pig(a.out, a.pieces, a.multi, a.seed)}")


if __name__ == "__main__":
    main()
