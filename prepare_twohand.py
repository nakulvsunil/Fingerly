"""Two-hand data pipeline report (README §C.4): coverage table, overlap histograms, split sizes.

    python prepare_twohand.py --pig-dir data/PianoFingeringDataset_v1.2 --report results/twohand/coverage.md
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from fingerly.pig import LEFT, RIGHT, load_pig
from fingerly.twohand.prepare import (PrepConfig, coverage_markdown, coverage_rows, make_twohand_split)


def within_hand_gaps(pig, train_ids):
    """Onset differences to the next later onset and overlap lengths, within one hand, on
    training pieces (§C.4.2: pick the tolerances where the distributions separate)."""
    diffs, overlaps = [], []
    for k in train_ids:
        for hand in (LEFT, RIGHT):
            ns = sorted((n for n in pig[k][0] if n.hand == hand), key=lambda n: n.onset)
            for i, a in enumerate(ns):
                for b in ns[i + 1:]:
                    if b.onset > a.onset:
                        diffs.append(b.onset - a.onset)
                        if a.offset > b.onset:
                            overlaps.append(a.offset - b.onset)
                        break
    return np.array(diffs), np.array(overlaps)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pig-dir", required=True)
    ap.add_argument("--chord-tol", type=float, default=0.03)
    ap.add_argument("--overlap-tol", type=float, default=0.05)
    ap.add_argument("--rest-gap", type=float, default=1.5)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--min-len", type=int, default=16)
    ap.add_argument("--chords", choices=["full", "reduce"], default="full")
    ap.add_argument("--report", default="results/twohand/coverage.md")
    args = ap.parse_args()

    cfg = PrepConfig(args.chord_tol, args.overlap_tol, args.rest_gap, args.max_len, args.min_len, args.chords)
    pig = load_pig(args.pig_dir)
    split = make_twohand_split(pig, cfg)
    print(split.summary())
    print("pipeline log:", split.log)

    rows = coverage_rows(pig, cfg)
    md = coverage_markdown(rows, cfg)

    ev = [e for s in split.train for e in s.events]
    sizes = Counter(e.size for e in ev)
    md += "\n## Training events by chord size\n\n| Size | Events |\n| --- | --- |\n"
    md += "".join(f"| {k} | {sizes[k]} |\n" for k in sorted(sizes))
    md += (f"\nPipeline log over all pieces: {split.log.get('legato', 0)} legato overlaps made sequential, "
           f"{split.log.get('held', 0)} held notes, {split.log.get('chord>5', 0)} chords cut to 5 notes.\n")
    md += f"\n## Split\n\n```\n{split.summary()}\n```\n"

    diffs, ovl = within_hand_gaps(pig, split.train_pieces)
    bins_d = [0, .01, .02, .03, .05, .08, .12, np.inf]
    bins_o = [0, .01, .02, .03, .05, .08, .12, .2, .5, np.inf]
    md += "\n## Within-hand histograms (training pieces)\n\n| Onset difference (s) | Count |\n| --- | --- |\n"
    md += "".join(f"| {lo}–{hi} | {int(((diffs >= lo) & (diffs < hi)).sum())} |\n" for lo, hi in zip(bins_d, bins_d[1:]))
    md += "\n| Overlap length (s) | Count |\n| --- | --- |\n"
    md += "".join(f"| {lo}–{hi} | {int(((ovl >= lo) & (ovl < hi)).sum())} |\n" for lo, hi in zip(bins_o, bins_o[1:]))

    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
        ax[0].hist(np.clip(diffs, 0, 0.3), bins=60)
        ax[0].axvline(args.chord_tol, color="r", ls="--", label=f"chord_tol {args.chord_tol}")
        ax[0].set(xlabel="onset difference to next onset, same hand (s)", ylabel="count", yscale="log")
        ax[0].legend()
        ax[1].hist(np.clip(ovl, 0, 0.5), bins=50)
        ax[1].axvline(args.overlap_tol, color="r", ls="--", label=f"overlap_tol {args.overlap_tol}")
        ax[1].set(xlabel="overlap length, same hand (s)", yscale="log")
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(out.with_name("overlap_hist.png"), dpi=120)
    except ImportError:
        pass
    print(md)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
