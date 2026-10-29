"""Learning curves for one or more runs (e.g. several seeds).

    python plot.py runs/rh_ddqn_s0 runs/rh_ddqn_s1 runs/rh_ddqn_s2 --out results/curves.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows


def col(rows, key):
    xs, ys = [], []
    for r in rows:
        if r.get(key) not in (None, ""):
            xs.append(float(r["step"]))
            ys.append(float(r[key]))
    return xs, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="curves.png")
    args = ap.parse_args()

    panels = [("train_log.csv", "mean_return_per_note", "training return / note"),
              ("eval_log.csv", "val_effort", "validation effort / note"),
              ("eval_log.csv", "val_match", "validation match rate (Mgen)")]
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 3.6))
    for ax, (fname, key, title) in zip(axes, panels):
        for run in args.runs:
            path = Path(run) / fname
            if not path.exists():
                continue
            xs, ys = col(read_csv(path), key)
            if xs:
                ax.plot(xs, ys, label=Path(run).name, linewidth=1.4)
        ax.set_title(title)
        ax.set_xlabel("environment steps")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
