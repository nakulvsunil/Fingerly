"""Reproduce Iman & Ahn (2024): DQN, DDQN, PG and A2C on PianoFingering-V0..V3, 10 runs each.

    python reproduce_paper.py --pig-dir data/PianoFingeringDataset_v1.2 --workers 4
    python reproduce_paper.py --pig-dir data/PianoFingeringDataset_v1.2 --report-only

Each run is saved to <out>/raw/<env>_<algo>_<seed>.json as soon as it finishes, and finished
runs are skipped, so the script can be stopped and restarted. --report-only (or the end of a
full run) writes <out>/report.md (our tables next to the paper's), <out>/summary.json and
<out>/fig4_v0.png.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from fingerly.paper.agents import ALGOS, PaperConfig
from fingerly.paper.difficulty import difficulty_level, piece_rules
from fingerly.paper.metrics import match_rates, optimal_fingering
from fingerly.paper.pieces import load_pieces

ENVS = ("V0", "V1", "V2", "V3")

# Published numbers, Iman & Ahn (2024).
PAPER_BEST_L = {  # Table II: least difficult result over 10 runs
    "DQN": [0.6207, 1.3721, 3.1814, 2.5441],
    "DDQN": [0.6207, 2.5163, 5.2647, 3.25],
    "PG": [8.1034, 10.712, 14.478, 11.294],
    "A2C": [2.7586, 3.8930, 7.0441, 6.6176],
    "FHMM1 [9]": [1.0697, 2.3349, 3.6029, 2.8320],
    "FHMM2 [9]": [0.6207, 1.8884, 3.3235, 2.8028],
    "FHMM3 [9]": [0.6207, 2.2512, 3.2941, 2.9859],
    "Human": [0.6207, 1.8419, 4.0441, 2.9155],
}
PAPER_MEAN_L = {  # Tables III and IV: mean +- std of L over 10 runs
    "DQN": [(1.2552, 0.905), (1.51, 0.17), (3.78, 0.30), (3.14, 0.53)],
    "DDQN": [(0.7000, 0.238), (2.92, 0.21), (6.05, 0.78), (3.76, 0.35)],
    "PG": [(9.7138, 1.267), (11.61, 0.39), (16.5, 0.98), (13.38, 1.05)],
    "A2C": [(3.7448, 2.759), (5.08, 0.83), (8.94, 1.33), (7.75, 0.56)],
}
PAPER_TIME = {  # seconds, Tables III and IV (RTX 2080 GPU)
    "DQN": [51.6, 424, 294, 136], "DDQN": [55.7, 458, 288, 146],
    "PG": [9.8, 60, 43, 21], "A2C": [45.1, 345, 237, 115],
}
PAPER_MATCH = {  # Table V
    "DQN": (66.50, 68.97, 78.52, 72.55), "DDQN": (58.23, 61.10, 70.27, 62.29),
    "PG": (24.20, 26.97, 34.60, 28.16), "A2C": (43.08, 46.30, 54.18, 47.02),
    "FHMM1 [9]": (63.50, 66.11, 79.47, 74.22), "FHMM2 [9]": (61.32, 63.96, 76.61, 70.41),
    "FHMM3 [9]": (60.62, 61.34, 75.42, 70.17),
}


def run_one(pig_dir: str, env_id: str, algo: str, seed: int, episodes: int, path: str) -> str:
    import torch

    torch.set_num_threads(1)
    from fingerly.paper.agents import train
    from fingerly.paper.env import PianoFingeringEnv

    piece = load_pieces(pig_dir)[env_id]
    r = train(algo, PianoFingeringEnv(piece), seed, PaperConfig(episodes=episodes))
    r.update(env=env_id, algo=algo, seed=seed, L=difficulty_level(piece.pitches, r["fingering"]))
    tmp = path + ".tmp"
    Path(tmp).write_text(json.dumps(r))
    os.replace(tmp, path)
    return f"{env_id} {algo:4s} seed {seed}: L = {r['L']:.4f}  ({r['runtime']:.0f}s)"


# ----------------------------------------------------------------------------- report
def _load(raw: Path):
    runs = {}
    for p in raw.glob("*.json"):
        r = json.loads(p.read_text())
        runs.setdefault((r["env"], r["algo"]), []).append(r)
    return runs


def _f(x, d=4):
    return "–" if x is None else f"{x:.{d}f}"


def report(out: Path, pig_dir: str) -> None:
    pieces = load_pieces(pig_dir)
    runs = _load(out / "raw")
    opt = {e: optimal_fingering(pieces[e].pitches) for e in ENVS}
    summary = {"optimal_L": {e: difficulty_level(pieces[e].pitches, opt[e]) for e in ENVS}, "algos": {}}
    lines = ["# Reproduction of Iman & Ahn (2024): results\n"]
    n_runs = {k: len(v) for k, v in runs.items()}
    lines.append(f"Runs found per (environment, algorithm): {sorted(set(n_runs.values())) or [0]}. "
                 "Ours = this repo, CPU (Intel Xeon 2.8 GHz, 1 thread per run). Paper = published numbers.\n")

    # Table II
    lines += ["## Table II: difficulty level L(s,a) of the best run (lower is better)\n",
              "| Method | V0 ours | V0 paper | V1 ours | V1 paper | V2 ours | V2 paper | V3 ours | V3 paper |",
              "|---|---|---|---|---|---|---|---|---|"]
    for algo in ALGOS:
        row, best = [algo], []
        for i, e in enumerate(ENVS):
            rs = runs.get((e, algo), [])
            b = min((r["L"] for r in rs), default=None)
            best.append(b)
            row += [_f(b), _f(PAPER_BEST_L[algo][i])]
        summary["algos"].setdefault(algo, {})["best_L"] = best
        lines.append("| " + " | ".join(row) + " |")
    for name in ("FHMM1 [9]", "FHMM2 [9]", "FHMM3 [9]"):
        lines.append("| " + " | ".join([name] + sum([["–", _f(v)] for v in PAPER_BEST_L[name]], [])) + " |")
    human = []
    for e in ENVS:
        hl = [difficulty_level(pieces[e].pitches, [f or 1 for f in lab]) for lab in pieces[e].labels]
        human.append(hl)
    lines.append("| Human (annotators, min–max) | " + " | ".join(
        f"{min(h):.4f}–{max(h):.4f} | {PAPER_BEST_L['Human'][i]:.4f}" for i, h in enumerate(human)) + " |")
    lines.append("| **Optimal (exact DP, ours)** | " + " | ".join(
        f"**{summary['optimal_L'][e]:.4f}** | –" for e in ENVS) + " |")
    lines.append("")

    # Tables III / IV
    lines += ["## Tables III and IV: mean ± std of L over runs, and run time per run\n",
              "| Method | Env | L ours | L paper | time ours (s, CPU) | time paper (s, GPU) |",
              "|---|---|---|---|---|---|"]
    for algo in ALGOS:
        stats = []
        for i, e in enumerate(ENVS):
            rs = runs.get((e, algo), [])
            if rs:
                L = np.array([r["L"] for r in rs])
                T = np.array([r["runtime"] for r in rs])
                ours, tim = f"{L.mean():.4f} ± {L.std():.4f}", f"{T.mean():.0f} ± {T.std():.0f}"
                stats.append({"mean": float(L.mean()), "std": float(L.std()), "time": float(T.mean()), "n": len(rs)})
            else:
                ours, tim = "–", "–"
                stats.append(None)
            pm, ps = PAPER_MEAN_L[algo][i]
            lines.append(f"| {algo} | {e} | {ours} | {pm} ± {ps} | {tim} | {PAPER_TIME[algo][i]} |")
        summary["algos"][algo]["mean_L"] = stats
    lines.append("")

    # Table V: match rates of the best run, averaged over V1..V3 (the multi-annotator pieces)
    lines += ["## Table V: match rate with the human fingerings (%), best run per piece, mean over V1–V3\n",
              "| Method | Mgen ours | Mgen paper | Mhigh ours | Mhigh paper | Msoft ours | Msoft paper | Mrec* ours | Mrec paper |",
              "|---|---|---|---|---|---|---|---|---|"]
    keys = ("Mgen", "Mhigh", "Msoft", "Mrec")

    def mean_match(fingerings):
        ms = [match_rates(fingerings[e], pieces[e].labels) for e in ("V1", "V2", "V3")]
        return [100 * float(np.mean([m[k] for m in ms])) for k in keys]

    for algo in ALGOS:
        best = {}
        for e in ("V1", "V2", "V3"):
            rs = runs.get((e, algo), [])
            if rs:
                best[e] = min(rs, key=lambda r: r["L"])["fingering"]
        if len(best) == 3:
            ours = mean_match(best)
            summary["algos"][algo]["match"] = dict(zip(keys, ours))
        else:
            ours = [None] * 4
        cells = sum([[_f(o, 2), f"{p:.2f}"] for o, p in zip(ours, PAPER_MATCH[algo])], [])
        lines.append("| " + " | ".join([algo] + cells) + " |")
    for name in ("FHMM1 [9]", "FHMM2 [9]", "FHMM3 [9]"):
        lines.append("| " + " | ".join([name] + sum([["–", f"{p:.2f}"] for p in PAPER_MATCH[name]], [])) + " |")
    o = mean_match(opt)
    summary["optimal_match"] = dict(zip(keys, o))
    lines.append("| **Optimal (exact DP, ours)** | " + " | ".join(f"{v:.2f} | –" for v in o) + " |")
    lines.append("\n*Mrec: switching annotator costs one match; the exact definition could not be checked.\n")

    # Which rules the best DQN fingering still pays for
    lines += ["## Where the difficulty comes from (best DQN run vs exact optimum, points per rule)\n",
              "| Env | Fingering | " + " | ".join(r.replace("_", " ") for r in piece_rules([60], [1])) + " |",
              "|---|---|" + "---|" * 12]
    for e in ENVS:
        rs = runs.get((e, "DQN"), [])
        rows = [("optimal", opt[e])]
        if rs:
            rows.insert(0, ("best DQN", min(rs, key=lambda r: r["L"])["fingering"]))
        for name, f in rows:
            pr = piece_rules(pieces[e].pitches, f)
            lines.append(f"| {e} | {name} | " + " | ".join(f"{v:.0f}" for v in pr.values()) + " |")
    lines.append("")

    (out / "report.md").write_text("\n".join(lines))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    _fig4(out, runs)
    print("\n".join(lines))


def _fig4(out: Path, runs) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.6))
    for ax, e in zip(axes, ENVS):
        for algo, style in zip(ALGOS, ("-", "--", ":", "-.")):
            rs = runs.get((e, algo), [])
            if not rs:
                continue
            R = np.array([r["returns"] for r in rs])
            m, s = R.mean(0), R.std(0)
            x = np.arange(1, len(m) + 1)
            ax.plot(x, m, style, label=f"{algo} (n={len(rs)})")
            ax.fill_between(x, m - s, m + s, alpha=0.2)
        ax.set_title(f"PianoFingering-{e}")
        ax.set_xlabel("episode")
    axes[0].set_ylabel("episode reward (−D)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out / "fig4_rewards.png", dpi=130)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pig-dir", required=True)
    ap.add_argument("--out", default="results/paper")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--envs", nargs="+", default=list(ENVS))
    ap.add_argument("--algos", nargs="+", default=list(ALGOS))
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)

    if not args.report_only:
        size = {"V0": 29, "V1": 215, "V2": 136, "V3": 68}
        cost = {"DQN": 6, "DDQN": 6, "A2C": 4, "PG": 1}
        jobs = [(e, a, s) for e in args.envs for a in args.algos for s in range(args.runs)
                if not (out / "raw" / f"{e}_{a}_{s}.json").exists()]
        jobs.sort(key=lambda j: -size[j[0]] * cost[j[1]])  # longest first
        print(f"{len(jobs)} runs to do with {args.workers} workers", flush=True)
        with ProcessPoolExecutor(args.workers) as ex:
            futs = [ex.submit(run_one, args.pig_dir, e, a, s, args.episodes, str(out / "raw" / f"{e}_{a}_{s}.json"))
                    for e, a, s in jobs]
            for i, f in enumerate(as_completed(futs), 1):
                print(f"[{i}/{len(jobs)}] {f.result()}", flush=True)
    report(out, args.pig_dir)


if __name__ == "__main__":
    main()
