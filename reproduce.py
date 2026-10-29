"""Part A: reproduce Iman & Ahn (2024). See README §A.

    python reproduce.py pieces       --pig-dir data/PianoFingeringDataset_v1.2
    python reproduce.py check-reward --pig-dir data/PianoFingeringDataset_v1.2
    python reproduce.py train  --env all --algo all --runs 10 --out runs/paper --pig-dir ...
    python reproduce.py report --runs runs/paper --pig-dir ... --out results/paper
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np

from fingerly.paper.pieces import ENVS, load_piece, write_pieces_json
from fingerly.paper.rules import RULE_NAMES, RuleConfig, breakdown, format_breakdown, level

ALGOS = ("dqn", "ddqn", "pg", "a2c")
ALGO_LABEL = {"dqn": "DQN", "ddqn": "DDQN", "pg": "PG", "a2c": "A2C"}

# Targets printed in the paper (README §A.6)
PAPER_BEST_L = {
    "DQN": (0.6207, 1.3721, 3.1814, 2.5441), "DDQN": (0.6207, 2.5163, 5.2647, 3.25),
    "PG": (8.1034, 10.712, 14.478, 11.294), "A2C": (2.7586, 3.8930, 7.0441, 6.6176),
    "FHMM1": (1.0697, 2.3349, 3.6029, 2.8320), "FHMM2": (0.6207, 1.8884, 3.3235, 2.8028),
    "FHMM3": (0.6207, 2.2512, 3.2941, 2.9859), "Human": (0.6207, 1.8419, 4.0441, 2.9155),
}
PAPER_MEAN_L = {
    "DQN": ("1.2552 ± 0.905", "1.51 ± 0.17", "3.78 ± 0.30", "3.14 ± 0.53"),
    "DDQN": ("0.7000 ± 0.238", "2.92 ± 0.21", "6.05 ± 0.78", "3.76 ± 0.35"),
    "PG": ("9.7138 ± 1.267", "11.61 ± 0.39", "16.5 ± 0.98", "13.38 ± 1.05"),
    "A2C": ("3.7448 ± 2.759", "5.08 ± 0.83", "8.94 ± 1.33", "7.75 ± 0.56"),
}
PAPER_MATCH = {
    "DQN": (66.50, 68.97, 78.52, 72.55), "DDQN": (58.23, 61.10, 70.27, 62.29),
    "PG": (24.20, 26.97, 34.60, 28.16), "A2C": (43.08, 46.30, 54.18, 47.02),
    "FHMM1": (63.50, 66.11, 79.47, 74.22), "FHMM2": (61.32, 63.96, 76.61, 70.41),
    "FHMM3": (60.62, 61.34, 75.42, 70.17),
}


def rule_cfg(args) -> RuleConfig:
    return RuleConfig(table1=args.table1, same_finger=args.same_finger)


# ---------------------------------------------------------------------------------------
# pieces / check-reward
# ---------------------------------------------------------------------------------------
def cmd_pieces(args):
    info = write_pieces_json(args.pig_dir)
    for env, m in info.items():
        print(f"{env}: PIG {m['pig_id']} {m['composer']}, {m['title']}: {m['right_hand_notes']} right-hand notes, "
              f"{m['chord_onsets']} chord onsets, {m['overlaps']} overlapping notes, annotators {m['annotators']}")
    v0 = load_piece("V0")
    print(f"V0: {v0.title}: {len(v0)} notes, {v0.chords} chord onsets, {v0.overlaps} overlaps (transcribed)")
    print("wrote data/paper/pieces.json")


def cmd_check_reward(args):
    from fingerly.paper.dp import optimal_fingering

    cfg = rule_cfg(args)
    print(f"rules: table1={cfg.table1}, same_finger={cfg.same_finger}\n")
    ok = True

    v0 = load_piece("V0")
    b = breakdown(v0.pitches, v0.labels[0], cfg)
    print(f"1. V0 composer's fingering: D = {b.sum():g} (must be 18): {format_breakdown(b)}")
    ok &= b.sum() == 18

    print("\n2. Human row of Table II (L = D / N), every annotator:")
    for i, env in enumerate(ENVS[1:], start=1):
        p = load_piece(env, args.pig_dir)
        paper = PAPER_BEST_L["Human"][i]
        print(f"   {env} (N = {len(p)}), paper L = {paper} (D = {paper * len(p):.2f})")
        for name, lab in zip(p.annotators, p.labels):
            print(f"      {name}: L = {level(p.pitches, lab, cfg):.4f}  D = {level(p.pitches, lab, cfg) * len(p):g}")

    print("\n3. Exact optimum by DP:")
    for env in ENVS:
        p = load_piece(env, args.pig_dir)
        D, _ = optimal_fingering(p.pitches, cfg)
        print(f"   {env}: D* = {D:g}, L* = {D / len(p):.4f}   (paper best DQN L = {PAPER_BEST_L['DQN'][ENVS.index(env)]})")
        if env == "V0" and D != 18:
            print("   !! V0 optimum differs from the paper's 18: same-finger handling is not the authors' (§A.7 item 2)")
            ok = False
    print("\nall checks passed" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


# ---------------------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------------------
_WORKER_DEVICE = "cpu"


def _init_worker(counter, n_gpu: int) -> None:
    """The first n_gpu workers of the pool train on the GPU, the others on the CPU. All of them
    take jobs from one queue, so a faster worker simply finishes more runs."""
    global _WORKER_DEVICE
    with counter.get_lock():
        idx = counter.value
        counter.value += 1
    _WORKER_DEVICE = "cuda" if idx < n_gpu else "cpu"


def _run_one(job: dict) -> dict:
    import torch

    torch.set_num_threads(1)
    from fingerly.paper.agents import PaperConfig, train
    from fingerly.paper.env import PaperFingeringEnv

    out = Path(job["out"])
    piece = load_piece(job["env"], job["pig_dir"])
    job["cfg"] = dict(job["cfg"], device=_WORKER_DEVICE)
    cfg = PaperConfig(**job["cfg"])
    rcfg = RuleConfig(**job["rules"])
    env = PaperFingeringEnv(piece.pitches, rcfg)
    res = train(job["algo"], env, cfg, seed=job["seed"])
    out.mkdir(parents=True, exist_ok=True)
    torch.save(res.pop("state_dict"), out / "final.pt")
    with open(out / "rewards.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "reward"])
        for i, r in enumerate(res["episode_rewards"]):
            w.writerow([i, r])
    res.update(env=job["env"], algo=job["algo"], seed=job["seed"], config=job["cfg"], rules=job["rules"],
               final_L=level(piece.pitches, res["final_fingering"], rcfg),
               best_episode_L=level(piece.pitches, res["best_episode_fingering"], rcfg))
    (out / "result.json").write_text(json.dumps(res))
    return {k: res[k] for k in ("env", "algo", "seed", "final_L", "best_episode_L", "seconds")} | {"device": cfg.device}


def cmd_train(args):
    envs = ENVS if args.env == "all" else [args.env]
    algos = ALGOS if args.algo == "all" else [args.algo]
    rules = asdict(rule_cfg(args))
    base = dict(episodes=args.episodes, gamma=args.gamma, lr=args.lr, buffer=args.buffer, batch=args.batch,
                target_sync=args.target_sync, eps_start=args.eps_start, eps_end=args.eps_end, eps_decay=args.eps_decay)
    jobs = []
    for env in envs:
        for algo in algos:
            for k in range(args.runs):
                out = Path(args.out) / env / algo / f"run{k}"
                if (out / "result.json").exists() and not args.overwrite:
                    continue
                jobs.append(dict(env=env, algo=algo, seed=args.seed_base + k, out=str(out), pig_dir=args.pig_dir,
                                 cfg=base, rules=rules))
    # slowest jobs first (long pieces, per-step updates) so the pool drains evenly
    size = {"V0": 29, "V1": 215, "V2": 136, "V3": 68}
    weight = {"dqn": 3, "ddqn": 3, "a2c": 2, "pg": 1}
    jobs.sort(key=lambda j: -size[j["env"]] * weight[j["algo"]])
    if not jobs:
        print("nothing to do (all runs exist; --overwrite to redo)")
        return
    gpu = args.gpu_workers if args.device == "cuda" else 0
    print(f"{len(jobs)} runs, {args.workers} CPU workers + {gpu} GPU workers", flush=True)
    t0 = time.time()
    done = 0
    import multiprocessing as mp

    # Each worker costs ~1.4 GB of committed memory (torch loads its CUDA libraries even on the
    # CPU), so keep --workers + --gpu-workers within what the machine can commit.
    pool = ProcessPoolExecutor(args.workers + gpu, initializer=_init_worker, initargs=(mp.Value("i", 0), gpu))
    futures = [pool.submit(_run_one, j) for j in jobs]
    failed = 0
    for fut in as_completed(futures):
        done += 1
        try:
            r = fut.result()
        except Exception as e:  # e.g. out of memory: keep the other runs going, rerun later (resumable)
            failed += 1
            print(f"[{done}/{len(jobs)}] FAILED: {type(e).__name__}: {e}", flush=True)
            continue
        print(f"[{done}/{len(jobs)} {(time.time() - t0) / 60:.1f} min] {r['env']} {r['algo']} seed {r['seed']} "
              f"({r['device']}): final L {r['final_L']:.4f}, best episode L {r['best_episode_L']:.4f}, "
              f"{r['seconds']:.0f}s", flush=True)
    pool.shutdown()
    if failed:
        print(f"{failed} runs failed; run the same command again to redo only those", flush=True)


# ---------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------
def _load_runs(root: Path):
    runs = {}
    for f in sorted(root.glob("*/*/run*/result.json")):
        r = json.loads(f.read_text())
        runs.setdefault((r["env"], r["algo"]), []).append(r)
    return runs


def cmd_report(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from fingerly.metrics import match_rates
    from fingerly.paper.dp import optimal_fingering

    cfg = rule_cfg(args)
    runs = _load_runs(Path(args.runs))
    if not runs:
        sys.exit(f"no runs under {args.runs}")
    pieces = {e: load_piece(e, args.pig_dir) for e in ENVS}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    key = "final_L" if args.policy == "final" else "best_episode_L"
    fkey = "final_fingering" if args.policy == "final" else "best_episode_fingering"
    L = lambda e, f: level(pieces[e].pitches, f, cfg)  # noqa: E731
    md = [f"# Part A results: Iman & Ahn (2024) reproduction\n",
          f"Rules: `--table1 {cfg.table1} --same-finger {cfg.same_finger}`. Policy scored: **{args.policy}** "
          f"({'greedy policy after training' if args.policy == 'final' else 'best training episode'}). "
          f"Runs per cell: {', '.join(f'{e}/{a}={len(v)}' for (e, a), v in sorted(runs.items()))}.\n",
          "Each cell is **ours / paper**.\n"]

    # Table II
    md += ["## Table II: best L over runs (lower is better)\n", "| Method | " + " | ".join(ENVS) + " |",
           "| --- |" + " --- |" * len(ENVS)]
    for a in ALGOS:
        cells = []
        for i, e in enumerate(ENVS):
            rs = runs.get((e, a), [])
            ours = f"{min(r[key] for r in rs):.4f}" if rs else "–"
            cells.append(f"{ours} / {PAPER_BEST_L[ALGO_LABEL[a]][i]}")
        md.append(f"| {ALGO_LABEL[a]} | " + " | ".join(cells) + " |")
    for name in ("FHMM1", "FHMM2", "FHMM3"):
        md.append(f"| {name} | " + " | ".join(f"– / {v}" for v in PAPER_BEST_L[name]) + " |")
    hum = []
    for i, e in enumerate(ENVS):
        Ls = [L(e, lab) for lab in pieces[e].labels]
        hum.append(f"{Ls[0]:.4f} (all: {', '.join(f'{x:.4f}' for x in Ls)}) / {PAPER_BEST_L['Human'][i]}")
    md.append("| Human (annotator 1) | " + " | ".join(hum) + " |")
    dp = {e: optimal_fingering(pieces[e].pitches, cfg) for e in ENVS}
    md.append("| Exact optimum (DP) | " + " | ".join(f"{dp[e][0] / len(pieces[e]):.4f} / –" for e in ENVS) + " |\n")

    # Tables III-IV
    md += ["## Tables III–IV: mean ± std of L over runs, and mean run time\n",
           "| Method | " + " | ".join(ENVS) + " | mean time (s) per run, V0..V3 |", "| --- |" + " --- |" * (len(ENVS) + 1)]
    for a in ALGOS:
        cells, times = [], []
        for i, e in enumerate(ENVS):
            rs = runs.get((e, a), [])
            if rs:
                v = np.array([r[key] for r in rs])
                cells.append(f"{v.mean():.4f} ± {v.std(ddof=1) if len(v) > 1 else 0:.3f} / {PAPER_MEAN_L[ALGO_LABEL[a]][i]}")
                times.append(f"{np.mean([r['seconds'] for r in rs]):.0f}")
            else:
                cells.append(f"– / {PAPER_MEAN_L[ALGO_LABEL[a]][i]}")
                times.append("–")
        md.append(f"| {ALGO_LABEL[a]} | " + " | ".join(cells) + f" | {', '.join(times)} |")
    md.append("")

    # Table V
    md += ["## Table V: match rates vs PIG annotations, averaged over V1–V3 (higher is better)\n",
           "Each algorithm row is the mean over its runs of the per-piece-averaged rates; "
           "`best` is the run with the lowest L on each piece.\n",
           "| Method | Mgen | Mhigh | Msoft | Mrec |", "| --- | --- | --- | --- | --- |"]
    per_piece_rows = []
    table5 = {}
    for a in ALGOS:
        allm, bestm = [], []
        for e in ENVS[1:]:
            rs = runs.get((e, a), [])
            if not rs:
                continue
            ms = [match_rates(r[fkey], pieces[e].labels) for r in rs]
            allm.append({k: np.mean([m[k] for m in ms]) for k in ms[0]})
            best = min(rs, key=lambda r: r[key])
            bm = match_rates(best[fkey], pieces[e].labels)
            bestm.append(bm)
            per_piece_rows.append(f"| {ALGO_LABEL[a]} | {e} | " + " | ".join(f"{100 * allm[-1][k]:.2f}" for k in ("Mgen", "Mhigh", "Msoft", "Mrec"))
                                  + " | " + " | ".join(f"{100 * bm[k]:.2f}" for k in ("Mgen", "Mhigh", "Msoft", "Mrec")) + " |")
        if not allm:
            continue
        mean = {k: 100 * np.mean([m[k] for m in allm]) for k in allm[0]}
        bmean = {k: 100 * np.mean([m[k] for m in bestm]) for k in bestm[0]}
        table5[a] = mean
        p = PAPER_MATCH[ALGO_LABEL[a]]
        md.append(f"| {ALGO_LABEL[a]} | " + " | ".join(f"{mean[k]:.2f} (best {bmean[k]:.2f}) / {p[i]:.2f}"
                                                    for i, k in enumerate(("Mgen", "Mhigh", "Msoft", "Mrec"))) + " |")
    dpm = [match_rates(dp[e][1], pieces[e].labels) for e in ENVS[1:]]
    md.append("| Exact optimum (DP) | " + " | ".join(f"{100 * np.mean([m[k] for m in dpm]):.2f} / –" for k in ("Mgen", "Mhigh", "Msoft", "Mrec")) + " |")
    for name in ("FHMM1", "FHMM2", "FHMM3"):
        md.append(f"| {name} | " + " | ".join(f"– / {v:.2f}" for v in PAPER_MATCH[name]) + " |")
    md += ["\nPer piece (mean over runs | best run):\n",
           "| Method | Piece | Mgen | Mhigh | Msoft | Mrec | best Mgen | best Mhigh | best Msoft | best Mrec |",
           "| --- | --- |" + " --- |" * 8] + per_piece_rows + [""]

    # Claims
    md.append("## Claims check (README §A.6)\n")
    for a, paper_ep in (("dqn", 124), ("ddqn", 202)):
        rs = runs.get(("V0", a), [])
        hits = [int(np.argmax(np.array(r["episode_rewards"]) >= -18)) for r in rs if max(r["episode_rewards"]) >= -18]
        final_opt = sum(abs(r["final_L"] * 29 - 18) < 1e-6 for r in rs)
        md.append(f"- **{ALGO_LABEL[a]} reaches the V0 optimum (−18):** {len(hits)}/{len(rs)} runs had a −18 training episode"
                  + (f" (first at episode {min(hits)}, median {int(np.median(hits))}; paper: {paper_ep})" if hits else "")
                  + f"; {final_opt}/{len(rs)} final greedy policies score 18.")
    for i, e in enumerate(ENVS):
        means = {a: np.mean([r[key] for r in runs[(e, a)]]) for a in ALGOS if (e, a) in runs}
        if len(means) == 4:
            holds = means["dqn"] <= means["ddqn"] < means["a2c"] < means["pg"]
            md.append(f"- **Ordering DQN ≤ DDQN < A2C < PG on {e}** (mean L): "
                      + ", ".join(f"{ALGO_LABEL[a]} {means[a]:.3f}" for a in ALGOS) + f" → {'holds' if holds else 'does NOT hold'}")
    if "dqn" in table5:
        md.append(f"- **DQN Mgen vs FHMM1:** ours {table5['dqn']['Mgen']:.2f}% vs paper's FHMM1 {PAPER_MATCH['FHMM1'][0]}% "
                  f"(paper's DQN {PAPER_MATCH['DQN'][0]}%) → gap {table5['dqn']['Mgen'] - PAPER_MATCH['FHMM1'][0]:+.2f} points.")
    md.append("\nFHMM1–3 were not run (Nakamura et al.'s programs are not in this repository); their numbers are the paper's.")
    md.append("V0 is a reconstruction from the paper's Fig. 5 (see `data/paper/V0_beyer_op101_no12_fingering.txt`).")
    (out / "tables.md").write_text("\n".join(md), encoding="utf-8")

    # Fig. 4
    fig, ax = plt.subplots(figsize=(7, 4))
    for a in ALGOS:
        rs = runs.get(("V0", a), [])
        if not rs:
            continue
        R = np.array([r["episode_rewards"] for r in rs])
        m, s = R.mean(0), R.std(0)
        x = np.arange(len(m))
        ax.plot(x, m, label=ALGO_LABEL[a])
        ax.fill_between(x, m - s, m + s, alpha=0.2)
    ax.axhline(-18, color="k", ls="--", lw=0.8, label="optimum (−18)")
    ax.set_xlabel("episode")
    ax.set_ylabel("episode reward")
    ax.set_title("V0: mean ± std episode reward")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig4_v0_curves.png", dpi=150)
    print("\n".join(md))
    print(f"\nwrote {out / 'tables.md'} and {out / 'fig4_v0_curves.png'}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # tables use ± and − (Windows consoles default to cp1252)
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = a.add_subparsers(dest="cmd", required=True)

    def common(p, pig_required=True):
        p.add_argument("--pig-dir", required=pig_required, default=None)
        p.add_argument("--table1", default="paper", choices=["paper", "parncutt"])
        p.add_argument("--same-finger", default="zero-span", choices=["zero-span", "free"])

    p = sub.add_parser("pieces")
    p.add_argument("--pig-dir", required=True)
    common(sub.add_parser("check-reward"))

    p = sub.add_parser("train")
    common(p, pig_required=False)
    p.add_argument("--env", default="V0", choices=list(ENVS) + ["all"])
    p.add_argument("--algo", default="dqn", choices=list(ALGOS) + ["all"])
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--out", default="runs/paper")
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--buffer", type=int, default=10_000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--target-sync", type=int, default=100)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.01)
    p.add_argument("--eps-decay", type=float, default=0.999)
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1),
                   help="CPU worker processes (each commits ~1.4 GB of memory)")
    p.add_argument("--gpu-workers", type=int, default=1, help="extra worker processes on the GPU (with --device cuda)")
    p.add_argument("--overwrite", action="store_true")

    p = sub.add_parser("report")
    common(p)
    p.add_argument("--runs", default="runs/paper")
    p.add_argument("--out", default="results/paper")
    p.add_argument("--policy", default="final", choices=["final", "best"],
                   help="score each run's final greedy policy (default) or its best training episode (§A.7 item 9)")

    args = a.parse_args()
    if args.cmd == "train" and args.env != "V0" and not args.pig_dir:
        a.error("--pig-dir is needed for V1-V3")
    return {"pieces": cmd_pieces, "check-reward": cmd_check_reward, "train": cmd_train,
            "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main() or 0)
