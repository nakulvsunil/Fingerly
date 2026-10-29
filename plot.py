"""Learning curves for one or more runs (e.g. several seeds).

    python plot.py runs/rh_ddqn_s0 runs/rh_ddqn_s1 runs/rh_ddqn_s2 --out results/curves.png
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
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



# ======================================================================================
# Part C: figures and tables of README §C.11.10
# ======================================================================================
TWOHAND_FLAGS = ("--curves", "--success", "--regret", "--seen-unseen", "--seeds", "--lambda-curve", "--tables",
                 "--r4", "--sensitivity", "--excerpts")
TWOHAND_PLOT_DOC = """Part C figures and tables (README §C.11.10).

    python plot.py --curves runs/ --threshold-gap 0.15 --out results/twohand/curves.png      # 1, 5
    python plot.py --success runs/ --out results/twohand/success.png                        # 3
    python plot.py --regret results/twohand/main.json --out results/twohand/regret.png      # 4
    python plot.py --seen-unseen results/twohand/main.json --out results/twohand/generalisation.png  # 6
    python plot.py --seeds results/twohand/main.json --out results/twohand/seeds.png        # 7
    python plot.py --lambda-curve results/twohand/main.json --paper-points data/twohand/eer2025_table5.json --out results/twohand/curve.png  # 9
    python plot.py --r4 results/twohand/main.json --out results/twohand/r4.png              # 10
    python plot.py --sensitivity results/twohand/sensitivity.json --out results/twohand/sensitivity.png  # 13
    python plot.py --tables results/twohand/main.json results/twohand/robustness.json --out results/twohand/tables.md  # 2, 7, 8, 11, 12
    python plot.py --excerpts runs/r3_lam0.5_s0/best.pt runs/s1_s0/best.pt --pig-dir data/PianoFingeringDataset_v1.2 --out results/twohand/excerpts.md  # 14
"""

GROUP_SKIP = ("smoke", "bench")


def _run_groups(root: str):
    """{method: [run dirs]} for every run folder with an eval_log.csv (seeds grouped by _s<seed>)."""
    import re

    groups = {}
    for f in sorted(Path(root).rglob("eval_log.csv")):
        d = f.parent
        if any(s in d.parts for s in GROUP_SKIP) and not root.rstrip("/\\").endswith(GROUP_SKIP):
            continue
        groups.setdefault(re.sub(r"_s\d+$", "", d.name), []).append(d)
    return groups


def _curve(d: Path, key: str):
    rows = read_csv(d / "eval_log.csv")
    if not rows or "step" not in rows[0]:
        return [], []
    return col(rows, key)


def _band(ax, runs, key, label, every=None):
    """Mean over seeds with a 95% bootstrap band, on the union of evaluation steps."""
    import numpy as np

    curves = [_curve(d, key) for d in runs]
    curves = [c for c in curves if c[0]]
    if not curves:
        return None
    steps = sorted({s for xs, _ in curves for s in xs})
    mat = np.array([np.interp(steps, xs, ys, left=np.nan, right=np.nan) for xs, ys in curves])
    mean = np.nanmean(mat, 0)
    line, = ax.plot(steps, mean, label=f"{label} (n={len(curves)})", lw=1.4)
    if len(curves) > 1:
        rng = np.random.default_rng(0)
        boots = np.array([np.nanmean(mat[rng.integers(0, len(curves), len(curves))], 0) for _ in range(500)])
        ax.fill_between(steps, np.nanquantile(boots, 0.025, 0), np.nanquantile(boots, 0.975, 0),
                        color=line.get_color(), alpha=0.2)
    return steps, mean


def _keep(name: str, include) -> bool:
    return (not include or any(name.startswith(i) for i in include)) and not name.startswith("sens_")


def plot_curves(root, out, threshold_gap=0.15, include=None):
    """Figure 1: mean validation return per event against steps (95% band over seeds), all RL
    methods; figure 5: validation gap to M0 with the threshold line and N* marked."""
    from fingerly.metrics import first_sustained

    groups = {k: v for k, v in _run_groups(root).items() if _keep(k, include)}
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, runs in groups.items():
        _band(ax, runs, "return_per_event", name)
    ax.set(xlabel="environment steps", ylabel="validation return per event (rule reward)",
           title="Learning curves (mean over seeds, 95% bootstrap band)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, runs in groups.items():
        res = _band(ax, runs, "gap", name)
        if res:
            n = first_sustained(res[0], res[1], lambda g: g <= threshold_gap)
            if n is not None:
                ax.axvline(n, color=ax.lines[-1].get_color(), ls=":", lw=1)
    ax.axhline(threshold_gap, color="k", ls="--", lw=1, label=f"threshold {threshold_gap}")
    ax.set(xlabel="environment steps", ylabel="validation gap to the optimum (agent - M0) / M0", yscale="log",
           title="Convergence: N* = first step with gap <= threshold for 3 evaluations (dotted)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    conv = Path(out).with_name(Path(out).stem + "_convergence.png")
    fig.savefig(conv, dpi=150)
    print(f"wrote {out} and {conv}")


def plot_success(root, out, include=None):
    """Figure 3: success rate on validation against steps."""
    groups = {k: v for k, v in _run_groups(root).items() if _keep(k, include)}
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, runs in groups.items():
        _band(ax, runs, "SR", name)
    ax.set(xlabel="environment steps", ylabel="validation success rate (no infeasible move, gap <= 10%)",
           title="Success rate during training")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def _main_methods(js, include_ablations=False):
    return {k: v for k, v in js["methods"].items()
            if include_ablations or not (k.startswith("abl_") or k.startswith("sens_"))}


def plot_regret(main_json, out):
    """Figure 4: cumulative regret over the test segments, oracle at 0."""
    js = json.loads(Path(main_json).read_text())
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, m in _main_methods(js).items():
        if m.get("cumulative_regret"):
            ax.plot(m["cumulative_regret"], label=name, lw=1.3)
    ax.axhline(0, color="k", lw=1.5, label="oracle (M0)")
    ax.set(xlabel="test segment (piece order)", ylabel="cumulative regret (agent cost - M0 cost)",
           title="Cumulative regret on the test segments (first seed)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def plot_seen_unseen(main_json, out):
    """Figure 6: generalisation, seen (training) vs unseen (test) pieces, per method."""
    import numpy as np

    js = json.loads(Path(main_json).read_text())
    ms = {k: v["generalisation"] for k, v in _main_methods(js).items() if v.get("generalisation")}
    keys = ("return_per_event", "Mgen", "d_rel_balliauw15")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, k in zip(axes, keys):
        names = [n for n in ms if k in ms[n]]
        x = np.arange(len(names))
        ax.bar(x - 0.2, [ms[n][k]["seen"] for n in names], 0.4, label="seen (training pieces)")
        ax.bar(x + 0.2, [ms[n][k]["unseen"] for n in names], 0.4, label="unseen (test pieces)")
        ax.set_xticks(x, names, rotation=60, ha="right", fontsize=7)
        ax.set_title(k)
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=8)
    fig.suptitle("Generalisation gap: J_seen - J_unseen, no imitation bonus")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def plot_seeds(main_json, out):
    """Figure 7: per-seed final test scores (box plots) for every method with several seeds."""
    js = json.loads(Path(main_json).read_text())
    by = {}
    for r in js["runs"]:
        if r["method"].startswith(("abl_", "sens_")):
            continue
        by.setdefault(r["method"], []).append(r["summary"])
    by = {k: v for k, v in by.items() if len(v) > 1}
    if not by:
        print("no method has more than one seed: no seed box plot")
        return
    keys = ("return_per_event", "Mgen", "d_rel_balliauw15")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, k in zip(axes, keys):
        names = list(by)
        ax.boxplot([[s.get(k) for s in by[n] if s.get(k) is not None] for n in names], tick_labels=names)
        ax.tick_params(axis="x", rotation=60, labelsize=7)
        ax.set_title(k)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Stability across seeds (final test scores)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def plot_lambda_curve(main_json, paper_json, out):
    """Figure 9: d_rel (balliauw15) against Mgen for every method, with the 2025 paper's points."""
    js = json.loads(Path(main_json).read_text())
    fig, ax = plt.subplots(figsize=(8, 6))
    pts = []
    for name, m in _main_methods(js).items():
        if "d_rel_balliauw15" in m and "Mgen" in m:
            x, y = m["d_rel_balliauw15"]["mean"], m["Mgen"]["mean"]
            ax.scatter(x, y, s=30)
            ax.annotate(name, (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
            if name.startswith("r2_lam") or name == "r1":
                lam = 0.0 if name == "r1" else float(name.split("lam")[1].split("_")[0])
                pts.append((lam, x, y))
    if pts:
        pts.sort()
        ax.plot([p[1] for p in pts], [p[2] for p in pts], "-", lw=1, color="C0", label="λ curve (R1, R2)")
    if paper_json and Path(paper_json).exists():
        t5 = json.loads(Path(paper_json).read_text())["rows"]
        hd = t5["Human"]["difficulty"]
        for k, v in t5.items():
            ax.scatter(v["difficulty"] / hd, v["Mgen"], marker="x", color="k")
            ax.annotate(f"2025 {k}", (v["difficulty"] / hd, v["Mgen"]), fontsize=7, xytext=(3, -9),
                        textcoords="offset points")
    ax.axvline(0.81, color="r", ls=":", lw=1)
    ax.axhline(0.62, color="r", ls=":", lw=1, label="target: d_rel <= 0.81, Mgen >= 0.62")
    ax.set(xlabel="d_rel = difficulty / pianists' difficulty (balliauw15)", ylabel="Mgen",
           title="Difficulty against match rate (ours: unseen pieces; 2025: fitted per test piece)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def plot_r4(main_json, out):
    """Figure 10: R4 fine-tuning curve, d_rel against episodes, with the 2025 paper's 500 episodes."""
    import numpy as np

    js = json.loads(Path(main_json).read_text())
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, m in js["methods"].items():
        for rl in m.get("rl", []):
            curves = rl.get("d_rel_curves") if isinstance(rl, dict) else None
            if not curves:
                continue
            mat = np.array([c for c in curves.values()], float)
            ax.plot(range(mat.shape[1]), mat.mean(0), lw=2, label=f"{name}: mean over {mat.shape[0]} pieces")
            for c in mat:
                ax.plot(range(len(c)), c, lw=0.5, alpha=0.25, color="grey")
    ax.axhline(0.81, color="r", ls=":", label="d_rel 0.81 (2025 Uniform / EER)")
    ax.axvline(500, color="k", ls="--", label="2025 paper: 500 episodes per piece")
    ax.set(xscale="symlog", xlabel="fine-tuning episodes on the test piece", ylabel="d_rel (balliauw15)",
           title="R4: fine-tuning on each test piece")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def plot_sensitivity(sens_json, out):
    """Figure 13: each validation metric against each setting (one seed, 500k steps)."""
    js = json.loads(Path(sens_json).read_text())
    settings = {}
    for name, m in js["methods"].items():
        if not name.startswith("sens_"):
            continue
        _, setting, value = name.split("_", 2)
        settings.setdefault(setting, []).append((value, m))
    keys = ("Mgen", "d_rel_balliauw15", "SR", "infeasible_rate")
    fig, axes = plt.subplots(len(keys), max(1, len(settings)), figsize=(3.2 * max(1, len(settings)), 2.6 * len(keys)),
                             squeeze=False)
    for j, (setting, vals) in enumerate(sorted(settings.items())):
        vals.sort(key=lambda v: float(v[0].split("-")[0]) if v[0].replace(".", "").replace("-", "").isdigit() else 0)
        for i, k in enumerate(keys):
            ax = axes[i][j]
            ax.plot([v[0] for v in vals], [v[1].get(k, {}).get("mean", float("nan")) for v in vals], "o-")
            if i == 0:
                ax.set_title(setting)
            if j == 0:
                ax.set_ylabel(k)
            ax.grid(alpha=0.3)
    fig.suptitle("Sensitivity (validation, 1 seed, 500k steps)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


# ---------------------------------------------------------------- tables
def _fmt(m, k, pct=False, digits=3):
    v = m.get(k)
    if not isinstance(v, dict):
        return "–"
    f = (lambda x: f"{100 * x:.1f}") if pct else (lambda x: f"{x:.{digits}f}")
    return f(v["mean"]) + (f" ± {f(v['std'])}" if v.get("std") else "")


def write_tables(main_json, robust_json, out):
    """Tables 2 (final performance), 7 (seed stability), 8 (compute), 11 (ablations),
    12 (robustness), plus the 2025 comparison, Wilcoxon tests and the success criteria."""
    js = json.loads(Path(main_json).read_text())
    M = js["methods"]
    L = ["# Part C results (README §C.11.10)", ""]
    main = {k: v for k, v in M.items() if not k.startswith(("abl_", "sens_"))}
    L += ["## Final performance on the 30 test pieces", "",
          "Mean ± std over seeds. Return = rule reward per event (no bonus). SR = segments with no infeasible "
          "transition and gap to M0 ≤ 10% (pooled over seeds, Wilson 95% interval). Rates in %.", "",
          "| Method | Mode | Seeds | Return / event | Median | SR (95% CI) | N* (gap ≤ 15%) | Mgen | Mhigh | Msoft | Mrec | "
          "Hand acc. | d_rel | Diff./note p12 | Gap to M0 | Infeasible | Large stretch | Time-leap | Crossing | Hand switch | Weak finger | Thumb on black |",
          "|" + " --- |" * 22]
    for n, m in main.items():
        sr = f"{100 * m['SR_pooled']:.1f} ({100 * m['SR_wilson95'][0]:.1f}–{100 * m['SR_wilson95'][1]:.1f})" \
            if "SR_pooled" in m else _fmt(m, "SR", True)
        nstar = m.get("N_star_gap15")
        nstar = ", ".join("–" if x is None else f"{x / 1000:.0f}k" for x in nstar) if nstar else "–"
        L.append(f"| {n} | {m.get('mode', '')} | {m.get('n', 1)} | {_fmt(m, 'return_per_event')} | "
                 f"{m.get('return_per_event', {}).get('median', float('nan')):.3f} | {sr} | {nstar} | "
                 f"{_fmt(m, 'Mgen', True)} | {_fmt(m, 'Mhigh', True)} | {_fmt(m, 'Msoft', True)} | {_fmt(m, 'Mrec', True)} | "
                 f"{_fmt(m, 'hand_accuracy', True)} | {_fmt(m, 'd_rel_balliauw15')} | {_fmt(m, 'difficulty_per_note_parncutt12', digits=2)} | "
                 f"{_fmt(m, 'gap')} | {_fmt(m, 'infeasible_rate', True)} | {_fmt(m, 'large_stretch_rate', True)} | "
                 f"{_fmt(m, 'time_leap_rate', True)} | {_fmt(m, 'crossing_rate', True)} | {_fmt(m, 'hand_switch_rate', True)} | "
                 f"{_fmt(m, 'weak_finger_rate', True)} | {_fmt(m, 'thumb_on_black_rate', True)} |")
    L += ["", "Per hand and singles vs chords (Mgen, %):", "",
          "| Method | Right | Left | Single notes | Chords | Chord exact match | Unreachable chords | Sequence breakdowns |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for n, m in main.items():
        L.append(f"| {n} | {_fmt(m, 'Mgen_R', True)} | {_fmt(m, 'Mgen_L', True)} | {_fmt(m, 'Mgen_single', True)} | "
                 f"{_fmt(m, 'Mgen_chord', True)} | {_fmt(m, 'chord_exact', True)} | {_fmt(m, 'unreachable_chord_rate', True)} | "
                 f"{_fmt(m, 'sequence_breakdown_rate', True)} |")
    # 2025 comparison
    if "paper_2025" in js:
        L += ["", "## Comparison with Iman & Ahn (2025), README §C.11.4", "",
              "Their numbers are fitted to each test piece; ours are on unseen pieces. d_rel = difficulty ÷ the pianists' "
              "difficulty, `balliauw15`.", "", "| Method | d_rel | Mgen | Mhigh | Msoft | Mrec |", "| --- | --- | --- | --- | --- | --- |"]
        for k, v in js["paper_2025"].items():
            L.append(f"| 2025 {k} | {v['d_rel']:.2f} | {v['Mgen']:.2f} | {v['Mhigh']:.2f} | {v['Msoft']:.2f} | {v['Mrec']:.2f} |")
        for n, m in main.items():
            if "Mgen" in m and "d_rel_balliauw15" in m:
                L.append(f"| {n} | {m['d_rel_balliauw15']['mean']:.2f} | {m['Mgen']['mean']:.2f} | "
                         f"{m.get('Mhigh', {}).get('mean', float('nan')):.2f} | {m.get('Msoft', {}).get('mean', float('nan')):.2f} | "
                         f"{m.get('Mrec', {}).get('mean', float('nan')):.2f} |")
    # RL-specific
    L += ["", "## RL-specific metrics (validation logs)", "",
          "AUC = area under the validation return curve to 1M steps, scaled so random-valid = 0 and M0 = 1.", "",
          "| Method | AUC (mean) | Return at 100k | 250k | 500k | 1M | N* Mgen ≥ 0.60 | Generalisation gap (return / Mgen / d_rel) | IQM Mgen [95% CI] |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    import numpy as np

    for n, m in main.items():
        rl = [x for x in m.get("rl", []) if isinstance(x, dict) and "curve" in x]
        if not rl:
            continue
        at = lambda k: np.nanmean([x.get(k) if x.get(k) is not None else np.nan for x in rl])  # noqa: E731
        g = m.get("generalisation", {})
        gen = " / ".join(f"{g[k]['gap']:.3f}" if k in g else "–" for k in ("return_per_event", "Mgen", "d_rel_balliauw15"))
        iq = m.get("Mgen_iqm")
        iqs = f"{iq['iqm']:.3f} [{iq['lo']:.3f}, {iq['hi']:.3f}]" if iq else "–"
        nm = m.get("N_star_mgen60")
        nm = ", ".join("–" if x is None else f"{x / 1000:.0f}k" for x in nm) if nm else "–"
        L.append(f"| {n} | {np.nanmean([x.get('auc_1M', np.nan) for x in rl]):.3f} | {at('val_return_at_100k'):.3f} | "
                 f"{at('val_return_at_250k'):.3f} | {at('val_return_at_500k'):.3f} | {at('val_return_at_1000k'):.3f} | {nm} | {gen} | {iqs} |")
    # stability
    L += ["", "## Stability across seeds (final test scores)", "",
          "| Method | Seeds | Return: std / CV / min–max | Mgen: std / CV / min–max | d_rel: std / CV / min–max |",
          "| --- | --- | --- | --- | --- |"]
    for n, m in main.items():
        cells = []
        for k in ("return_per_event", "Mgen", "d_rel_balliauw15"):
            s = m.get(f"stability_{k}")
            cells.append(f"{s['std']:.4f} / {s['cv']:.3f} / {s['min']:.3f}–{s['max']:.3f}" if s else "–")
        if any(c != "–" for c in cells):
            L.append(f"| {n} | {m.get('n', 1)} | " + " | ".join(cells) + " |")
    # compute
    L += ["", "## Compute", "", "Hardware: i7-12650H, RTX 3050 laptop GPU. The 2025 paper used about 45 GPU-days.", "",
          "| Method | Device | Train min / run | CPU-hours (all seeds) | Steps/s | Params | Peak RAM (MB) | Inference ms/note | s/piece | Energy Wh (45 W est.) |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    total_h = 0.0
    for n, m in M.items():
        c = m.get("compute")
        inf = m.get("inference_ms_per_note", {}).get("mean")
        if not c and inf is None:
            continue
        if c:
            total_h += c["cpu_hours_total"]
        L.append(f"| {n} | {c['device'] if c else '–'} | {c['train_minutes_mean']:.1f} | {c['cpu_hours_total']:.2f} | "
                 f"{c['steps_per_sec']:.0f} | {m.get('params') or '–'} | {c['peak_ram_mb']:.0f} | "
                 f"{inf if inf is not None else float('nan'):.3f} | {m.get('inference_s_per_piece', {}).get('mean', float('nan')):.2f} | "
                 f"{c['energy_wh_estimate_45w']:.1f} |" if c else
                 f"| {n} | – | – | – | – | {m.get('params') or '–'} | – | {inf:.3f} | "
                 f"{m.get('inference_s_per_piece', {}).get('mean', float('nan')):.2f} | – |")
    L.append(f"\nTotal training time of the evaluated runs: {total_h:.1f} hours.")
    # ablations
    abl = {k: v for k, v in M.items() if k.startswith("abl_")}
    if abl:
        full = next((k for k in main if k.startswith("r3_")), None)
        L += ["", "## Ablations (one component removed from the full method)", "",
              f"Full method: {full}. Wilcoxon over test pieces against the full method (p < 0.05 marked *).", "",
              "| Variant | Mgen | d_rel | Diff./note p12 | Infeasible | Time-leap | Crossing | Hand acc. | Gap | p (Mgen) | p (d_rel) |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        wil = js.get("wilcoxon", {})

        def p_of(metric, a, b):
            for r in wil.get(metric, []):
                if {r["a"], r["b"]} == {a, b}:
                    return f"{r['p']:.3g}{'*' if r.get('significant') else ''}"
            return "–"

        for n, m in ([(full, M[full])] if full else []) + sorted(abl.items()):
            L.append(f"| {n} | {_fmt(m, 'Mgen', True)} | {_fmt(m, 'd_rel_balliauw15')} | {_fmt(m, 'difficulty_per_note_parncutt12', digits=2)} | "
                     f"{_fmt(m, 'infeasible_rate', True)} | {_fmt(m, 'time_leap_rate', True)} | {_fmt(m, 'crossing_rate', True)} | "
                     f"{_fmt(m, 'hand_accuracy', True)} | {_fmt(m, 'gap')} | "
                     f"{p_of('Mgen', n, full) if n != full else ''} | {p_of('d_rel_balliauw15', n, full) if n != full else ''} |")
    # robustness
    if robust_json and Path(robust_json).exists():
        R = json.loads(Path(robust_json).read_text())["methods"]
        L += ["", "## Robustness (frozen policy): degradation Δ = clean − perturbed", "",
              "Under transposition and hand size, Mgen still uses the original annotations, although a pianist might "
              "finger the changed passage differently.", "",
              "| Method | Perturbation | Δ d_rel | Δ Mgen | Δ SR | Δ infeasible | d_rel | Mgen | SR | infeasible |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for n, m in R.items():
            for rob in m.get("robustness", [])[:1]:
                for name, v in rob.items():
                    if name == "era":
                        continue
                    d = v.get("delta", {})
                    f = lambda x: "–" if x is None else f"{x:+.3f}"  # noqa: E731
                    g = lambda x: "–" if x is None else f"{x:.3f}"  # noqa: E731
                    L.append(f"| {n} | {name} | {f(d.get('d_rel_balliauw15'))} | {f(d.get('Mgen'))} | {f(d.get('SR'))} | "
                             f"{f(d.get('infeasible_rate'))} | {g(v.get('d_rel_balliauw15'))} | {g(v.get('Mgen'))} | "
                             f"{g(v.get('SR'))} | {g(v.get('infeasible_rate'))} |")
                if "era" in rob:
                    L += ["", f"{n}, by composer era (clean test results):", "", "| Era | Pieces | Mgen | d_rel |",
                          "| --- | --- | --- | --- |"]
                    for e, v in rob["era"].items():
                        L.append(f"| {e} | {v['pieces']} | {v['Mgen']:.3f} | {v['d_rel_balliauw15']:.3f} |")
                    L.append("")
    # Wilcoxon highlights
    if "wilcoxon" in js:
        L += ["", "## Wilcoxon signed-rank tests over the test pieces (main methods)", "",
              "| Metric | A | B | Pieces | mean A | mean B | p | Better |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for metric, rows in js["wilcoxon"].items():
            for r in rows:
                if r["a"] in main and r["b"] in main:
                    L.append(f"| {metric} | {r['a']} | {r['b']} | {r['pieces']} | {r['mean_a']:.3f} | {r['mean_b']:.3f} | "
                             f"{r['p']:.3g}{'*' if r.get('significant') else ''} | {r.get('better', '–')} |")
    Path(out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out}")


# ---------------------------------------------------------------- excerpts (figure 14)
def write_excerpts(agent_ckpt, copy_ckpt, pig_dir, out, n=3):
    """Score excerpts on the 2025 paper's failure cases (large leaps, held notes): notes, the
    pianist's fingers, the agent's and the copy-only model's, with the hand (L/R)."""
    from fingerly.dqn import Agent
    from fingerly.metrics import rollout
    from fingerly.twohand.actions import fingers_of
    from fingerly.twohand.env import SegmentPool
    from fingerly.twohand.prepare import Context, PrepConfig

    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    agents = {"agent": Agent.load(agent_ckpt), "copy-only (S1)": Agent.load(copy_ckpt)}
    meta = agents["agent"].extra
    ctx = Context(pig_dir, PrepConfig(**meta.get("prep", {})))
    from evaluate import _env_cfg

    pool = SegmentPool(ctx.split.test, _env_cfg(meta["env"], ctx.v_max), bonus=False)
    eps = {k: {(e.info.seg.piece, e.info.seg.index): e for e in rollout(pool, a.greedy())} for k, a in agents.items()}

    def windows(info):
        ev = info.seg.events
        out_w = []
        for j in range(1, len(info.order)):
            a, b = ev[info.order[j - 1]], ev[info.order[j]]
            leap = abs(b.pitches[0] - a.pitches[0]) if a.hand == b.hand else 0
            held = any(u > -1e9 for u in b.hold_until)
            out_w.append((leap, held, j))
        return out_w

    cases = {"large leaps": [], "held notes": []}
    for info in pool.infos:
        for leap, held, j in windows(info):
            if leap >= 12:
                cases["large leaps"].append((leap, info, j))
            if held:
                cases["held notes"].append((1, info, j))
    L = ["# Score excerpts on the 2025 paper's failure cases (README §C.11.10, item 14)", "",
         "Each cell is hand+fingers (R135 = right hand, fingers 1-3-5 from the lowest note).", ""]
    for case, items in cases.items():
        items.sort(key=lambda x: -x[0])
        seen = set()
        k = 0
        for _, info, j in items:
            key = (info.seg.piece, info.seg.index, j // 8)
            if key in seen:
                continue
            seen.add(key)
            lo, hi = max(0, j - 4), min(len(info.order), j + 4)
            idx = info.order[lo:hi]
            L += [f"## {case}: piece {info.seg.piece}, segment {info.seg.index}, events {lo}-{hi - 1}", ""]
            L.append("| | " + " | ".join("+".join(names[p % 12] + str(p // 12 - 1) for p in info.seg.events[i].pitches)
                                         for i in idx) + " |")
            L.append("|" + " --- |" * (len(idx) + 1))
            row = []
            for i in idx:
                lab = info.seg.events[i].labels[0] if info.seg.events[i].labels else ()
                row.append("".join(sorted({h or "?" for h, _ in lab})) + "".join(str(f) for _, f in lab))
            L.append("| pianist 1 | " + " | ".join(row) + " |")
            for name, eset in eps.items():
                ep = eset[(info.seg.piece, info.seg.index)]
                recs = {r.event: r for r in ep.records}
                L.append(f"| {name} | " + " | ".join(
                    recs[i].hand + "".join(map(str, fingers_of(recs[i].set, recs[i].hand))) for i in idx) + " |")
            L.append("")
            k += 1
            if k >= n:
                break
    Path(out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out}")


def twohand_plot_main(argv=None):
    ap = argparse.ArgumentParser(description=TWOHAND_PLOT_DOC, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--curves", help="runs root: learning and convergence curves")
    ap.add_argument("--threshold-gap", type=float, default=0.15)
    ap.add_argument("--success", help="runs root: success rate against steps")
    ap.add_argument("--regret", help="main.json")
    ap.add_argument("--seen-unseen", help="main.json")
    ap.add_argument("--seeds", help="main.json")
    ap.add_argument("--lambda-curve", help="main.json")
    ap.add_argument("--paper-points", default="data/twohand/eer2025_table5.json")
    ap.add_argument("--r4", help="main.json")
    ap.add_argument("--sensitivity", help="sensitivity.json")
    ap.add_argument("--tables", nargs="+", help="main.json [robustness.json]")
    ap.add_argument("--excerpts", nargs=2, metavar=("AGENT_CKPT", "COPY_CKPT"))
    ap.add_argument("--pig-dir", default="data/PianoFingeringDataset_v1.2")
    ap.add_argument("--include", nargs="*", default=None, help="method name prefixes to plot")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if args.curves:
        plot_curves(args.curves, args.out, args.threshold_gap, args.include)
    if args.success:
        plot_success(args.success, args.out, args.include)
    if args.regret:
        plot_regret(args.regret, args.out)
    if args.seen_unseen:
        plot_seen_unseen(args.seen_unseen, args.out)
    if args.seeds:
        plot_seeds(args.seeds, args.out)
    if args.lambda_curve:
        plot_lambda_curve(args.lambda_curve, args.paper_points, args.out)
    if args.r4:
        plot_r4(args.r4, args.out)
    if args.sensitivity:
        plot_sensitivity(args.sensitivity, args.out)
    if args.tables:
        write_tables(args.tables[0], args.tables[1] if len(args.tables) > 1 else None, args.out)
    if args.excerpts:
        write_excerpts(args.excerpts[0], args.excerpts[1], args.pig_dir, args.out)


if __name__ == "__main__":
    if any(x.split("=")[0] in TWOHAND_FLAGS for x in sys.argv[1:]):
        twohand_plot_main()  # Part C (README §C.11.10)
    else:
        main()
