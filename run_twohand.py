"""Run every Part C experiment, E0-E12, in the three rounds of README §C.12, then evaluate and plot.

    python run_twohand.py --pig-dir data/PianoFingeringDataset_v1.2

Round 1: E0 checks and M0, E1 S1/S2, E2 R1, E3 R2 lambda sweep, E8 benchmarks.
         -> best two lambdas, chosen on validation
Round 2: E4 R3 at the best two lambdas, E5 A1-A3, E9 extra seeds.
         -> best R configuration (R2 or R3 at the best lambda), chosen on validation
Round 3: E6 R4, E7 2c, E10 ablations, E12 sensitivity; then E11 robustness, the main
         evaluation on the test pieces, and every figure and table of §C.11.10.

At most --jobs processes run at once, at most --gpu-jobs of them on the GPU, and a job starts only
when enough RAM is free. A job whose run folder holds done.json is skipped; an interrupted job
resumes from its checkpoint; a failed job is retried. Progress goes to results/twohand/progress.log.
Selections use validation only; the test pieces are used only in the final evaluation, and by R4 and
tabular Q-learning, which are defined per test piece with the rule reward and no labels (§C.9).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

PY = sys.executable
RES = Path("results/twohand")
RUNS = Path("runs")
LAMBDAS = (0.25, 0.5, 1.0, 2.0)


@dataclass
class Job:
    name: str
    exp: str
    cmd: List[str]
    done: Path  # marker file written when the job finished
    gpu: bool = False
    tries: int = 0
    proc: Optional[subprocess.Popen] = None
    started: float = 0.0
    log: Optional[Path] = None
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------- utilities
def log(msg: str):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    with open(RES / "progress.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def free_ram_gb() -> float:
    try:
        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return min(m.ullAvailPhys, m.ullAvailPageFile) / 2**30
    except Exception:  # noqa: BLE001
        return 99.0


def run_jobs(jobs: List[Job], max_jobs: int, max_gpu: int, min_free_gb: float, retries: int = 2):
    """Run jobs in parallel, respecting the job/GPU/RAM limits; skip finished ones; retry failures."""
    todo = [j for j in jobs if not j.done.exists()]
    skipped = len(jobs) - len(todo)
    if skipped:
        log(f"  {skipped} finished jobs skipped")
    running: List[Job] = []
    failed: List[Job] = []
    while todo or running:
        for j in list(running):
            rc = j.proc.poll()
            if rc is None:
                continue
            running.remove(j)
            j.proc = None
            mins = (time.time() - j.started) / 60
            if rc == 0 and j.done.exists():
                log(f"  done  {j.exp:<4} {j.name} ({mins:.1f} min)")
            else:
                j.tries += 1
                log(f"  FAIL  {j.exp:<4} {j.name} (exit {rc}, try {j.tries}; see {j.log})")
                (todo if j.tries <= retries else failed).append(j)
        started = False
        for j in list(todo):
            if len(running) >= max_jobs:
                break
            if j.gpu and sum(r.gpu for r in running) >= max_gpu:
                continue
            if running and free_ram_gb() < min_free_gb:
                break
            j.log = RUNS / "logs" / f"{j.name}.log"
            j.log.parent.mkdir(parents=True, exist_ok=True)
            j.proc = subprocess.Popen(j.cmd, stdout=open(j.log, "a"), stderr=subprocess.STDOUT,
                                      env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            j.started = time.time()
            running.append(j)
            todo.remove(j)
            log(f"  start {j.exp:<4} {j.name} ({'gpu' if j.gpu else 'cpu'}; {len(running)} running, "
                f"{free_ram_gb():.1f} GB free)")
            started = True
            time.sleep(5)  # stagger start-up (data loading, CUDA context)
        if not started:
            time.sleep(10)
    for j in failed:
        log(f"  GAVE UP {j.exp} {j.name} after {j.tries} tries")
    return failed


# ---------------------------------------------------------------- job builders
class Plan:
    def __init__(self, a):
        self.a = a
        self.P = ["--pig-dir", a.pig_dir]

    def train(self, name, exp, args: List[str], seed: int, steps: Optional[int] = None, gpu: bool = True) -> Job:
        out = RUNS / f"{name}_s{seed}"
        cmd = [PY, "train.py", *self.P, *args, "--seed", str(seed), "--out", str(out),
               "--device", self.a.device if gpu else "cpu"]
        if steps:
            cmd += ["--steps", str(steps)]
        return Job(f"{name}_s{seed}", exp, cmd, out / "done.json", gpu=gpu and self.a.device == "cuda")

    def given(self, lam, elite=False):
        return ["--mode", "given", "--cost", "parncutt12", "--lam", f"{lam:g}"] + (["--elite", "1"] if elite else [])

    def evaluate(self, name, exp, args: List[str], out: Path) -> Job:
        cmd = [PY, "evaluate.py", *self.P, *args, "--workers", str(self.a.eval_workers), "--out", str(out)]
        return Job(name, exp, cmd, out)


def round1(p: Plan) -> List[Job]:
    J = []
    # E0: reward checks, coverage, M0 on validation and test
    J.append(Job("E0_coverage", "E0", [PY, "prepare_twohand.py", *p.P, "--report", str(RES / "coverage.md")],
                 RES / "coverage.md"))
    J.append(Job("E0_check_balliauw15", "E0", [PY, "check_reward.py", *p.P, "--cost", "balliauw15", "--table-a1",
                                               "data/twohand/eer2025_table_a1.csv", "--out", str(RES / "reward_checks_table_a1.md")],
                 RES / "reward_checks_table_a1.md"))
    J.append(Job("E0_check_parncutt12", "E0", [PY, "check_reward.py", *p.P, "--cost", "parncutt12", "--mirror", "--chords",
                                               "--vmax", "--part-a", "--out", str(RES / "reward_checks.md")],
                 RES / "reward_checks.md"))
    for split in ("val", "test"):
        J.append(Job(f"E0_m0_given_{split}", "E0", [PY, "search.py", *p.P, "--split", split, "--mode", "given",
                                                    "--workers", "2", "--out", str(RES / f"m0_given_{split}.json")],
                     RES / f"m0_given_{split}.json"))
        J.append(Job(f"E0_m0_assign_{split}", "E0", [PY, "search.py", *p.P, "--split", split, "--mode", "assign",
                                                     "--beam", "10000", "--workers", "2",
                                                     "--out", str(RES / f"m0_assign_{split}.json")],
                     RES / f"m0_assign_{split}.json"))
    J.append(Job("E0_beam_check", "E0", [PY, "search.py", *p.P, "--brute-check", "200", "--workers", "2",
                                         "--out", str(RES / "beam_check.json")], RES / "beam_check.json"))
    # E1: S1 (5 seeds); S2 is fitted inside the evaluation (deterministic)
    for s in range(5):
        out = RUNS / f"s1_s{s}"
        J.append(Job(f"s1_s{s}", "E1", [PY, "train_bc.py", *p.P, "--mode", "given", "--seed", str(s), "--out", str(out)],
                     out / "done.json"))
    # E2: R1; E3: R2 lambda sweep
    for s in range(3):
        J.append(p.train("r1", "E2", p.given(0), s))
        for lam in LAMBDAS:
            J.append(p.train(f"r2_lam{lam:g}", "E3", p.given(lam), s))
    # E8: benchmarks (lambda 0.5, the README's default example; tabular Q-learning: rule reward only)
    for s in range(3):
        J.append(p.train("dqn", "E8", p.given(0.5) + ["--algo", "dqn"], s))
        out = RUNS / f"ppo_s{s}"
        J.append(Job(f"ppo_s{s}", "E8", [PY, "train_ppo.py", *p.P, "--mode", "given", "--lam", "0.5", "--seed", str(s),
                                         "--out", str(out)], out / "done.json"))
    out = RUNS / "qlearn"
    J.append(Job("qlearn", "E8", [PY, "train.py", *p.P, "--mode", "given", "--algo", "qlearn", "--split", "test",
                                  "--episodes", "500", "--out", str(out)], out / "148.json"))
    return J


def target_distance(m: dict) -> float:
    """Distance of a validation result to the README's target region (d_rel <= 0.81, Mgen >= 0.62)."""
    d = m.get("d_rel_balliauw15", {}).get("mean", 9.0)
    g = m.get("Mgen", {}).get("mean", 0.0)
    return max(0.0, d - 0.81) + max(0.0, 0.62 - g)


def rank_on_validation(p: Plan, names: List[str], tag: str) -> List[tuple]:
    """Evaluate the named methods (all seeds) on validation; rank by distance to the target region,
    ties broken by higher Mgen."""
    out = RES / f"select_{tag}.json"
    ck = [str(RUNS / f"{n}_s*" / "best.pt") for n in names]
    j = p.evaluate(f"select_{tag}", "sel", ["--split", "val", "--mode", "given", "--no-baselines",
                                             "--reference", str(RES / "m0_given_val.json"), "--ckpt", *ck], out)
    run_jobs([j], 1, 0, 0.0)
    js = json.loads(out.read_text())["methods"]
    rows = sorted(((target_distance(js[n]), -js[n].get("Mgen", {}).get("mean", 0), n) for n in names if n in js))
    for d, g, n in rows:
        log(f"  validation {n}: distance to target {d:.4f}, Mgen {-g:.4f}, "
            f"d_rel {js[n].get('d_rel_balliauw15', {}).get('mean', float('nan')):.3f}")
    return rows


def round2(p: Plan, lams: List[float], best_lam: float) -> List[Job]:
    J = []
    for lam in lams:  # E4: R3 at the best two lambdas
        for s in range(3):
            J.append(p.train(f"r3_lam{lam:g}", "E4", p.given(lam, elite=True), s))
    full = p.given(best_lam, elite=True)  # E5: accelerators on the R3 configuration at the best lambda
    for s in range(3):
        J.append(p.train("a1", "E5", full + ["--bc-init", str(RUNS / f"s1_s{s}" / "best.pt")], s))
        J.append(p.train("a2", "E5", full + ["--demos", "0.25"], s))
        J.append(p.train("a3", "E5", full + ["--elite-seed-optimum"], s))
    for s in (3, 4):  # E9: best R2 and R3 to 5 seeds
        J.append(p.train(f"r2_lam{best_lam:g}", "E9", p.given(best_lam), s))
        J.append(p.train(f"r3_lam{best_lam:g}", "E9", full, s))
    return J


def round3(p: Plan, best_name: str, best_args: List[str], best_lam: float) -> List[Job]:
    J = []
    full = p.given(best_lam, elite=True)
    for s in range(3):
        # E6: R4 fine-tuning of the best configuration on every test piece
        out = RUNS / f"r4_s{s}"
        J.append(Job(f"r4_s{s}", "E6", [PY, "finetune.py", "--ckpt", str(RUNS / f"{best_name}_s{s}" / "best.pt"), *p.P,
                                        "--split", "test", "--episodes", "50", "--seed", str(s), "--device", p.a.device,
                                        "--out", str(out)], out / "done.json", gpu=p.a.device == "cuda"))
        # E7: 2c, trunk from the 2b run of the same seed, curriculum
        assign = ["--mode", "assign"] + best_args[2:] + ["--init", str(RUNS / f"{best_name}_s{s}" / "best.pt"), "--curriculum"]
        J.append(p.train("2c", "E7", assign, s))
    # E10: ablations, one component removed from the full method (2 seeds)
    variants = {
        "abl_nobonus": ["--mode", "given", "--lam", "0", "--elite", "1"],
        "abl_noelite": p.given(best_lam),
        "abl_nomask": full + ["--no-mask"],
        "abl_notime": full + ["--w-time", "0"],
        "abl_nolook": full + ["--lookahead", "0"],
        "abl_nohandpos": full + ["--no-hand-pos"],
        "abl_nstep1": full + ["--n-step", "1"],
        "abl_nodouble": full + ["--no-double", "--no-dueling"],
        "abl_reduce": full + ["--chords", "reduce"],
        "abl_nooverlap": full + ["--overlap-tol", "0"],
        "abl_balliauw": ["--mode", "given", "--cost", "balliauw15", "--lam", f"{best_lam:g}", "--elite", "1"],
    }
    for s in range(2):
        for name, args in variants.items():
            J.append(p.train(name, "E10", args, s))
        # crossing term exists only with hand assignment: 2c without it, against 2c
        nocross = ["--mode", "assign"] + full[2:] + ["--w-cross", "0", "--init",
                                                       str(RUNS / f"{best_name}_s{s}" / "best.pt"), "--curriculum"]
        J.append(p.train("abl_nocross", "E10", nocross, s))
    # E12: sensitivity, 1 seed, 500k steps, evaluated on validation
    sens = {f"sens_lam_{lam:g}": p.given(lam) for lam in (0.0, 0.25, 0.5, 1.0, 2.0)}
    sens.update({f"sens_elite_{e:g}": full[:-2] + ["--elite", f"{e:g}"] for e in (0.1, 0.25, 0.5)})
    sens.update({f"sens_lr_{lr:g}": full + ["--lr", f"{lr:g}"] for lr in (1e-4, 3e-4, 1e-3)})
    sens.update({f"sens_gamma_{g:g}": full + ["--gamma", f"{g:g}"] for g in (0.9, 0.99)})
    sens.update({f"sens_tol_{c:g}-{o:g}": full + ["--chord-tol", f"{c:g}", "--overlap-tol", f"{o:g}"]
                 for c, o in ((0.02, 0.03), (0.03, 0.05), (0.05, 0.08))})
    sens.update({f"sens_vmax_{q:g}": full + ["--vmax-percentile", f"{q:g}"] for q in (90, 95, 99)})
    for name, args in sens.items():
        J.append(p.train(name, "E12", args, 0, steps=500_000))
    return J


def final(p: Plan, best_name: str, best_lam: float) -> List[Job]:
    refs = ["--reference", str(RES / "m0_given_test.json"), str(RES / "m0_assign_test.json")]
    main_ck = [str(RUNS / "*_s*" / "best.pt"), str(RUNS / "ppo_s*" / "best.zip")]
    J = [p.evaluate("E_main", "eval", ["--split", "test", "--mode", "given", "--cost", "parncutt12", "balliauw15", *refs,
                                       "--success-gap", "0.10", "--seen-unseen", "--wilcoxon", "--bootstrap", "10000",
                                       "--compute", "--ckpt", *main_ck, "--fingerings", str(RUNS / "r4_s*"),
                                       str(RUNS / "qlearn")], RES / "main_all.json")]
    rob = [str(RUNS / f"{n}_s*" / "best.pt") for n in ("s1", f"r2_lam{best_lam:g}", f"r3_lam{best_lam:g}")]
    J.append(p.evaluate("E11_robustness", "E11", ["--split", "test", "--mode", "given", *refs, "--no-baselines",
                                                  "--robustness", "tempo=0.75,1.5", "transpose=-5,-2,-1,1,2,5",
                                                  "jitter=0.01,0.02", "span=0.85,1.15", "era", "stress",
                                                  "--ckpt", *rob], RES / "robustness.json"))
    J.append(p.evaluate("E12_sensitivity", "E12", ["--split", "val", "--mode", "given", "--no-baselines",
                                                   "--reference", str(RES / "m0_given_val.json"),
                                                   "--ckpt", str(RUNS / "sens_*_s0" / "best.pt")], RES / "sensitivity.json"))
    return J


def plots(p: Plan, best_name: str):
    main = RES / "main.json"
    if (RES / "main_all.json").exists():
        js = json.loads((RES / "main_all.json").read_text())
        js["methods"] = {k: v for k, v in js["methods"].items() if not k.startswith("sens_")}
        js["runs"] = [r for r in js["runs"] if not r["method"].startswith("sens_")]
        main.write_text(json.dumps(js, default=float))
    cmds = [
        ["--curves", str(RUNS), "--threshold-gap", "0.15", "--out", str(RES / "curves.png")],
        ["--success", str(RUNS), "--out", str(RES / "success.png")],
        ["--regret", str(main), "--out", str(RES / "regret.png")],
        ["--seen-unseen", str(main), "--out", str(RES / "generalisation.png")],
        ["--seeds", str(main), "--out", str(RES / "seeds.png")],
        ["--lambda-curve", str(main), "--paper-points", "data/twohand/eer2025_table5.json", "--out", str(RES / "curve.png")],
        ["--r4", str(main), "--out", str(RES / "r4.png")],
        ["--sensitivity", str(RES / "sensitivity.json"), "--out", str(RES / "sensitivity.png")],
        ["--tables", str(main), str(RES / "robustness.json"), "--out", str(RES / "tables.md")],
        ["--excerpts", str(RUNS / f"{best_name}_s0" / "best.pt"), str(RUNS / "s1_s0" / "best.pt"), "--pig-dir",
         p.a.pig_dir, "--out", str(RES / "excerpts.md")],
    ]
    for c in cmds:
        r = subprocess.run([PY, "plot.py", *c], capture_output=True, text=True, encoding="utf-8", errors="replace")
        log(f"  plot {c[0]}: {'ok' if r.returncode == 0 else 'FAILED: ' + r.stderr.strip().splitlines()[-1]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pig-dir", required=True)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--gpu-jobs", type=int, default=4)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--min-free-gb", type=float, default=2.5, help="start a job only with this much RAM free")
    ap.add_argument("--eval-workers", type=int, default=6)
    ap.add_argument("--rounds", nargs="*", type=int, default=[1, 2, 3, 4], help="4 = final evaluation and plots")
    args = ap.parse_args()
    p = Plan(args)
    try:  # keep the machine awake while the runner lives (ES_CONTINUOUS | ES_SYSTEM_REQUIRED); no settings change
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception:  # noqa: BLE001
        pass
    state_f = RES / "runner_state.json"
    state = json.loads(state_f.read_text()) if state_f.exists() else {}
    t0 = time.time()
    log(f"run_twohand: rounds {args.rounds}, {args.jobs} jobs at once ({args.gpu_jobs} on the GPU)")

    if 1 in args.rounds:
        log("round 1: E0 checks and M0, E1 S1, E2 R1, E3 R2 lambda sweep, E8 benchmarks")
        run_jobs(round1(p), args.jobs, args.gpu_jobs, args.min_free_gb)
    if "lams" not in state:
        rows = rank_on_validation(p, [f"r2_lam{lam:g}" for lam in LAMBDAS], "lambda")
        lams = [float(n.split("lam")[1]) for _, _, n in rows[:2]]
        state.update(lams=lams, best_lam=lams[0])
        state_f.write_text(json.dumps(state, indent=1))
    log(f"best lambdas on validation: {state['lams']} (best {state['best_lam']})")

    if 2 in args.rounds:
        log("round 2: E4 R3, E5 A1-A3, E9 extra seeds")
        run_jobs(round2(p, state["lams"], state["best_lam"]), args.jobs, args.gpu_jobs, args.min_free_gb)
    if "best_name" not in state:
        lam = state["best_lam"]
        rows = rank_on_validation(p, [f"r2_lam{lam:g}", f"r3_lam{lam:g}"], "best_r")
        state["best_name"] = rows[0][2]
        state_f.write_text(json.dumps(state, indent=1))
    best_name, best_lam = state["best_name"], state["best_lam"]
    best_args = p.given(best_lam, elite=best_name.startswith("r3"))
    log(f"best R configuration on validation: {best_name}")

    if 3 in args.rounds:
        log("round 3: E6 R4, E7 2c, E10 ablations, E12 sensitivity")
        run_jobs(round3(p, best_name, best_args, best_lam), args.jobs, args.gpu_jobs, args.min_free_gb)
    if 4 in args.rounds:
        log("final: E11 robustness, main evaluation, sensitivity evaluation, figures and tables")
        # one evaluation at a time: two pools of eval workers together exhausted Windows commit memory (WinError 1450)
        run_jobs(final(p, best_name, best_lam), 1, 0, args.min_free_gb)
        plots(p, best_name)
    log(f"run_twohand finished in {(time.time() - t0) / 3600:.2f} h")


if __name__ == "__main__":
    main()
