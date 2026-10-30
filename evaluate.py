"""Evaluate trained agents and baselines on the PIG test set (one hand, monophonic passages).

Every method is run on exactly the same passages, so the numbers are comparable to each other
(but not to published PIG numbers, which include chords and both hands).

    python evaluate.py --pig-dir data/PIG --ckpt runs/rh_ddqn_s*/best.pt --out results/rh.json
    python evaluate.py --pig-dir data/PIG --baselines-only --show 2
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

from fingerly.baselines import HMM1, Greedy, OptimalCost, RandomValid
from fingerly.cost import CostConfig
from fingerly.data import build_split
from fingerly.env import FingeringEnv, run_policy
from fingerly.metrics import evaluate_predictor, human_agreement

COLUMNS = ["Mgen", "Mhigh", "Msoft", "effort_per_note", "impossible_rate",
           "crossing_rate", "shift_rate", "weak_finger_rate"]


def fmt(v, pct):
    return f"{100 * v:6.2f}" if pct else f"{v:6.3f}"


def print_table(rows):
    pct = {"Mgen", "Mhigh", "Msoft", "impossible_rate", "crossing_rate", "shift_rate", "weak_finger_rate"}
    head = f"{'method':<28}" + "".join(f"{c[:14]:>16}" for c in COLUMNS)
    print(head)
    print("-" * len(head))
    for name, m, sd in rows:
        cells = []
        for c in COLUMNS:
            if c not in m:
                cells.append(f"{'-':>16}")
                continue
            s = fmt(m[c], c in pct)
            if sd and c in sd:
                s += "±" + fmt(sd[c], c in pct).strip()
            cells.append(f"{s:>16}")
        print(f"{name:<28}" + "".join(cells))
    print("(match rates, impossible/crossing/shift/weak-finger rates in %; effort in cost units per note)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pig-dir", required=True)
    ap.add_argument("--ckpt", nargs="*", default=[], help="checkpoint paths or globs (seeds are averaged)")
    ap.add_argument("--hand", choices=["R", "L"], default=None, help="default: taken from the checkpoint")
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--min-len", type=int, default=None)
    ap.add_argument("--val-pieces", type=int, default=None)
    ap.add_argument("--split-seed", type=int, default=None)
    ap.add_argument("--baselines-only", action="store_true")
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--show", type=int, default=0, help="print N example passages with all fingerings")
    ap.add_argument("--out", default=None, help="write all numbers to this JSON file")
    args = ap.parse_args()

    ckpts = sorted({p for pat in args.ckpt for p in glob.glob(pat)})
    if args.ckpt and not ckpts:
        raise SystemExit(f"no checkpoints match {args.ckpt}")
    agents = []
    if ckpts and not args.baselines_only:
        from fingerly.dqn import DQNAgent  # torch is only needed when there are checkpoints

        agents = [DQNAgent.load(p) for p in ckpts]
    meta = agents[0].extra if agents else {}

    hand = args.hand or meta.get("hand", "R")
    min_len = args.min_len or meta.get("min_len", 8)
    val_pieces = args.val_pieces if args.val_pieces is not None else meta.get("val_pieces", 12)
    split_seed = args.split_seed if args.split_seed is not None else meta.get("split_seed", 0)
    span_scale = meta.get("span_scale", 1.0)
    lookahead = meta.get("lookahead", 3)

    split = build_split(args.pig_dir, hand, min_len, val_pieces, split_seed)
    passages = split.test if args.split == "test" else split.val
    cfg = CostConfig(span_scale=span_scale)
    rows, results, preds = [], {}, {}

    hum = human_agreement(passages)
    if hum:
        rows.append(("human agreement (LOO)", hum, None))
        results["human_agreement"] = hum

    if agents:
        env = FingeringEnv(passages, hand=hand, lookahead=lookahead, cost_cfg=cfg)
        per_seed = []
        for path, agent in zip(ckpts, agents):
            policy = agent.policy()
            m = evaluate_predictor(lambda p: run_policy(env, p, policy), passages, cfg)
            per_seed.append(m)
            results[f"dqn:{path}"] = m
            preds.setdefault("DQN (seed 1)", [run_policy(env, p, policy) for p in passages[: args.show]])
        keys = [k for k in per_seed[0] if isinstance(per_seed[0][k], float)]
        mean = {k: float(np.mean([m[k] for m in per_seed])) for k in keys}
        sd = {k: float(np.std([m[k] for m in per_seed])) for k in keys} if len(per_seed) > 1 else None
        name = f"DQN agent (n={len(per_seed)})"
        rows.append((name, mean, sd))
        results["dqn_mean"], results["dqn_std"] = mean, sd

    if not args.no_baselines:
        hmm = HMM1().fit(split.train)
        for b in [RandomValid(cfg), Greedy(cfg), OptimalCost(cfg), hmm]:
            m = evaluate_predictor(b.predict, passages, cfg)
            rows.append((b.name, m, None))
            results[b.name] = m
            preds[b.name] = [b.predict(p) for p in passages[: args.show]]

    print(f"\n{args.split} set: {len(passages)} passages, {sum(map(len, passages))} notes, hand {hand}\n")
    print_table(rows)

    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    for i, p in enumerate(passages[: args.show]):
        print(f"\n--- {p.piece}, passage {i} ({len(p)} notes) ---")
        print(f"{'notes':<20}" + " ".join(f"{names[x % 12] + str(x // 12 - 1):>4}" for x in p.pitches))
        for k, lab in enumerate(p.labels):
            print(f"{'expert ' + str(k + 1):<20}" + " ".join(f"{f if f else '-':>4}" for f in lab))
        for name, ps in preds.items():
            print(f"{name[:19]:<20}" + " ".join(f"{f:>4}" for f in ps[i]))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")



# ======================================================================================
# Part C: `python evaluate.py --mode given|assign|lh ...` (README §C.11)
# ======================================================================================
TWOHAND_EVAL_DOC = """Part C evaluation (README §C.11): every metric, statistic and robustness test.

    python evaluate.py --pig-dir data/PianoFingeringDataset_v1.2 --split test --mode given \\
        --ckpt "runs/*/best.pt" --cost parncutt12 balliauw15 \\
        --reference results/twohand/m0_given_test.json --success-gap 0.10 --seen-unseen \\
        --wilcoxon --bootstrap 10000 --compute --out results/twohand/main.json

Each checkpoint (.pt from train.py / train_bc.py, .zip from train_ppo.py) and each fingering folder
(--fingerings: finetune.py's or Q-learning's per-piece JSON files) is evaluated in its own mode and
with its own settings, with the frozen greedy policy and no imitation bonus (§C.11.1). Seeds of one
method (run folders that differ only in a trailing _s<seed>) are aggregated. Baselines: random-valid,
greedy, S2 (HMM over finger sets), M0 and the pianists.
"""


def twohand_eval_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=TWOHAND_EVAL_DOC, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pig-dir", required=True)
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--mode", choices=["lh", "given", "assign"], required=True, help="mode of the baselines")
    ap.add_argument("--ckpt", nargs="*", default=[], help="checkpoints or globs (.pt / .zip)")
    ap.add_argument("--fingerings", nargs="*", default=[], help="folders of per-piece fingerings (R4, Q-learning)")
    ap.add_argument("--cost", nargs="+", default=["parncutt12", "balliauw15"], help="cost models reported")
    ap.add_argument("--reference", nargs="*", default=[], help="M0 JSON files from search.py (used when settings match)")
    ap.add_argument("--m0-beam", type=int, default=1000, help="beam when M0 must be computed (2c)")
    ap.add_argument("--success-gap", type=float, default=0.10)
    ap.add_argument("--seen-unseen", action="store_true")
    ap.add_argument("--wilcoxon", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=0)
    ap.add_argument("--compute", action="store_true", help="compute table (always collected; kept for README syntax)")
    ap.add_argument("--robustness", nargs="*", default=None,
                    help="tempo=0.75,1.5 transpose=-5,-2,-1,1,2,5 jitter=0.01,0.02 span=0.85,1.15 era stress")
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--job-timeout", type=float, default=3600, help="exit with an error if no run finishes in this time (s)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    return ap


# ---------------------------------------------------------------- per-process caches
_CTX, _POOLS, _HUMAN, _M0 = {}, {}, {}, {}


def _ctx(pig_dir, prep_d):
    from fingerly.twohand.prepare import Context, PrepConfig

    key = json.dumps(prep_d, sort_keys=True)
    if key not in _CTX:
        _CTX[key] = Context(pig_dir, PrepConfig(**prep_d))
    return _CTX[key]


def _env_cfg(env_d, v_max, **over):
    from fingerly.twohand.env import EnvConfig

    d = {k: v for k, v in env_d.items() if k in EnvConfig.__dataclass_fields__}
    d.update(v_max=v_max, lam=0.0, **over)
    return EnvConfig(**d)


def _pool(segs, ecfg, tag):
    from fingerly.twohand.env import SegmentPool

    key = (tag, json.dumps(ecfg.__dict__, sort_keys=True, default=str))
    if key not in _POOLS:
        _POOLS[key] = SegmentPool(segs, ecfg, bonus=False)
    return _POOLS[key]


def _models(names, ecfg):
    from fingerly.twohand.rules import make_model

    return {n: make_model(n, ecfg.chord_horizontal, ecfg.span_scale) for n in names}


def _human(pool, names, tag):
    from fingerly.metrics import human_reference

    key = (tag, pool.cfg.mode, pool.cfg.chord_horizontal, pool.cfg.span_scale, tuple(names))
    if key not in _HUMAN:
        _HUMAN[key] = human_reference(pool.infos, _models(names, pool.cfg), pool.cfg.v_max)
    return _HUMAN[key]


def _m0(pool, beam, reference, tag):
    """M0 costs {(piece, index): cost} for this pool's settings: from --reference when its settings
    match, else from a disk cache, else computed (exact DP in 2a/2b, beam search in 2c)."""
    import hashlib

    from fingerly.twohand.search import m0_costs

    cfg = pool.cfg
    sig = dict(mode=cfg.mode, cost=cfg.cost, chord_horizontal=cfg.chord_horizontal, span_scale=cfg.span_scale,
               v_max=round(cfg.v_max, 6), w_time=cfg.w_time, w_cross=cfg.w_cross, cross=cfg.cross,
               cross_margin=cfg.cross_margin, tag=tag)
    key = json.dumps(sig, sort_keys=True)
    if key in _M0:
        return _M0[key]
    keys = {(i.seg.piece, i.seg.index) for i in pool.infos}
    refs = [reference] if isinstance(reference, str) else list(reference or [])
    for r_path in refs:
        if not (Path(r_path).exists() and tag.startswith(("test|{}", "val|{}"))):
            continue
        ref = json.loads(Path(r_path).read_text())
        rc = ref.get("config", {})
        same = all(rc.get(k) == sig[k] for k in ("mode", "cost", "chord_horizontal", "span_scale", "w_time", "cross"))
        same = same and abs(rc.get("v_max", 0) - cfg.v_max) < 1e-6 and (cfg.mode != "assign" or rc.get("w_cross") == cfg.w_cross)
        if same and ref.get("config", {}).get("split", tag.split("|")[0]) == tag.split("|")[0]:
            m = {(r["piece"], r["index"]): r["cost"] for r in ref["segments"] if (r["piece"], r["index"]) in keys}
            if len(m) == len(keys):
                _M0[key] = m
                return m
    sig_all = key + f"|{beam}|" + json.dumps(sorted(keys))
    f = Path("cache/twohand") / f"m0_{hashlib.sha1(sig_all.encode()).hexdigest()[:16]}.json"
    if f.exists():
        m = {(k.split("|", 1)[0], int(k.split("|", 1)[1])): v for k, v in json.loads(f.read_text()).items()}
    else:
        m = m0_costs(pool, beam)
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps({f"{a}|{b}": v for (a, b), v in m.items()}))
        try:
            os.replace(tmp, f)
        except PermissionError:  # Windows: another worker wrote the same file and has it open; same content
            if not f.exists():
                raise
            tmp.unlink(missing_ok=True)
    _M0[key] = m
    return m


# ---------------------------------------------------------------- policies
def _method_name(path: Path) -> str:
    import re

    d = path if path.is_dir() else path.parent
    return re.sub(r"_s\d+$", "", d.name)


def _load(path: Path, device: str):
    """(batch policy, meta, n_params, kind); a fingering folder gives (None, meta, 0, 'fingerings')."""
    if path.is_dir():
        return None, json.loads((path / "config.json").read_text()), 0, "fingerings"
    if path.suffix == ".zip":
        from sb3_contrib import MaskablePPO

        model = MaskablePPO.load(path, device=device)
        meta = json.loads((path.parent / "config.json").read_text())

        def pol(obs, masks):
            act, _ = model.predict(obs, action_masks=masks, deterministic=True)
            return np.asarray(act)

        return pol, meta, sum(p.numel() for p in model.policy.parameters()), "ppo"
    from fingerly.dqn import Agent

    ag = Agent.load(path, device)
    return ag.greedy(), ag.extra, sum(p.numel() for p in ag.q.parameters()), "dqn"


def _fingering_lookup(folder: Path):
    out = {}
    for f in folder.glob("*.json"):
        if f.name in ("config.json", "done.json"):
            continue
        for s in json.loads(f.read_text()).get("segments", []):
            out[(s["piece"], s["index"])] = [tuple(x) for x in s["fingering"]]
    return out


# ---------------------------------------------------------------- RL-specific metrics from the logs (§C.11.2)
def _rl_from_logs(run_dir: Path, val_lo: float, val_hi: float, lam: float) -> dict:
    import csv

    from fingerly.metrics import curve_auc, first_sustained, value_at

    out = {}
    f = run_dir / "eval_log.csv"
    if f.exists():
        rows = list(csv.DictReader(open(f)))
        if rows and "step" in rows[0]:
            num = lambda r, k: float(r[k]) if r.get(k) not in (None, "") else float("nan")  # noqa: E731
            steps = [num(r, "step") for r in rows]
            ret = [num(r, "return_per_event") for r in rows]
            out["curve"] = dict(steps=steps, return_per_event=ret, SR=[num(r, "SR") for r in rows],
                                gap=[num(r, "gap") for r in rows], Mgen=[num(r, "Mgen") for r in rows])
            out["N_star_gap15"] = first_sustained(steps, out["curve"]["gap"], lambda g: g <= 0.15)
            if lam > 0:
                out["N_star_mgen60"] = first_sustained(steps, out["curve"]["Mgen"], lambda m: m >= 0.60)
            out["auc_1M"] = curve_auc(steps, ret, val_lo, val_hi, 1_000_000)
            for at in (100_000, 250_000, 500_000, 1_000_000):
                out[f"val_return_at_{at // 1000}k"] = value_at(steps, ret, at)
    d = run_dir / "done.json"
    if d.exists():
        out["done"] = json.loads(d.read_text())
    t = run_dir / "train_log.csv"
    if t.exists():
        sps = [float(r["steps_per_sec"]) for r in csv.DictReader(open(t)) if r.get("steps_per_sec")]
        if sps:
            out["steps_per_sec"] = float(np.median(sps))
    return out


def _val_reference(env_d, prep_d, pig_dir, v_max):
    """Validation return per event of a random-valid policy (lo) and of M0 (hi): the AUC scale."""
    from fingerly.metrics import rollout, summarize

    ctx = _ctx(pig_dir, prep_d)
    ecfg = _env_cfg(env_d, v_max)
    tag = "val|" + json.dumps(prep_d, sort_keys=True)
    pool = _pool(ctx.split.val, ecfg, tag)
    rng = np.random.default_rng(0)
    lo = summarize(rollout(pool, lambda o, m: np.array([rng.choice(np.flatnonzero(x)) for x in m])))["return_per_event"]
    m0 = _m0(pool, 1000, None, tag)
    hi = -sum(m0.values()) / max(sum(len(i.order) for i in pool.infos), 1)
    return lo, hi


# ---------------------------------------------------------------- one run
def _eval_one(job) -> dict:
    """Evaluate one checkpoint or fingering folder (in a worker process)."""
    import time as _time

    import torch

    from fingerly.metrics import method_metrics, replay, rollout

    torch.set_num_threads(1)
    path, a = Path(job["path"]), job["args"]
    pol, meta, n_params, kind = _load(path, a["device"])
    env_d = dict(meta.get("env", {}))
    prep_d = meta.get("prep") or {}
    ctx = _ctx(a["pig_dir"], prep_d)
    ecfg = _env_cfg(env_d, ctx.v_max)
    ptag = json.dumps(prep_d, sort_keys=True)
    split = a["split"]
    tag = f"{split}|{ptag}"
    pool = _pool(ctx.segments(split), ecfg, tag)
    models = _models(a["cost"], ecfg)
    human = _human(pool, a["cost"], tag)
    m0 = _m0(pool, a["m0_beam"], a["reference"], tag)
    t0 = _time.time()
    eps = replay(pool, _fingering_lookup(path)) if kind == "fingerings" else rollout(pool, pol)
    infer = _time.time() - t0
    res = method_metrics(eps, models, human, ctx.v_max, m0, a["success_gap"])
    notes = sum(r["notes"] for r in res["segments"])
    res["summary"]["inference_ms_per_note"] = 1000 * infer / max(notes, 1)
    res["summary"]["inference_s_per_piece"] = infer / max(len(res["pieces"]), 1)
    res.update(path=str(path), method=_method_name(path), kind=kind, mode=ecfg.mode, params=n_params,
               seed=meta.get("seed", meta.get("args", {}).get("seed")), lam=env_d.get("lam", 0.0))
    if kind == "fingerings":
        res["rl"] = _r4_curves(path, human)
        return res
    if a["seen_unseen"]:
        tpool = _pool(ctx.split.train, ecfg, f"train|{ptag}")
        seen = method_metrics(rollout(tpool, pol), models, _human(tpool, a["cost"], f"train|{ptag}"), ctx.v_max)["summary"]
        res["seen"] = {k: seen.get(k) for k in ("return_per_event", "Mgen", "d_rel_balliauw15", "d_rel_parncutt12")}
    lo, hi = _val_reference(env_d, prep_d, a["pig_dir"], ctx.v_max)
    res["rl"] = _rl_from_logs(path.parent, lo, hi, env_d.get("lam", 0.0))
    res["rl"]["val_random"], res["rl"]["val_m0"] = lo, hi
    if a["robustness"] is not None and job.get("robust"):
        res["robustness"] = _robustness(pol, ecfg, ctx, prep_d, a, res)
    return res


def _result_file(job) -> Path:
    """Disk cache for one run's result: keyed by the checkpoint (path and mtime) and the evaluation settings."""
    import hashlib

    p = Path(job["path"])
    stamp = max((f.stat().st_mtime for f in p.glob("*.json")), default=0) if p.is_dir() else p.stat().st_mtime
    sig = json.dumps(dict(job, stamp=stamp), sort_keys=True, default=str)
    return Path("cache/twohand/eval") / f"{p.parent.name if p.is_file() else p.name}_{hashlib.sha1(sig.encode()).hexdigest()[:16]}.json"


def _eval_cached(job) -> str:
    """Worker entry: evaluate one run unless cached, write its result to disk and return the file path, so only a
    short string crosses the process pipe (large results through the pipe failed with WinError 1450)."""
    f = _result_file(job)
    if not f.exists():
        res = _eval_one(job)
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(res, default=float))
        os.replace(tmp, f)
    return str(f)


def _r4_curves(folder: Path, human) -> dict:
    """R4: per piece, d_rel (balliauw15) after each fine-tuning episode; episodes to reach 0.81
    against the 2025 paper's 500 per piece. Q-learning folders have no curve."""
    out, needed = {}, []
    for f in folder.glob("*.json"):
        if f.name in ("config.json", "done.json"):
            continue
        d = json.loads(f.read_text())
        h = human.difficulty.get("balliauw15", {}).get(d.get("piece", f.stem))
        if "curve" not in d or not isinstance(d["curve"], list) or not d["curve"] or not isinstance(d["curve"][0], dict) or not h:
            continue
        drel = [c["balliauw15"] / h for c in d["curve"]]
        out[d["piece"]] = drel
        needed.append(next((k for k, v in enumerate(drel) if v <= 0.81), None))
    hit = [e for e in needed if e is not None]
    return dict(d_rel_curves=out, pieces=len(needed), reached_081=len(hit),
                median_episodes_to_081=float(np.median(hit)) if hit else None)


# ---------------------------------------------------------------- robustness (§C.11.7)
def _robustness(pol, ecfg, ctx, prep_d, a, clean) -> dict:
    import dataclasses

    from fingerly.metrics import HumanRef, composer_eras, method_metrics, rollout
    from fingerly.twohand.robust import parse_spec, perturbed_segments, stress_segments

    out = {}
    keys = ("d_rel_balliauw15", "Mgen", "SR", "infeasible_rate")
    base = clean["summary"]
    ptag = json.dumps(prep_d, sort_keys=True)
    ids = ctx.split.test_pieces if a["split"] == "test" else ctx.split.val_pieces
    for kind_p, level in parse_spec(a["robustness"]):
        name = kind_p if level is None else f"{kind_p}={level:g}"
        if kind_p == "era":
            eras = composer_eras(a["pig_dir"])
            groups = {}
            for p, v in clean["pieces"].items():
                groups.setdefault(eras.get(p, "Other"), []).append(v)
            out[name] = {e: dict(pieces=len(vs), **{k: float(np.nanmean([x.get(k, np.nan) for x in vs]))
                                                    for k in ("Mgen", "d_rel_balliauw15")}) for e, vs in groups.items()}
            continue
        cfg, labelled = ecfg, True
        if kind_p == "span":
            cfg, segs = dataclasses.replace(ecfg, span_scale=level), ctx.segments(a["split"])
        elif kind_p == "stress":
            segs, labelled = stress_segments(40, 0, ctx.prep), False
        else:
            segs = perturbed_segments(ctx.pig, ids, kind_p, level, ctx.prep)
        tag = f"robust|{name}|{ptag}"
        pool = _pool(segs, cfg, tag)
        models = _models(a["cost"], cfg)
        human = _human(pool, a["cost"], tag) if labelled else HumanRef({n: {} for n in models}, {n: {} for n in models},
                                                                       {}, float("inf"), {})
        s = method_metrics(rollout(pool, pol), models, human, ctx.v_max, _m0(pool, a["m0_beam"], None, tag),
                           a["success_gap"])["summary"]
        out[name] = {k: s.get(k) for k in keys + ("difficulty_per_note_parncutt12",)}
        out[name]["delta"] = {k: (base[k] - s[k]) if (k in s and k in base) else None for k in keys}
    return out


# ---------------------------------------------------------------- baselines
def _baselines(a, mode) -> dict:
    from fingerly.baselines import HMMSets
    from fingerly.metrics import method_metrics, rates, replay, rollout
    from fingerly.twohand.env import Episode
    from fingerly.twohand.search import search_segment

    ctx = _ctx(a["pig_dir"], {})
    ecfg = _env_cfg(dict(mode=mode), ctx.v_max)
    tag = f"{a['split']}|{json.dumps({})}"
    pool = _pool(ctx.segments(a["split"]), ecfg, tag)
    models = _models(a["cost"], ecfg)
    human = _human(pool, a["cost"], tag)
    m0 = _m0(pool, a["m0_beam"], a["reference"], tag)
    rng = np.random.default_rng(0)
    run = lambda eps: method_metrics(eps, models, human, ctx.v_max, m0, a["success_gap"])  # noqa: E731
    out = {"random-valid": run(rollout(pool, lambda o, m: np.array([rng.choice(np.flatnonzero(x)) for x in m])))}
    greedy = []
    for info in pool.infos:
        ep = Episode(pool.cfg, pool.model)
        ep.reset(info)
        while not ep.done:
            ep.step(int(min(np.flatnonzero(ep.mask()), key=ep.action_cost)))
        greedy.append(ep)
    out["greedy"] = run(greedy)
    hmm = HMMSets().fit(ctx.split.train)
    out["S2 (HMM)"] = run(replay(pool, lambda info: hmm.predict_segment(info.seg, info.order)))
    ref = None
    for r_path in a["reference"] or []:
        r_js = json.loads(Path(r_path).read_text()) if Path(r_path).exists() else {}
        rc = r_js.get("config", {})
        if rc.get("mode") == mode and rc.get("split") == a["split"] and rc.get("cost") == ecfg.cost:
            ref = r_js
    if ref:
        fing = {(r["piece"], r["index"]): [tuple(x) for x in r["fingering"]] for r in ref["segments"]}
    else:
        fing = {(i.seg.piece, i.seg.index): search_segment(i, pool.cfg, a["m0_beam"]).fingering for i in pool.infos}
    out["M0 (optimum)"] = run(replay(pool, fing))
    main = a["cost"][0]
    summ = {f"d_rel_{n}": 1.0 for n in a["cost"]}
    summ.update({f"difficulty_total_{n}": float(sum(human.difficulty[n].values())) for n in a["cost"]})
    summ.update({f"annotator_{k}": v for k, v in human.agreement.items()})
    summ.update(rates(human.counts))
    summ["Mgen"] = human.agreement.get("Mgen", float("nan"))
    summ["hand_accuracy"] = human.agreement.get("hand_accuracy", float("nan"))
    out["Human (pianists)"] = dict(summary=summ, pieces={p: {f"difficulty_{n}": human.difficulty[n][p] for n in a["cost"]}
                                                          for p in human.difficulty[main]})
    return out


# ---------------------------------------------------------------- aggregation and statistics (§C.11.8)
PER_PIECE_KEYS = ("Mgen", "d_rel_balliauw15", "d_rel_parncutt12", "difficulty_per_note_parncutt12",
                  "difficulty_per_note_balliauw15", "infeasible_rate", "return_per_event", "hand_accuracy")


def _aggregate(runs: list, a) -> dict:
    from fingerly.metrics import bootstrap_ci, seed_stability, stratified_bootstrap_iqm, wilson

    by = {}
    for r in runs:
        by.setdefault(r["method"], []).append(r)
    out = {}
    for name, rs in by.items():
        agg = dict(seeds=[r.get("seed") for r in rs], n=len(rs), mode=rs[0]["mode"], kind=rs[0]["kind"],
                   lam=rs[0].get("lam"), params=rs[0].get("params"))
        for k in sorted({k for r in rs for k, v in r["summary"].items() if isinstance(v, (int, float))}):
            vals = [r["summary"][k] for r in rs if isinstance(r["summary"].get(k), (int, float))]
            agg[k] = dict(mean=float(np.mean(vals)), std=float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                          median=float(np.median(vals)))
        for k in ("return_per_event", "Mgen", "d_rel_balliauw15"):
            vals = [r["summary"][k] for r in rs if k in r["summary"]]
            if vals:
                agg[f"stability_{k}"] = seed_stability(vals)
        succ = sum(sum(bool(s.get("success")) for s in r["segments"]) for r in rs)
        tot = sum(sum("success" in s for s in r["segments"]) for r in rs)
        if tot:
            agg["SR_pooled"] = succ / tot
            agg["SR_wilson95"] = wilson(succ, tot)
        pieces = sorted({p for r in rs for p in r["pieces"]})
        per_piece = {}
        for k in PER_PIECE_KEYS:
            mat = np.array([[r["pieces"].get(p, {}).get(k, np.nan) for p in pieces] for r in rs], float)
            if mat.size == 0 or np.all(np.isnan(mat)):
                continue
            per_piece[k] = {p: float(np.nanmean(mat[:, j])) for j, p in enumerate(pieces) if not np.all(np.isnan(mat[:, j]))}
            if a["bootstrap"]:
                agg[f"{k}_ci95"] = bootstrap_ci(list(per_piece[k].values()), a["bootstrap"])
                keep = ~np.any(np.isnan(mat), axis=0)
                if len(rs) > 1 and keep.any() and k in ("Mgen", "d_rel_balliauw15", "return_per_event"):
                    agg[f"{k}_iqm"] = stratified_bootstrap_iqm(mat[:, keep], min(a["bootstrap"], 2000))
        agg["per_piece"] = per_piece
        if rs[0]["segments"] and "regret" in rs[0]["segments"][0]:
            agg["cumulative_regret"] = [float(x) for x in np.cumsum([s["regret"] for s in rs[0]["segments"]])]
        seen = [r for r in rs if "seen" in r]
        if seen:
            gen = {}
            for k in ("return_per_event", "Mgen", "d_rel_balliauw15"):
                sv = [r["seen"][k] for r in seen if r["seen"].get(k) is not None]
                uv = [r["summary"][k] for r in seen if k in r["summary"]]
                if sv and uv:
                    gen[k] = dict(seen=float(np.mean(sv)), unseen=float(np.mean(uv)), gap=float(np.mean(sv) - np.mean(uv)))
            agg["generalisation"] = gen
        rl = [r.get("rl", {}) for r in rs]
        agg["rl"] = rl
        done = [x["done"] for x in rl if "done" in x]
        if done:
            secs = [d["seconds"] for d in done]
            agg["compute"] = dict(train_minutes_mean=float(np.mean(secs)) / 60, cpu_hours_total=float(np.sum(secs)) / 3600,
                                  steps_per_sec=float(np.nanmean([x.get("steps_per_sec", np.nan) for x in rl])),
                                  peak_ram_mb=float(np.nanmax([d.get("peak_ram_mb", np.nan) for d in done])),
                                  device=done[0].get("device"), gpu=done[0].get("device") == "cuda",
                                  energy_wh_estimate_45w=float(np.sum(secs)) / 3600 * 45)
        for k in ("N_star_gap15", "N_star_mgen60", "auc_1M"):
            vals = [x.get(k) for x in rl if k in x]
            if vals:
                agg[k] = vals
        rob = [r["robustness"] for r in rs if "robustness" in r]
        if rob:
            agg["robustness"] = rob
        out[name] = agg
    return out


def twohand_eval_main(argv=None):
    import sys
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

    sys.stdout.reconfigure(encoding="utf-8")
    args = twohand_eval_parser().parse_args(argv)
    paths = sorted({Path(p) for pat in args.ckpt for p in glob.glob(pat)})
    # Part C runs have an "env" section in config.json; skip others a glob catches (Part B's runs/rh_ddqn_s0)
    paths = [p for p in paths if p.suffix == ".zip" or "env" in json.loads((p.parent / "config.json").read_text())]
    paths +=sorted({Path(p) for pat in args.fingerings for p in glob.glob(pat) if Path(p).is_dir()})
    a = dict(pig_dir=args.pig_dir, split=args.split, cost=args.cost, reference=args.reference, m0_beam=args.m0_beam,
             success_gap=args.success_gap, seen_unseen=args.seen_unseen, robustness=args.robustness,
             device=args.device, bootstrap=args.bootstrap)
    jobs = [dict(path=str(p), args=a, robust=args.robustness is not None) for p in paths]
    print(f"evaluating {len(jobs)} runs on {args.split} with {args.workers} workers", flush=True)
    files = {i: str(_result_file(j)) for i, j in enumerate(jobs) if _result_file(j).exists()}
    if files:
        print(f"  {len(files)} runs already evaluated (cache/twohand/eval)", flush=True)
    todo = [i for i in range(len(jobs)) if i not in files]
    if todo:
        ex = ProcessPoolExecutor(max(1, min(args.workers, len(todo))), max_tasks_per_child=8)
        futs = {ex.submit(_eval_cached, jobs[i]): i for i in todo}
        pending = set(futs)
        while pending:
            # a dead pool manager thread (seen on Windows: WinError 1450) leaves futures pending forever, so wait
            # with a timeout and exit with an error instead; the runner retries and the cache keeps finished runs
            done, pending = wait(pending, timeout=args.job_timeout, return_when=FIRST_COMPLETED)
            if not done:
                print(f"no run finished in {args.job_timeout} s; the worker pool looks stuck, exiting", flush=True)
                for proc in list(getattr(ex, "_processes", {}).values()):
                    proc.terminate()
                os._exit(3)
            for fu in done:
                files[futs[fu]] = fu.result()
        ex.shutdown()
    runs = []
    for i in range(len(jobs)):
        r = json.loads(Path(files[i]).read_text())
        s = r["summary"]
        print(f"  {r['method']:<30} seed {r.get('seed')}: Mgen {s.get('Mgen', float('nan')):.4f} "
              f"d_rel {s.get('d_rel_balliauw15', float('nan')):.3f} gap {s.get('gap', float('nan')):.3f} "
              f"infeasible {s['infeasible_rate']:.4f}", flush=True)
        runs.append(r)
    methods = _aggregate(runs, a)
    if not args.no_baselines:
        for name, r in _baselines(a, args.mode).items():
            methods[name] = dict(
                {k: dict(mean=v, std=0.0, median=v) for k, v in r["summary"].items() if isinstance(v, (int, float))},
                n=1, mode=args.mode, kind="baseline",
                per_piece={k: {p: v[k] for p, v in r.get("pieces", {}).items() if k in v} for k in PER_PIECE_KEYS})
            if "segments" in r and r["segments"] and "regret" in r["segments"][0]:
                methods[name]["cumulative_regret"] = [float(x) for x in np.cumsum([s["regret"] for s in r["segments"]])]
    out = dict(config=vars(args), methods=methods, runs=runs)
    if args.wilcoxon:
        from fingerly.metrics import wilcoxon_pairs

        out["wilcoxon"] = {}
        for k, hb in (("Mgen", True), ("d_rel_balliauw15", False), ("difficulty_per_note_parncutt12", False),
                      ("infeasible_rate", False), ("return_per_event", True)):
            pp = {m: v["per_piece"][k] for m, v in methods.items() if v.get("per_piece", {}).get(k)}
            out["wilcoxon"][k] = wilcoxon_pairs(pp, higher_better=hb)
    paper = Path("data/twohand/eer2025_table5.json")
    if paper.exists():
        t5 = json.loads(paper.read_text())["rows"]
        out["paper_2025"] = {k: dict(v, d_rel=v["difficulty"] / t5["Human"]["difficulty"]) for k, v in t5.items()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, default=float))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    import sys

    if any(x == "--mode" or x.startswith("--mode=") for x in sys.argv[1:]):
        twohand_eval_main()  # Part C (README §C.11)
    else:
        main()
