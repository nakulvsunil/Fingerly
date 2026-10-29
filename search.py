"""M0: the optimum reference for Part C (README §C.8). A measuring stick, never used for training.

    python search.py --pig-dir data/PianoFingeringDataset_v1.2 --split test --mode given  --out results/twohand/m0_given.json
    python search.py --pig-dir data/PianoFingeringDataset_v1.2 --split test --mode assign --beam 10000 --out results/twohand/m0_assign.json
    python search.py --pig-dir data/PianoFingeringDataset_v1.2 --brute-check 200 --out results/twohand/beam_check.json

2a and 2b (without --cross) use the exact per-hand DP; 2c and 2b with --cross use beam search.
The output lists, per segment, the optimum environment cost, its rule points under both cost
models, the fingering, and the time taken. --brute-check compares the beam with an unpruned search
on windows of at most 12 events (pass: optimum found in >= 99% of cases).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from fingerly.twohand.env import EnvConfig, SegInfo
from fingerly.twohand.prepare import Context
from fingerly.twohand.prepare import PrepConfig, Segment
from fingerly.twohand.rules import make_model
from fingerly.twohand.rules import rule_points
from fingerly.twohand.search import beam_search, search_segment


def _one(job):
    seg, cfg, beam = job
    model = make_model(cfg.cost, cfg.chord_horizontal, cfg.span_scale)
    info = SegInfo(seg, cfg, model, bonus=False)
    r = search_segment(info, cfg, beam)
    other = {n: rule_points(seg, info.order, r.fingering, make_model(n, cfg.chord_horizontal, cfg.span_scale))
             for n in ("parncutt12", "balliauw15")}
    return dict(piece=seg.piece, index=seg.index, events=len(info.order), notes=sum(seg.events[i].size for i in info.order),
                cost=r.cost, rules=r.rules, rules_by_model=other, seconds=r.seconds, exact=r.exact,
                states_max=r.states_max, fingering=[[h, s] for h, s in r.fingering])


def _window_check(job):
    seg, cfg, beam = job
    model = make_model(cfg.cost, cfg.chord_horizontal, cfg.span_scale)
    info = SegInfo(seg, cfg, model, bonus=False)
    a = beam_search(info, cfg, beam)
    b = beam_search(info, cfg, None)
    return dict(piece=seg.piece, events=len(seg), beam=a.cost, exact=b.cost, states_exact=b.states_max,
                match=bool(abs(a.cost - b.cost) < 1e-6))


def windows(segments, n, length_range=(8, 12), seed=0):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        s = segments[int(rng.integers(len(segments)))]
        L = int(rng.integers(length_range[0], length_range[1] + 1))
        if len(s) < L:
            continue
        st = int(rng.integers(0, len(s) - L + 1))
        out.append(Segment(s.piece, s.index, s.events[st:st + L]))
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pig-dir", required=True)
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--mode", choices=["lh", "given", "assign"], default="given")
    ap.add_argument("--cost", choices=["parncutt12", "balliauw15"], default="parncutt12")
    ap.add_argument("--chord-horizontal", choices=["all", "extremes"], default="all")
    ap.add_argument("--chords", choices=["full", "reduce"], default="full")
    ap.add_argument("--cross", action="store_true", help="crossing term in 2b (beam search)")
    ap.add_argument("--beam", type=int, default=10_000)
    ap.add_argument("--span-scale", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    ap.add_argument("--brute-check", type=int, default=0, help="N windows of <= 12 events: beam vs unpruned")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ctx = Context(args.pig_dir, PrepConfig(chords=args.chords), verbose=True)
    cfg = EnvConfig(mode=args.mode, cost=args.cost, chord_horizontal=args.chord_horizontal, cross=args.cross,
                    span_scale=args.span_scale, v_max=ctx.v_max)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if args.brute_check:
        res = {}
        for mode in ("given", "assign"):
            c = EnvConfig(**{**cfg.__dict__, "mode": mode, "cross": True})
            wins = windows(ctx.split.val + ctx.split.test, args.brute_check)
            with ProcessPoolExecutor(args.workers) as ex:
                rows = list(ex.map(_window_check, [(w, c, args.beam) for w in wins], chunksize=4))
            rate = float(np.mean([r["match"] for r in rows]))
            res[mode] = dict(windows=len(rows), beam=args.beam, agreement=rate, passed=rate >= 0.99,
                             max_states_unpruned=int(max(r["states_exact"] for r in rows)), rows=rows)
            print(f"{mode} (with crossing): beam {args.beam} finds the unpruned optimum on "
                  f"{100 * rate:.1f}% of {len(rows)} windows; max states unpruned {res[mode]['max_states_unpruned']}")
        out.write_text(json.dumps(res, indent=1))
        print(f"wrote {out}")
        return

    segs = ctx.segments(args.split)
    jobs = sorted(((s, cfg, args.beam) for s in segs), key=lambda j: -len(j[0]))
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(_one, jobs))
    rows.sort(key=lambda r: (r["piece"], r["index"]))
    pieces = {}
    for r in rows:
        p = pieces.setdefault(r["piece"], dict(cost=0.0, rules=0.0, notes=0, events=0, parncutt12=0.0, balliauw15=0.0))
        p["cost"] += r["cost"]
        p["rules"] += r["rules"]
        p["notes"] += r["notes"]
        p["events"] += r["events"]
        for n, v in r["rules_by_model"].items():
            p[n] += v
    summary = dict(segments=len(rows), events=sum(r["events"] for r in rows), notes=sum(r["notes"] for r in rows),
                   cost=sum(r["cost"] for r in rows), rules=sum(r["rules"] for r in rows),
                   cost_per_event=sum(r["cost"] for r in rows) / max(1, sum(r["events"] for r in rows)),
                   seconds_total=sum(r["seconds"] for r in rows), wall_seconds=time.time() - t0,
                   seconds_per_piece=sum(r["seconds"] for r in rows) / max(1, len(pieces)),
                   exact=all(r["exact"] for r in rows))
    json.dump(dict(config={**cfg.__dict__, "split": args.split, "beam": args.beam}, summary=summary,
                   pieces=pieces, segments=rows), open(out, "w"), indent=1)
    print(json.dumps(summary, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
