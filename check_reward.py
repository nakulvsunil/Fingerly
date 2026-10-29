"""Reward checks for Part C, no training (README §C.6.6).

    python check_reward.py --pig-dir data/PianoFingeringDataset_v1.2 --cost balliauw15 --table-a1 data/twohand/eer2025_table_a1.csv
    python check_reward.py --pig-dir data/PianoFingeringDataset_v1.2 --cost parncutt12 --mirror --chords --vmax

With no check flags, every check runs. Results go to results/twohand/reward_checks.md.

1. Table A1: pianists' fingerings of the 30 test pieces under balliauw15 (lowest-difficulty
   annotator), per piece against the 2025 paper's Table A1; pass = within 5% on >= 25 of 30.
2. Mirror: a mirrored left-hand line gets the right-hand breakdown; for balliauw15 the finger-swap
   reading equals interval mirroring.
3. Chord masks: share of the pianists' chord fingerings rejected by the masks of §C.5.3 (< 2%).
4. v_max: 95th percentile of human hand speed on training pieces; humans pay a time-leap cost on
   < 5% of events.
5. Part A regression: parncutt12 on Part A's right-hand pieces gives Part A's numbers.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from fingerly.pig import LEFT, RIGHT, Note, load_pig
from fingerly.twohand.actions import canonical
from fingerly.twohand.prepare import Event, PrepConfig, piece_events, TEST_IDS_2025
from fingerly.twohand.rules import (Balliauw15, Parncutt12, balliauw15_swap_pair, make_model, thumb_position,
                                    time_leap)

MIRROR_AXIS = 124  # p -> 124 - p reflects the keyboard around D4 (62): colours and positions map exactly


# --------------------------------------------------------------------------------------
# Human fingerings as hand streams
# --------------------------------------------------------------------------------------
def annotator_streams(notes: Sequence[Note], cfg: PrepConfig) -> Dict[str, List[Tuple[Event, Tuple[int, ...]]]]:
    """One annotator's own hands and fingers: {hand: [(event, fingers aligned with event.pitches)]}.
    Events are built from this annotator's hand assignment; unfingered notes are dropped."""
    evs = piece_events([[n for n in notes if n.finger]], cfg)
    out: Dict[str, list] = {LEFT: [], RIGHT: []}
    for e in evs:
        fingers = tuple(f for (_, f) in e.labels[0])
        out[e.hand].append((e, fingers))
    return out


def stream_cost(model, hand: str, stream) -> np.ndarray:
    return model.sequence(hand, [(e.pitches, f) for e, f in stream])


# --------------------------------------------------------------------------------------
# 1. Table A1
# --------------------------------------------------------------------------------------
def check_table_a1(pig, table_path: str, cfg: PrepConfig, horizontals=("all", "extremes"), rule7b="middle-thumb"):
    rows = list(csv.DictReader(open(table_path, encoding="utf-8")))
    paper = {r["pig_id"].zfill(3): float(r["human_difficulty_total"]) for r in rows}
    subset = [r["pig_id"].zfill(3) for r in rows if r["in_30_subset"].strip() == "1"]
    results = {}
    for hz in horizontals:
        model = Balliauw15(chord_horizontal=hz, rule7b=rule7b)
        per_piece = {}
        for k in subset:
            best = None
            for ann in pig[k]:
                st = annotator_streams(ann, cfg)
                cL, cR = stream_cost(model, LEFT, st[LEFT]).sum(), stream_cost(model, RIGHT, st[RIGHT]).sum()
                n = sum(e.size for e, _ in st[LEFT] + st[RIGHT])
                rec = dict(L=cL, R=cR, total=cL + cR, notes=n)
                if best is None or rec["total"] < best["total"]:
                    best = rec
            per_piece[k] = {**best, "paper": paper[k], "annotators": len(pig[k])}
        results[hz] = per_piece
    return results, subset


def within(x, ref, tol=0.05):
    return abs(x - ref) <= tol * abs(ref)


def table_a1_markdown(results, subset) -> Tuple[str, dict]:
    lines, summary = [], {}
    for hz, per in results.items():
        cands = {
            "whole piece": lambda r: r["total"],
            "right hand": lambda r: r["R"],
            "left hand": lambda r: r["L"],
            "max(hand)": lambda r: max(r["L"], r["R"]),
            "whole piece / notes × 100": lambda r: 100 * r["total"] / r["notes"],
        }
        summary[hz] = {name: int(sum(within(f(per[k]), per[k]["paper"]) for k in subset)) for name, f in cands.items()}
        ratios = np.array([per[k]["total"] / per[k]["paper"] for k in subset])
        summary[hz]["ratio_total_over_paper"] = (float(np.median(ratios)), float(ratios.min()), float(ratios.max()))
        # Diagnostics: which normalisation tracks the paper (correlation, scale, spread of the scale)
        p = np.array([per[k]["paper"] for k in subset])
        diag = {}
        for name, v in (("whole piece", [per[k]["total"] for k in subset]),
                        ("per note", [per[k]["total"] / per[k]["notes"] for k in subset])):
            v = np.array(v, float)
            r = p / v
            diag[name] = dict(corr=float(np.corrcoef(v, p)[0, 1]), paper_over_ours_median=float(np.median(r)),
                              cv=float(r.std() / r.mean()),
                              within5_after_scaling=int((abs(r / np.median(r) - 1) <= 0.05).sum()))
        summary[hz]["diagnostics"] = diag
        lines.append(f"\n### `--chord-horizontal {hz}`\n")
        lines.append("Pieces within 5% of Table A1 (pass: ≥ 25 of 30): "
                     + ", ".join(f"{n} **{v}**" for n, v in summary[hz].items() if isinstance(v, int)))
        for name, d in summary[hz]["diagnostics"].items():
            lines.append(f"\n- {name}: correlation with Table A1 {d['corr']:.3f}; Table A1 ÷ ours median "
                         f"{d['paper_over_ours_median']:.3f} (CV {d['cv']:.2f}); within 5% after that scaling: "
                         f"{d['within5_after_scaling']}/30")
        med, lo, hi = summary[hz]["ratio_total_over_paper"]
        lines.append(f"\nOurs (whole piece) ÷ Table A1: median {med:.2f}, range {lo:.2f}–{hi:.2f}.\n")
        lines.append("| PIG | Pianists (PIG) | Notes | Left | Right | Whole piece | Table A1 | Whole ÷ A1 |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for k in subset:
            r = per[k]
            lines.append(f"| {k} | {r['annotators']} | {r['notes']} | {r['L']:.1f} | {r['R']:.1f} | {r['total']:.1f} | "
                         f"{r['paper']:.3f} | {r['total'] / r['paper']:.1f} |")
    return "\n".join(lines), summary


# --------------------------------------------------------------------------------------
# 2. Mirror
# --------------------------------------------------------------------------------------
def check_mirror(pig, cfg: PrepConfig, n_pieces: int = 20) -> dict:
    """Right-hand streams of the first n pieces (annotator 1), mirrored around D4 and given to the
    left hand with the same fingers: breakdowns must be equal, for both models and both chord
    readings. Then balliauw15's finger-swap reading against interval mirroring on every pair."""
    out = {}
    for name in ("parncutt12", "balliauw15"):
        for hz in ("all", "extremes"):
            m = make_model(name, hz)
            bad = events = 0
            for k in list(pig)[:n_pieces]:
                st = annotator_streams(pig[k][0], cfg)[RIGHT]
                right = [(e.pitches, f) for e, f in st]
                left = [(tuple(sorted(MIRROR_AXIS - p for p in ps)),
                         tuple(f for _, f in sorted(zip((MIRROR_AXIS - p for p in ps), fs))))
                        for ps, fs in right]
                events += len(right)
                for deferred in (False, True):
                    bad += int(not np.allclose(m.sequence(RIGHT, right, deferred), m.sequence(LEFT, left, deferred)))
            out[f"{name}/{hz}"] = dict(pieces=n_pieces, events=events, mismatches=bad)
    # balliauw15: swap reading vs mirroring, all finger pairs, all moves within two octaves
    m = Balliauw15()
    bad = total = 0
    for f, g in itertools.product(range(1, 6), repeat=2):
        for pa in range(48, 73):
            for pb in range(pa - 24, pa + 25):
                total += 1
                mirrored = m._span_rules(f, m.hand_event(LEFT, [pa], [f]).vals[0], g, m.hand_event(LEFT, [pb], [g]).vals[0])
                bad += int(not np.allclose(mirrored, balliauw15_swap_pair(LEFT, f, pa, g, pb)))
    out["balliauw15 swap vs mirror"] = dict(cases=total, mismatches=bad)
    # deferred vs immediate charging on real two-hand data
    for name in ("parncutt12", "balliauw15"):
        m = make_model(name)
        bad = 0
        for k in list(pig)[:n_pieces]:
            st = annotator_streams(pig[k][0], cfg)
            for h in (LEFT, RIGHT):
                s = [(e.pitches, f) for e, f in st[h]]
                bad += int(not np.isclose(m.sequence(h, s, False).sum(), m.sequence(h, s, True).sum()))
        out[f"{name} deferred vs immediate"] = dict(streams=2 * n_pieces, mismatches=bad)
    return out


# --------------------------------------------------------------------------------------
# 3. Chord masks
# --------------------------------------------------------------------------------------
def check_chords(pig, piece_ids, cfg: PrepConfig, model_name: str) -> dict:
    """Pianists' chord fingerings (every annotator, own hands) against §C.5.3: repeated or
    missing fingers, non pitch-order assignment, chord reach, occupied fingers."""
    out = {}
    for hz in ("all",):
        m = make_model(model_name, hz)
        c = defaultdict(int)
        for k in piece_ids:
            for ann in pig[k]:
                st = annotator_streams(ann, cfg)
                for hand in (LEFT, RIGHT):
                    held: List[Tuple[int, float]] = []  # (finger, until)
                    for e, fs in st[hand]:
                        held = [(f, u) for f, u in held if e.onset < u - cfg.overlap_tol]
                        busy = {f for f, _ in held}
                        if e.size > 1:
                            c["chords"] += 1
                            reasons = []
                            if len(set(fs)) != len(fs):
                                reasons.append("repeated finger")
                            elif not canonical(fs, hand):
                                reasons.append("not pitch order")
                            elif not m.chord_reach_ok(m.hand_event(hand, e.pitches, fs)):
                                reasons.append("reach")
                            if busy & set(fs):
                                reasons.append("occupied")
                            if reasons:
                                c["rejected"] += 1
                                for r in reasons:
                                    c[r] += 1
                        elif busy & set(fs):
                            c["single occupied"] += 1
                        c["events"] += 1
                        for p, f, u in zip(e.pitches, fs, e.hold_until):
                            if u > -np.inf:
                                held.append((f, u))
        out[hz] = dict(c)
    return out


# --------------------------------------------------------------------------------------
# 4. v_max
# --------------------------------------------------------------------------------------
def hand_speeds(pig, piece_ids, cfg: PrepConfig):
    """|thumb shift| and onset gap between consecutive events of each hand, every annotator."""
    shifts, dts = [], []
    for k in piece_ids:
        for ann in pig[k]:
            st = annotator_streams(ann, cfg)
            for hand in (LEFT, RIGHT):
                prev = None
                for e, fs in st[hand]:
                    if len(set(fs)) != len(fs):
                        prev = None
                        continue
                    th = thumb_position(hand, e.pitches, fs)
                    if prev is not None:
                        shifts.append(abs(th - prev[0]))
                        dts.append(e.onset - prev[1])
                    prev = (th, e.onset)
    return np.array(shifts), np.array(dts)


def check_vmax(pig, train_ids, eval_ids, cfg: PrepConfig, pct: float = 95.0) -> dict:
    s, dt = hand_speeds(pig, train_ids, cfg)
    ok = dt > 0
    speed = s[ok] / dt[ok]
    v_all = float(np.percentile(speed, pct))
    moving = speed[s[ok] > 0]
    v_moving = float(np.percentile(moving, pct))
    res = dict(percentile=pct, v_max_all_transitions=v_all, v_max_moving_only=v_moving,
               train_transitions=int(ok.sum()), moving_share=float((s[ok] > 0).mean()))
    for name, v in (("all", v_all), ("moving", v_moving)):
        for split_name, ids in (("train", train_ids), ("test", eval_ids)):
            s2, dt2 = hand_speeds(pig, ids, cfg)
            pay = np.array([time_leap(a, b, v) > 0 for a, b in zip(s2, dt2)])
            res[f"pay_rate_{split_name}_vmax_{name}"] = float(pay.mean())
    return res


# --------------------------------------------------------------------------------------
# 5. Part A regression
# --------------------------------------------------------------------------------------
def check_part_a(pig_dir: str) -> dict:
    from fingerly.paper.pieces import load_piece
    from fingerly.paper.rules import difficulty

    m = Parncutt12()
    out = {}
    for env in ("V0", "V1", "V2", "V3"):
        try:
            pc = load_piece(env, pig_dir)
        except Exception as ex:  # noqa: BLE001
            out[env] = f"not loaded: {ex}"
            continue
        rows = []
        for lab in pc.labels:
            if any(f == 0 for f in lab):
                continue
            a = difficulty(pc.pitches, lab)
            stream = [((p,), (f,)) for p, f in zip(pc.pitches, lab)]
            b = m.sequence(RIGHT, stream).sum()
            c = m.sequence(RIGHT, stream, deferred=True).sum()
            rows.append((a, b, c))
        out[env] = dict(annotators=len(rows), part_a=[r[0] for r in rows], parncutt12=[r[1] for r in rows],
                        deferred=[r[2] for r in rows], equal=all(r[0] == r[1] == r[2] for r in rows))
    return out


# --------------------------------------------------------------------------------------
def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pig-dir", required=True)
    ap.add_argument("--cost", choices=["parncutt12", "balliauw15"], default="parncutt12",
                    help="cost model for the chord-mask check (Table A1 always uses balliauw15)")
    ap.add_argument("--table-a1", default=None, help="data/twohand/eer2025_table_a1.csv")
    ap.add_argument("--mirror", action="store_true")
    ap.add_argument("--chords", action="store_true")
    ap.add_argument("--vmax", action="store_true")
    ap.add_argument("--part-a", action="store_true")
    ap.add_argument("--rule7b", choices=["middle-thumb", "any-thumb"], default="middle-thumb")
    ap.add_argument("--chord-tol", type=float, default=0.03)
    ap.add_argument("--overlap-tol", type=float, default=0.05)
    ap.add_argument("--vmax-percentile", type=float, default=95.0)
    ap.add_argument("--out", default="results/twohand/reward_checks.md")
    args = ap.parse_args()
    run_all = not (args.table_a1 or args.mirror or args.chords or args.vmax or args.part_a)
    if run_all and args.table_a1 is None and Path("data/twohand/eer2025_table_a1.csv").exists():
        args.table_a1 = "data/twohand/eer2025_table_a1.csv"

    cfg = PrepConfig(chord_tol=args.chord_tol, overlap_tol=args.overlap_tol)
    pig = load_pig(args.pig_dir)
    from fingerly.twohand.prepare import make_twohand_split

    split = make_twohand_split(pig, cfg)
    md = ["# Part C reward checks (README §C.6.6)", ""]
    js = {}

    if args.table_a1:
        res, subset = check_table_a1(pig, args.table_a1, cfg, rule7b=args.rule7b)
        text, summ = table_a1_markdown(res, subset)
        md += ["## 1. Table A1 (balliauw15, lowest-difficulty annotator)", text]
        js["table_a1"] = summ
        print("Table A1:", json.dumps(summ, indent=1, default=float))
    if run_all or args.mirror:
        r = check_mirror(pig, cfg)
        md += ["", "## 2. Mirror test", "", "```", json.dumps(r, indent=1, default=float), "```"]
        js["mirror"] = r
        print("Mirror:", r)
    if run_all or args.chords:
        r = check_chords(pig, split.train_pieces + split.val_pieces + split.test_pieces, cfg, args.cost)
        c = r["all"]
        share = c.get("rejected", 0) / max(c.get("chords", 1), 1)
        md += ["", f"## 3. Chord masks ({args.cost})", "",
               f"{c.get('rejected', 0)} of {c.get('chords', 0)} pianists' chord fingerings rejected "
               f"(**{100 * share:.2f}%**, pass < 2%).", "", "```", json.dumps(c, indent=1, default=float), "```"]
        js["chords"] = {**c, "share": share}
        print("Chords:", c, f"share {share:.4f}")
    if run_all or args.vmax:
        r = check_vmax(pig, split.train_pieces, split.test_pieces, cfg, args.vmax_percentile)
        md += ["", "## 4. v_max", "", "```", json.dumps(r, indent=1, default=float), "```"]
        js["vmax"] = r
        print("v_max:", r)
    if run_all or args.part_a:
        r = check_part_a(args.pig_dir)
        md += ["", "## 5. Part A regression (parncutt12 vs Part A, right hand)", "", "```", json.dumps(r, indent=1, default=float), "```"]
        js["part_a"] = r
        print("Part A:", r)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(js, indent=1, default=float))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
