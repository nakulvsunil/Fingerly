"""Evaluation: PIG match rates and playability.

Match rates follow Nakamura et al. (2020). For one piece with annotators k = 1..K:

    Mgen  = mean_k  (matches with annotator k) / N
    Mhigh = max_k   (matches with annotator k) / N
    Msoft = (notes whose predicted finger was chosen by at least one annotator) / N

computed per piece and then averaged over pieces. A note counts as a match only if the
annotator gave it to the same hand and the same finger. Pieces with one annotator get
Mgen = Mhigh = Msoft = plain match rate.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .cost import CostConfig, sequence_cost
from .pig import Passage

Predictor = Callable[[Passage], List[int]]


def evaluate_predictor(
    predict: Predictor,
    passages: Sequence[Passage],
    cfg: Optional[CostConfig] = None,
) -> Dict[str, float]:
    cfg = cfg or CostConfig()
    per_piece = defaultdict(lambda: {"n": 0, "ann": None, "soft": 0})
    notes = effort = impossible = crossings = shifts = weak = stretch = 0.0
    labelled = False

    for p in passages:
        pred = predict(p)
        assert len(pred) == len(p), "predictor returned the wrong number of fingers"
        total, n_imp, details = sequence_cost(p.pitches, pred, p.hand, cfg)
        notes += len(p)
        effort += total
        impossible += n_imp
        crossings += sum(d.crossing for d in details)
        shifts += sum(d.shift for d in details)
        stretch += sum(d.stretch for d in details)
        weak += sum(f in (4, 5) for f in pred)

        if p.labels:
            labelled = True
            rec = per_piece[p.piece]
            lab = np.array(p.labels)  # K x N
            hit = lab == np.array(pred)[None, :]
            rec["n"] += len(p)
            rec["ann"] = hit.sum(1) if rec["ann"] is None else rec["ann"] + hit.sum(1)
            rec["soft"] += int(hit.any(0).sum())

    out = {
        "notes": float(notes),
        "effort_per_note": effort / max(notes, 1),
        "impossible_rate": impossible / max(notes, 1),
        "crossing_rate": crossings / max(notes, 1),
        "shift_rate": shifts / max(notes, 1),
        "weak_finger_rate": weak / max(notes, 1),
        "stretch_per_note": stretch / max(notes, 1),
    }
    if labelled:
        gen, high, soft, total_hits, total_n = [], [], [], 0.0, 0
        for rec in per_piece.values():
            rates = rec["ann"] / rec["n"]
            gen.append(rates.mean())
            high.append(rates.max())
            soft.append(rec["soft"] / rec["n"])
            total_hits += rates.mean() * rec["n"]
            total_n += rec["n"]
        out.update(
            {
                "Mgen": float(np.mean(gen)),
                "Mhigh": float(np.mean(high)),
                "Msoft": float(np.mean(soft)),
                "Mgen_note_weighted": total_hits / max(total_n, 1),
                "pieces": float(len(per_piece)),
            }
        )
    return out


def recombination_match(pred: Sequence[int], labels: Sequence[Sequence[int]]) -> float:
    """Mrec: match rate against the best fingering stitched together from the annotators.

    DP over which annotator is followed at each note, maximising matches. Following annotator k
    at note n-1 and k' at note n is allowed only if k and k' gave note n-1 the same finger, so
    the stitched sequence is locally one annotator's choice at every junction. This is our
    reading of the switching rule of Nakamura, Saito & Yoshii (2020); check it before citing.
    """
    lab = np.asarray(labels)  # K x N
    pred = np.asarray(pred)
    K, N = lab.shape
    hit = (lab == pred[None, :]).astype(float)
    score = hit[:, 0].copy()
    for n in range(1, N):
        new = np.empty(K)
        for k2 in range(K):
            ok = [k for k in range(K) if k == k2 or lab[k, n - 1] == lab[k2, n - 1]]
            new[k2] = max(score[k] for k in ok) + hit[k2, n]
        score = new
    return float(score.max() / N)


def match_rates(pred: Sequence[int], labels: Sequence[Sequence[int]]) -> Dict[str, float]:
    """Mgen, Mhigh, Msoft and Mrec of one fingering against K annotator fingerings."""
    lab = np.asarray(labels)
    hit = lab == np.asarray(pred)[None, :]
    rates = hit.mean(1)
    return {
        "Mgen": float(rates.mean()),
        "Mhigh": float(rates.max()),
        "Msoft": float(hit.any(0).mean()),
        "Mrec": recombination_match(pred, labels),
    }


def human_agreement(passages: Sequence[Passage]) -> Dict[str, float]:
    """Leave-one-annotator-out ceiling: score each annotator against the others, average.
    Only pieces with >= 2 annotators are used."""
    gen, high, soft = [], [], []
    by_piece = defaultdict(list)
    for p in passages:
        if len(p.labels) >= 2:
            by_piece[p.piece].append(p)
    for ps in by_piece.values():
        K = len(ps[0].labels)
        for k in range(K):
            n, hits, s = 0, np.zeros(K - 1), 0
            for p in ps:
                lab = np.array(p.labels)
                me = lab[k]
                others = np.delete(lab, k, axis=0)
                valid = me > 0
                h = (others == me[None, :]) & valid[None, :]
                n += len(me)
                hits += h.sum(1)
                s += int(h.any(0).sum())
            gen.append((hits / n).mean())
            high.append((hits / n).max())
            soft.append(s / n)
    if not gen:
        return {}
    return {"Mgen": float(np.mean(gen)), "Mhigh": float(np.mean(high)), "Msoft": float(np.mean(soft))}


# ======================================================================================
# Part C: greedy rollouts, hand-aware match rates, fingering replay (README §C.11.1-C.11.3)
# ======================================================================================
from .twohand.actions import fingers_of  # noqa: E402
from .twohand.env import Episode, SegInfo, SegmentPool  # noqa: E402
from .twohand.prepare import Segment  # noqa: E402
from .twohand.rules import CostModel, merged_cost, thumb_position, time_leap  # noqa: E402

BatchPolicy = Callable[[np.ndarray, np.ndarray], np.ndarray]  # (obs[B, D], mask[B, A]) -> actions[B]

def rollout(pool: SegmentPool, policy: BatchPolicy, infos: Optional[Sequence[SegInfo]] = None) -> List[Episode]:
    """Run the greedy policy on every segment of the pool at once (one batched forward per step)."""
    infos = list(infos if infos is not None else pool.infos)
    eps = [Episode(pool.cfg, pool.model) for _ in infos]
    obs = np.zeros((len(infos), pool.cfg.obs_dim), np.float32)
    masks = np.zeros((len(infos), pool.cfg.n_actions), bool)
    for k, (ep, info) in enumerate(zip(eps, infos)):
        obs[k], masks[k] = ep.reset(info)
    live = [k for k, ep in enumerate(eps) if not ep.done]
    while live:
        acts = policy(obs[live], masks[live])
        nxt = []
        for k, a in zip(live, acts):
            ep = eps[k]
            ep.step(int(a))
            if not ep.done:
                obs[k], masks[k] = ep.obs(), ep.mask()
                nxt.append(k)
        live = nxt
    return eps


def note_codes(ep: Episode):
    """Per visited note: predicted code, per-annotator codes (K x N), chord flag, hand."""
    seg = ep.info.seg
    pred, labs, chord, hands = [], [], [], []
    K = seg.n_annotators
    for r in ep.records:
        e = seg.events[r.event]
        fs = fingers_of(r.set, r.hand)
        for j, f in enumerate(fs):
            pred.append(f if r.hand == "R" else -f)
            row = []
            for a in range(K):
                h, g = e.labels[a][j]
                row.append(0 if not g or h is None else (g if h == "R" else -g))
            labs.append(row)
            chord.append(e.size > 1)
            hands.append(r.hand)
    lab = np.array(labs, int).T if labs else np.zeros((K, 0), int)
    return np.array(pred, int), lab, np.array(chord, bool), np.array(hands)


def piece_match(eps: Sequence[Episode]) -> Dict[str, Dict[str, float]]:
    """{piece: {Mgen, Mhigh, Msoft, Mrec, notes}} over the episodes of each piece."""
    by = defaultdict(list)
    for ep in eps:
        if ep.info.seg.n_annotators:
            by[ep.info.seg.piece].append(note_codes(ep))
    out = {}
    for piece, parts in by.items():
        pred = np.concatenate([p[0] for p in parts])
        lab = np.concatenate([p[1] for p in parts], axis=1)
        if not len(pred):
            continue
        m = match_rates(pred, lab)
        m["notes"] = float(len(pred))
        out[piece] = m
    return out


def summarize(eps: Sequence[Episode], m0: Optional[Dict[tuple, float]] = None, success_gap: float = 0.10) -> Dict[str, float]:
    """Validation/test summary: per-event return (rule reward only), difficulty, match rates,
    infeasible rate, and, with M0 costs {(piece, index): cost}, the gap and success rates."""
    events = sum(len(ep.records) for ep in eps)
    notes = sum(ep.info.seg.events[r.event].size for ep in eps for r in ep.records)
    rules = sum(r.rules for ep in eps for r in ep.records)
    extra = sum(r.extra for ep in eps for r in ep.records)
    infeas = sum(r.infeasible or r.forced for ep in eps for r in ep.records)
    ret = -(rules + extra)
    out = dict(events=float(events), notes=float(notes), return_per_event=ret / max(events, 1),
               difficulty_per_note=rules / max(notes, 1), cost=rules + extra,
               infeasible_rate=infeas / max(events, 1))
    pm = piece_match(eps)
    if pm:
        for k in ("Mgen", "Mhigh", "Msoft", "Mrec"):
            out[k] = float(np.mean([v[k] for v in pm.values()]))
    if m0:
        gaps, succ, play = [], [], []
        agent_tot = opt_tot = 0.0
        for ep in eps:
            key = (ep.info.seg.piece, ep.info.seg.index)
            if key not in m0:
                continue
            c = sum(r.rules + r.extra for r in ep.records)
            opt = m0[key]
            agent_tot += c
            opt_tot += opt
            gap = (c - opt) / opt if opt > 0 else (0.0 if c <= 1e-9 else np.inf)
            ok_play = not any(r.infeasible or r.forced for r in ep.records)
            gaps.append(gap)
            play.append(ok_play)
            succ.append(ok_play and gap <= success_gap)
        if gaps:
            out["gap"] = (agent_tot - opt_tot) / max(opt_tot, 1e-9)
            out["SR"] = float(np.mean(succ))
            out["SR_play"] = float(np.mean(play))
    return out


def replay(pool: SegmentPool, fingerings, infos: Optional[Sequence[SegInfo]] = None) -> List[Episode]:
    """Replay fixed fingerings {(piece, index): [(hand, global set), ...]} (or a function
    info -> list) through unmasked environments: moves the masks would forbid are kept and counted
    as infeasible. Used for S2, Q-learning, M0 and the pianists."""
    import dataclasses

    from .twohand.actions import N_SETS

    cfg = dataclasses.replace(pool.cfg, use_mask=False, lam=0.0)
    out = []
    for info in (infos if infos is not None else pool.infos):
        fing = fingerings(info) if callable(fingerings) else fingerings.get((info.seg.piece, info.seg.index))
        if fing is None:
            continue
        ep = Episode(cfg, pool.model)
        ep.reset(info)
        for h, s in fing:
            ep.step(int(s) + (N_SETS if cfg.mode == "assign" and h == "R" else 0))
        out.append(ep)
    return out


# ======================================================================================
# Part C: every metric of one method on one split, pianists included (README §C.11.2-C.11.4)
# ======================================================================================
ERAS = {
    "Baroque": {"Bach", "Scarlatti"},
    "Classical": {"Mozart", "Beethoven"},
    "Romantic": {"Chopin", "Schumann", "Schubert", "Mendelssohn", "Liszt", "Brahms", "Grieg", "Tchaikovsky",
                 "Dvorak", "Saint-Saens", "Mussorgsky"},
    "Late Romantic / 20th c.": {"Debussy", "Ravel", "Rachmaninoff", "Scriabin", "Satie", "Faure", "Joplin",
                                "Bartok", "Albeniz"},
}

Item = Tuple[int, str, Tuple[int, ...], Tuple[int, ...], float]  # (event, hand, pitches, fingers, onset)


def composer_eras(pig_dir: str) -> Dict[str, str]:
    """{piece id: era} from PIG's List.csv."""
    import csv
    from .pig import find_fingering_dir

    root = find_fingering_dir(pig_dir).parent
    out = {}
    path = root / "List.csv"
    if not path.exists():
        return out
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            comp = row.get("Composer", "").strip()
            era = next((e for e, names in ERAS.items() if comp in names), "Other")
            out[row["Id"].strip().zfill(3)] = era
    return out


# --------------------------------------------------------------------------------------
# Fingered event lists
# --------------------------------------------------------------------------------------
def agent_items(ep: Episode) -> List[Item]:
    seg = ep.info.seg
    return [(r.event, r.hand, seg.events[r.event].pitches, fingers_of(r.set, r.hand), seg.events[r.event].onset)
            for r in ep.records]


def human_items(seg: Segment, order: Sequence[int], k: int) -> List[Item]:
    """Annotator k's fingering of the visited events, split by hand where the annotator split a
    chord across hands; unfingered notes are dropped."""
    out = []
    for i in order:
        e = seg.events[i]
        by = defaultdict(list)
        for p, (h, f) in zip(e.pitches, e.labels[k]):
            if h is not None and f:
                by[h].append((p, f))
        for h in ("L", "R"):
            if by[h]:
                ps, fs = zip(*sorted(by[h]))
                out.append((i, h, tuple(ps), tuple(fs), e.onset))
    return out


def rule_total(items: Sequence[Item], model: CostModel) -> float:
    return float(merged_cost(model, [(h, p, f) for _, h, p, f, _ in items]).sum())


def playability(items: Sequence[Item], model: CostModel, v_max: float) -> Dict[str, float]:
    """Counts behind the rates of §C.11.3 (divide by the denominators returned)."""
    c = defaultdict(float)
    last: Dict[str, Optional[Item]] = {"L": None, "R": None}
    thumb = {"L": None, "R": None}
    prev_single_hand = None
    for it in items:
        i, h, ps, fs, on = it
        c["notes"] += len(ps)
        c["events"] += 1
        c["weak"] += sum(f in (4, 5) for f in fs)
        c["thumb_black"] += sum(f == 1 and p % 12 in (1, 3, 6, 8, 10) for p, f in zip(ps, fs))
        prev = last[h]
        th = thumb_position(h, ps, fs) if len(set(fs)) == len(fs) else None
        if prev is not None:
            c["transitions"] += 1
            ev1 = model.hand_event(h, prev[2], prev[3])
            ev0 = model.hand_event(h, ps, fs)
            if model.outside_comf(ev1, ev0):
                c["large_stretch"] += 1
            if th is not None and thumb[h] is not None and time_leap(th - thumb[h], on - prev[4], v_max) > 0:
                c["time_leap"] += 1
        last[h] = it
        thumb[h] = th
        if thumb["L"] is not None and thumb["R"] is not None:
            c["both_placed"] += 1
            if thumb["L"] > thumb["R"]:
                c["crossing"] += 1
        if len(ps) == 1:
            if prev_single_hand is not None:
                c["single_pairs"] += 1
                c["hand_switch"] += prev_single_hand != h
            prev_single_hand = h
        else:
            prev_single_hand = None
    return dict(c)


def rates(c: Dict[str, float]) -> Dict[str, float]:
    d = lambda a, b: c.get(a, 0.0) / max(c.get(b, 0.0), 1.0)  # noqa: E731
    return dict(weak_finger_rate=d("weak", "notes"), thumb_on_black_rate=d("thumb_black", "notes"),
                large_stretch_rate=d("large_stretch", "transitions"), time_leap_rate=d("time_leap", "transitions"),
                crossing_rate=d("crossing", "both_placed"), hand_switch_rate=d("hand_switch", "single_pairs"))


def add_counts(a: Dict[str, float], b: Dict[str, float]):
    for k, v in b.items():
        a[k] = a.get(k, 0.0) + v


# --------------------------------------------------------------------------------------
# Match rates
# --------------------------------------------------------------------------------------
def codes(items: Sequence[Item], seg: Segment):
    """Per note: predicted code (+f right, -f left), annotator codes (K x N), event hand, chord flag,
    predicted hand, annotator hands."""
    K = seg.n_annotators
    pred, lab, ehand, chord, phand, lhand = [], [], [], [], [], []
    for i, h, ps, fs, _ in items:
        e = seg.events[i]
        pos = {p: j for j, p in enumerate(e.pitches)}
        for p, f in zip(ps, fs):
            j = pos[p]
            pred.append(f if h == "R" else -f)
            row, hrow = [], []
            for a in range(K):
                hh, g = e.labels[a][j]
                row.append(0 if not g or hh is None else (g if hh == "R" else -g))
                hrow.append(hh or "")
            lab.append(row)
            lhand.append(hrow)
            ehand.append(e.hand)
            chord.append(e.size > 1)
            phand.append(h)
    return (np.array(pred, int), np.array(lab, int).reshape(-1, K).T, np.array(ehand), np.array(chord, bool),
            np.array(phand), np.array(lhand).reshape(-1, K).T)


def piece_scores(per_piece_items: Dict[str, List[Tuple[Segment, List[Item]]]]) -> Dict[str, Dict[str, float]]:
    """Per piece: Mgen/Mhigh/Msoft/Mrec overall, per hand, singles vs chords; hand accuracy;
    chord exact match."""
    out = {}
    for piece, parts in per_piece_items.items():
        cs = [codes(items, seg) for seg, items in parts if seg.n_annotators]
        if not cs or not sum(len(c[0]) for c in cs):
            continue
        pred = np.concatenate([c[0] for c in cs])
        lab = np.concatenate([c[1] for c in cs], axis=1)
        eh = np.concatenate([c[2] for c in cs])
        ch = np.concatenate([c[3] for c in cs])
        ph = np.concatenate([c[4] for c in cs])
        lh = np.concatenate([c[5] for c in cs], axis=1)
        m = dict(match_rates(pred, lab), notes=float(len(pred)))
        for name, sel in (("R", eh == "R"), ("L", eh == "L"), ("single", ~ch), ("chord", ch)):
            if sel.any():
                for k, v in match_rates(pred[sel], lab[:, sel]).items():
                    m[f"{k}_{name}"] = v
        m["hand_accuracy"] = float((lh == ph[None, :]).mean(1).mean())
        # chord exact match: every note of the chord matches one annotator (hand and finger)
        n_ch = hit = 0
        for (seg, items) in parts:
            for i, h, ps, fs, _ in items:
                e = seg.events[i]
                if e.size < 2 or len(ps) != e.size:
                    continue
                n_ch += 1
                code = tuple(f if h == "R" else -f for f in fs)
                for row in e.labels:
                    if tuple(0 if not g or hh is None else (g if hh == "R" else -g) for hh, g in row) == code:
                        hit += 1
                        break
        m["chords"] = float(n_ch)
        m["chord_exact"] = hit / n_ch if n_ch else float("nan")
        out[piece] = m
    return out


def annotator_agreement(segs_by_piece: Dict[str, List[Tuple[Segment, Sequence[int]]]]) -> Dict[str, float]:
    """Leave-one-annotator-out ceiling on pieces with >= 2 annotators: each annotator scored
    against the others (Mgen, Mhigh, Msoft, hand accuracy)."""
    vals = defaultdict(list)
    for piece, parts in segs_by_piece.items():
        K = parts[0][0].n_annotators
        if K < 2:
            continue
        for k in range(K):
            pred, lab_all, hand_p, hand_l = [], [], [], []
            for seg, order in parts:
                for i in order:
                    e = seg.events[i]
                    for j in range(e.size):
                        h, f = e.labels[k][j]
                        if h is None or not f:
                            continue
                        pred.append(f if h == "R" else -f)
                        row = [0 if not g or hh is None else (g if hh == "R" else -g)
                               for a, (hh, g) in enumerate(e.labels[x][j] for x in range(K)) if a != k]
                        lab_all.append(row)
                        hand_p.append(h)
                        hand_l.append([e.labels[x][j][0] or "" for x in range(K) if x != k])
            if not pred:
                continue
            m = match_rates(np.array(pred), np.array(lab_all).T)
            for key in ("Mgen", "Mhigh", "Msoft"):
                vals[key].append(m[key])
            vals["hand_accuracy"].append(float((np.array(hand_l).T == np.array(hand_p)[None, :]).mean()))
    return {k: float(np.mean(v)) for k, v in vals.items()}


# --------------------------------------------------------------------------------------
# One method on one split
# --------------------------------------------------------------------------------------
@dataclass
class HumanRef:
    """Per piece: the pianists' difficulty (mean over annotators) under both models, and the
    pianists' playability counts. Computed once per split and cost settings."""

    difficulty: Dict[str, Dict[str, float]]  # model -> piece -> mean over annotators
    difficulty_min: Dict[str, Dict[str, float]]  # model -> piece -> lowest annotator
    counts: Dict[str, float]
    per_event_cost_p95: float
    agreement: Dict[str, float]


def human_reference(infos: Sequence[SegInfo], models: Dict[str, CostModel], v_max: float) -> HumanRef:
    by = defaultdict(list)
    for info in infos:
        by[info.seg.piece].append(info)
    diff = {n: {} for n in models}
    dmin = {n: {} for n in models}
    counts: Dict[str, float] = {}
    per_event = []
    main = next(iter(models.values()))
    for piece, parts in by.items():
        K = parts[0].seg.n_annotators
        if not K:
            continue
        tot = {n: np.zeros(K) for n in models}
        for k in range(K):
            for info in parts:
                items = human_items(info.seg, info.order, k)
                for n, m in models.items():
                    tot[n][k] += rule_total(items, m)
                add_counts(counts, playability(items, main, v_max))
                if k == 0:
                    per_event += _per_event_costs(items, main)
        for n in models:
            diff[n][piece] = float(tot[n].mean())
            dmin[n][piece] = float(tot[n].min())
    agreement = annotator_agreement({p: [(i.seg, i.order) for i in parts] for p, parts in by.items()})
    return HumanRef(diff, dmin, counts, float(np.percentile(per_event, 95)) if per_event else np.inf, agreement)


def _per_event_costs(items: Sequence[Item], model: CostModel) -> List[float]:
    """Rule points charged at each event (per-hand history), for the 'very hard' threshold."""
    out = []
    hist = {"L": [], "R": []}
    for _, h, ps, fs, _ in items:
        ev0 = model.hand_event(h, ps, fs)
        hh = hist[h]
        out.append(float(model.event_rules(hh[-2] if len(hh) > 1 else None, hh[-1] if hh else None, ev0).sum()))
        hh.append(ev0)
    return out


def method_metrics(eps: Sequence[Episode], models: Dict[str, CostModel], human: HumanRef, v_max: float,
                   m0: Optional[Dict[tuple, float]] = None, success_gap: float = 0.10) -> Dict:
    """All problem-specific and per-segment RL metrics of one run (one seed) on one split."""
    main_name = next(iter(models))
    main = models[main_name]
    seg_rows = []
    per_piece_items = defaultdict(list)
    counts: Dict[str, float] = {}
    diff = {n: defaultdict(float) for n in models}
    infeas = forced = events = 0
    breakdowns = 0
    unreachable = chord_events = 0
    for ep in eps:
        seg, info = ep.info.seg, ep.info
        items = agent_items(ep)
        per_piece_items[seg.piece].append((seg, items))
        add_counts(counts, playability(items, main, v_max))
        for n, m in models.items():
            diff[n][seg.piece] += rule_total(items, m)
        cost = sum(r.rules + r.extra for r in ep.records)
        n_inf = sum(r.infeasible or r.forced for r in ep.records)
        infeas += n_inf
        forced += sum(r.forced for r in ep.records)
        events += len(ep.records)
        for r in ep.records:
            e = seg.events[r.event]
            if e.size > 1:
                chord_events += 1
                if not info.reach(r.event, r.hand).any():
                    unreachable += 1
        breakdowns += _breakdowns(ep, items, main, human.per_event_cost_p95)
        row = dict(piece=seg.piece, index=seg.index, events=len(ep.records),
                   notes=sum(seg.events[r.event].size for r in ep.records), cost=cost,
                   ret_per_event=-cost / max(1, len(ep.records)), infeasible=n_inf)
        if m0 is not None and (seg.piece, seg.index) in m0:
            opt = m0[(seg.piece, seg.index)]
            row["m0"] = opt
            row["regret"] = cost - opt
            row["gap"] = (cost - opt) / opt if opt > 0 else 0.0
            row["success"] = bool(n_inf == 0 and row["gap"] <= success_gap)
            row["success_play"] = bool(n_inf == 0)
        seg_rows.append(row)
    pieces = piece_scores(per_piece_items)
    notes_by_piece = defaultdict(float)
    ev_by_piece, inf_by_piece, ret_by_piece = defaultdict(float), defaultdict(float), defaultdict(float)
    for r in seg_rows:
        notes_by_piece[r["piece"]] += r["notes"]
        ev_by_piece[r["piece"]] += r["events"]
        inf_by_piece[r["piece"]] += r["infeasible"]
        ret_by_piece[r["piece"]] -= r["cost"]
    for p in ev_by_piece:
        pieces.setdefault(p, {"notes": notes_by_piece[p]})
        pieces[p]["infeasible_rate"] = inf_by_piece[p] / max(ev_by_piece[p], 1)
        pieces[p]["return_per_event"] = ret_by_piece[p] / max(ev_by_piece[p], 1)
        pieces[p]["return_total"] = ret_by_piece[p]
    for p in pieces:
        for n in models:
            pieces[p][f"difficulty_{n}"] = diff[n][p]
            pieces[p][f"difficulty_per_note_{n}"] = diff[n][p] / max(notes_by_piece[p], 1)
            h = human.difficulty.get(n, {}).get(p)
            if h:
                pieces[p][f"d_rel_{n}"] = diff[n][p] / h
    out = dict(segments=seg_rows, pieces=pieces)
    agg = {}
    keys = sorted({k for v in pieces.values() for k in v if isinstance(v[k], float)})
    for k in keys:
        vals = [v[k] for v in pieces.values() if k in v and not np.isnan(v[k])]
        if vals:
            agg[k] = float(np.mean(vals))
    total_notes = sum(r["notes"] for r in seg_rows)
    for n in models:
        tot = sum(diff[n].values())
        agg[f"difficulty_total_{n}"] = tot
        agg[f"difficulty_per_note_{n}"] = tot / max(total_notes, 1)
        common = [p for p in diff[n] if human.difficulty.get(n, {}).get(p)]
        if common:
            agg[f"d_rel_{n}"] = sum(diff[n][p] for p in common) / sum(human.difficulty[n][p] for p in common)
            agg[f"d_rel_{n}_piece_mean"] = float(np.mean([diff[n][p] / human.difficulty[n][p] for p in common]))
    agg.update(rates(counts))
    agg["infeasible_rate"] = infeas / max(events, 1)
    agg["forced_rate"] = forced / max(events, 1)
    agg["unreachable_chord_rate"] = unreachable / max(chord_events, 1)
    agg["sequence_breakdown_rate"] = breakdowns / max(events, 1)
    agg["return_per_event"] = float(np.mean([r["ret_per_event"] for r in seg_rows]))
    agg["return_per_event_median"] = float(np.median([r["ret_per_event"] for r in seg_rows]))
    agg["return_total"] = -sum(r["cost"] for r in seg_rows)
    if m0 is not None and seg_rows and "gap" in seg_rows[0]:
        rows = [r for r in seg_rows if "gap" in r]
        agg["gap"] = sum(r["cost"] for r in rows) / sum(r["m0"] for r in rows) - 1
        agg["SR"] = float(np.mean([r["success"] for r in rows]))
        agg["SR_play"] = float(np.mean([r["success_play"] for r in rows]))
        agg["regret_total"] = float(sum(r["regret"] for r in rows))
        agg["negative_regret_segments"] = int(sum(r["regret"] < -1e-9 for r in rows))
    out["summary"] = agg
    return out


def _breakdowns(ep: Episode, items: Sequence[Item], model: CostModel, p95: float) -> int:
    """Events whose fingering matches a pianist while the hand's next transition is infeasible or
    very hard (rule cost above the pianists' 95th percentile per event)."""
    seg = ep.info.seg
    recs = ep.records
    costs = _per_event_costs(items, model)
    n = 0
    nxt_of = {}
    last = {}
    for k in range(len(recs) - 1, -1, -1):
        h = recs[k].hand
        nxt_of[k] = last.get(h)
        last[h] = k
    for k, r in enumerate(recs):
        e = seg.events[r.event]
        fs = fingers_of(r.set, r.hand)
        code = tuple((r.hand, f) for f in fs)
        if not any(tuple(row) == code for row in e.labels):
            continue
        j = nxt_of[k]
        if j is None:
            continue
        if recs[j].infeasible or recs[j].forced or costs[j] > p95:
            n += 1
    return n


# ======================================================================================
# Part C: statistics (README §C.11.8)
# ======================================================================================
def wilson(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a success rate."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (float(centre - half), float(centre + half))


def bootstrap_ci(values: Sequence[float], n_boot: int = 10_000, stat=np.mean, seed: int = 0,
                 alpha: float = 0.05) -> Tuple[float, float]:
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    s = np.apply_along_axis(stat, 1, v[idx]) if stat is not np.mean else v[idx].mean(1)
    return (float(np.quantile(s, alpha / 2)), float(np.quantile(s, 1 - alpha / 2)))


def iqm(x: np.ndarray) -> float:
    """Interquartile mean: the mean of the middle 50% of the values."""
    x = np.sort(np.asarray(x, float).ravel())
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan")
    lo, hi = int(np.floor(0.25 * len(x))), int(np.ceil(0.75 * len(x)))
    return float(x[lo:max(hi, lo + 1)].mean())


def stratified_bootstrap_iqm(scores: np.ndarray, n_boot: int = 10_000, seed: int = 0,
                             alpha: float = 0.05) -> Dict[str, float]:
    """scores: runs x tasks (seeds x pieces). Resample runs independently within each task
    (Agarwal et al., 2021) and recompute the IQM over all scores."""
    s = np.asarray(scores, float)
    rng = np.random.default_rng(seed)
    R, T = s.shape
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, R, size=(R, T))
        boots[b] = iqm(np.take_along_axis(s, idx, 0))
    return dict(iqm=iqm(s), lo=float(np.quantile(boots, alpha / 2)), hi=float(np.quantile(boots, 1 - alpha / 2)))


def wilcoxon_pairs(per_piece: Dict[str, Dict[str, float]], higher_better: bool = True) -> List[Dict]:
    """Wilcoxon signed-rank test over pieces for every pair of methods.
    per_piece: {method: {piece: value}} (values averaged over seeds)."""
    from scipy.stats import wilcoxon

    out = []
    for a, b in combinations(sorted(per_piece), 2):
        common = sorted(set(per_piece[a]) & set(per_piece[b]))
        x = np.array([per_piece[a][p] for p in common], float)
        y = np.array([per_piece[b][p] for p in common], float)
        ok = ~(np.isnan(x) | np.isnan(y))
        x, y = x[ok], y[ok]
        row = dict(a=a, b=b, pieces=int(len(x)), mean_a=float(x.mean()) if len(x) else np.nan,
                   mean_b=float(y.mean()) if len(y) else np.nan)
        if len(x) >= 5 and np.any(x != y):
            res = wilcoxon(x, y)
            row["p"] = float(res.pvalue)
            row["significant"] = bool(res.pvalue < 0.05)
            better = (x.mean() > y.mean()) == higher_better
            row["better"] = a if better else b
        else:
            row["p"] = float("nan")
            row["significant"] = False
        out.append(row)
    return out


def seed_stability(values: Sequence[float]) -> Dict[str, float]:
    v = np.asarray(values, float)
    m = float(v.mean())
    return dict(mean=m, std=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                cv=float(v.std(ddof=1) / abs(m)) if len(v) > 1 and m else 0.0,
                min=float(v.min()), max=float(v.max()), median=float(np.median(v)), n=int(len(v)))


def first_sustained(steps: Sequence[float], values: Sequence[float], ok, k: int = 3) -> Optional[float]:
    """First step N at which ok(value) holds for k consecutive evaluations (convergence N*)."""
    run = 0
    for i, v in enumerate(values):
        run = run + 1 if ok(v) else 0
        if run == k:
            return float(steps[i - k + 1])
    return None


def curve_auc(steps: Sequence[float], values: Sequence[float], lo: float, hi: float, horizon: float) -> float:
    """Area under the normalised learning curve up to `horizon` steps, in [0, 1]. Values are
    normalised with (v - lo) / (hi - lo) (lo = random policy, hi = M0) and clipped to [0, 1]."""
    s = np.asarray(steps, float)
    v = np.clip((np.asarray(values, float) - lo) / max(hi - lo, 1e-9), 0, 1)
    keep = s <= horizon
    s, v = np.concatenate([[0.0], s[keep]]), np.concatenate([[0.0], v[keep]])
    if s[-1] < horizon:  # hold the last value to the horizon
        s, v = np.append(s, horizon), np.append(v, v[-1])
    return float(np.trapezoid(v, s) / horizon)


def value_at(steps: Sequence[float], values: Sequence[float], at: float) -> Optional[float]:
    s = np.asarray(steps, float)
    if not len(s) or s[0] > at:
        return None
    return float(np.asarray(values, float)[np.searchsorted(s, at, side="right") - 1])
