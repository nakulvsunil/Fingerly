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
from typing import Callable, Dict, List, Optional, Sequence

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
