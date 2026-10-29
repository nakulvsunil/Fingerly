"""Optimum reference M0 (README §C.8). A measuring stick only: never used to train the main method.

* `dp_hand` / `dp_segment`: exact second-order DP per hand over (finger set at t-1, finger set at t)
  using the environment's cost tables and static masks. Exact for 2a and for 2b without
  inter-hand terms (2b = the sum of the two hands' DPs).
* `beam_search`: search over full two-hand states for 2c (hand choice) and 2b with `--cross`.
  A state is each hand's last two (event, set), each hand's sounding time and the last hand;
  states with equal keys have identical futures and are merged, so with `beam=None` (no pruning)
  the search is exact ("brute force"). With a finite beam the lowest-cost states are kept.

Both minimise the environment's cost: rule points + time-leap + forced-shift (+ crossing).
Occupied fingers (held notes) are **not** enforced: they depend on choices further back than the
state holds. M0 is therefore a lower bound on what the environment allows, and the agent's gap to
it is never negative in 2a/2b.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .actions import N_SETS, SETS_OF_SIZE, fingers_of
from .costs import SHIFT_COST, HandTables
from .env import EnvConfig, SegInfo
from .rules import crossing, thumb_position, time_leap

INF = float("inf")


@dataclass
class SearchResult:
    cost: float  # environment cost (rules + extra), what the agent's return is compared with
    rules: float  # rule points only
    fingering: List[Tuple[str, int]]  # per event in segment order: (hand, global set)
    seconds: float
    exact: bool
    states_max: int = 0


# --------------------------------------------------------------------------------------
# Exact DP per hand (2a, 2b)
# --------------------------------------------------------------------------------------
def dp_hand(ht: HandTables) -> Tuple[float, List[int]]:
    """Minimum total cost C over one hand's events; returns (cost, local set per event)."""
    if not ht.tables:
        return 0.0, []
    V = np.zeros((1, 1))
    back: List[np.ndarray] = []
    for T in ht.tables:
        tot = V[:, :, None] + T.C  # (n2, n1, n0)
        arg = tot.argmin(0)
        best = tot.min(0)  # (n1, n0)
        best = np.where(T.allowed, best, INF)
        back.append(arg)
        V = best
    s1, s0 = np.unravel_index(int(np.argmin(V)), V.shape)
    cost = float(V[s1, s0])
    seq = [int(s0)]
    cur1, cur0 = int(s1), int(s0)
    for k in range(len(ht.tables) - 1, 0, -1):
        s2 = int(back[k][cur1, cur0])
        seq.append(cur1)
        cur1, cur0 = s2, cur1
    seq = seq[::-1]
    return cost, seq


def _rules_of(info: SegInfo, hand: str, events: List[int], local: List[int]) -> float:
    """Rule points (no extra terms) of a hand's chosen local sets, via the tables' D."""
    ht = info.tables.hands[hand]
    tot = 0.0
    for k, T in enumerate(ht.tables):
        s0 = local[k]
        s1 = local[k - 1] if k >= 1 else 0
        s2 = local[k - 2] if (k >= 2 and T.D.shape[0] > 1) else 0
        tot += float(T.D[s2, s1, s0])
    return tot


def dp_segment(info: SegInfo) -> SearchResult:
    t0 = time.time()
    fing: Dict[int, Tuple[str, int]] = {}
    cost = rules = 0.0
    for h, ht in info.tables.hands.items():
        c, loc = dp_hand(ht)
        cost += c
        rules += _rules_of(info, h, ht.events, loc)
        for k, i in enumerate(ht.events):
            fing[i] = (h, int(ht.tables[k].sets[loc[k]]))
    order = info.order
    return SearchResult(cost, rules, [fing[i] for i in order], time.time() - t0, True)


# --------------------------------------------------------------------------------------
# Beam search over two-hand states (2c, 2b with --cross)
# --------------------------------------------------------------------------------------
def beam_search(info: SegInfo, cfg: EnvConfig, beam: Optional[int] = 10_000) -> SearchResult:
    """Lowest-cost fingering (and hands, in 2c) of one segment. beam=None: no pruning (exact)."""
    t0 = time.time()
    model = info.model
    events = info.seg.events
    rule_cache: Dict[tuple, float] = {}
    trans_cache: Dict[tuple, np.ndarray] = {}

    def rules(i2, s2, i1, s1, i0, s0, hand):
        k = (hand, i2, s2, i1, s1, i0, s0)
        v = rule_cache.get(k)
        if v is None:
            H = info.hand_events
            ev0 = H(i0, hand)[s0]
            ev1 = H(i1, hand)[s1] if i1 >= 0 else None
            ev2 = H(i2, hand)[s2] if i2 >= 0 else None
            v = float(model.event_rules(ev2, ev1, ev0).sum())
            if ev1 is not None:
                v += float(model.next_terms(ev1, ev0).sum())
            rule_cache[k] = v
        return v

    def allowed_sets(i0, hand, i1, s1):
        """Local allowed sets of event i0 for `hand` after (i1, s1), and the shift flag."""
        k = (i0, hand, i1, s1)
        v = trans_cache.get(k)
        if v is None:
            H0 = info.hand_events(i0, hand)
            reach = info.reach(i0, hand)
            ev1 = info.hand_events(i1, hand)[s1] if i1 >= 0 else None
            trans = np.array([model.transition_ok(ev1, h) for h in H0])
            base = reach & trans
            shift = not base.any()
            allowed = reach.copy() if shift else base
            if not allowed.any():
                allowed[:] = True
            v = (np.flatnonzero(allowed), shift)
            trans_cache[k] = v
        return v

    def thumb(hand, i, s):
        return thumb_position(hand, events[i].pitches, fingers_of(int(SETS_OF_SIZE[events[i].size][s]), hand))

    # state: (L_i2, L_s2, L_i1, L_s1, R_i2, R_s2, R_i1, R_s1, L_sound, R_sound, last_hand)
    start = (-1, -1, -1, -1, -1, -1, -1, -1, -INF, -INF, "")
    layer: Dict[tuple, Tuple[float, float]] = {start: (0.0, 0.0)}  # state -> (cost, rules)
    backs: List[Dict[tuple, Tuple[tuple, str, int]]] = []
    states_max = 1
    order = info.order
    for pos, i0 in enumerate(order):
        e = events[i0]
        simult = pos > 0 and e.onset - info.onset[order[pos - 1]] <= cfg.chord_tol
        new: Dict[tuple, Tuple[float, float]] = {}
        back: Dict[tuple, Tuple[tuple, str, int]] = {}
        for st, (c, r) in layer.items():
            if cfg.mode == "assign" and e.size == 1:
                hands = [h for h, snd in (("L", st[8]), ("R", st[9])) if not snd > e.onset + cfg.overlap_tol]
                if simult and st[10]:
                    other = "R" if st[10] == "L" else "L"
                    hands = [h for h in hands if h == other] or [other]
                hands = hands or ["L", "R"]
            else:
                hands = [e.hand]
            for h in hands:
                o = 0 if h == "L" else 4
                i2, s2, i1, s1 = st[o:o + 4]
                loc, shift = allowed_sets(i0, h, i1, s1)
                dt = e.onset - events[i1].onset if i1 >= 0 else 0.0
                th1 = thumb(h, i1, s1) if i1 >= 0 else None
                for s0 in loc:
                    s0 = int(s0)
                    rr = rules(i2, s2, i1, s1, i0, s0, h)
                    th0 = thumb(h, i0, s0)
                    extra = SHIFT_COST if shift else 0.0
                    if th1 is not None:
                        extra += time_leap(th0 - th1, dt, cfg.v_max, cfg.w_time)
                    if cfg.crossing_on:
                        oth = "R" if h == "L" else "L"
                        oo = 0 if oth == "L" else 4
                        th_o = thumb(oth, st[oo + 2], st[oo + 3]) if st[oo + 2] >= 0 else None
                        tl, tr = (th0, th_o) if h == "L" else (th_o, th0)
                        extra += crossing(tl, tr, cfg.cross_margin, cfg.w_cross)
                    snd = max(st[8 if h == "L" else 9], e.offset)
                    hist = (i1, s1, i0, s0)
                    if h == "L":
                        ns = hist + st[4:8] + (snd, st[9], h)
                    else:
                        ns = st[0:4] + hist + (st[8], snd, h)
                    nc = c + rr + extra
                    old = new.get(ns)
                    if old is None or nc < old[0]:
                        new[ns] = (nc, r + rr)
                        back[ns] = (st, h, int(SETS_OF_SIZE[e.size][s0]))
        if beam is not None and len(new) > beam:
            keep = sorted(new, key=lambda k: new[k][0])[:beam]
            new = {k: new[k] for k in keep}
            back = {k: back[k] for k in keep}
        states_max = max(states_max, len(new))
        layer = new
        backs.append(back)
    best = min(layer, key=lambda k: layer[k][0])
    cost, rules_tot = layer[best]
    fing: List[Tuple[str, int]] = []
    st = best
    for b in reversed(backs):
        prev, h, s = b[st]
        fing.append((h, s))
        st = prev
    return SearchResult(cost, rules_tot, fing[::-1], time.time() - t0, beam is None, states_max)


def search_segment(info: SegInfo, cfg: EnvConfig, beam: Optional[int] = 10_000) -> SearchResult:
    """M0 for one segment: exact DP where it applies, beam search otherwise."""
    if cfg.mode in ("lh", "given") and not cfg.crossing_on:
        return dp_segment(info)
    return beam_search(info, cfg, beam)


def enumerate_segment(info: SegInfo, cfg: EnvConfig) -> float:
    """Plain enumeration of every action sequence through the environment (tiny segments only)."""
    from .env import Episode

    best = INF

    def rec(ep_actions):
        nonlocal best
        ep = Episode(cfg, info.model)
        ep.reset(info)
        for a in ep_actions:
            ep.step(a)
        if ep.done:
            cost = sum(r.rules + r.extra for r in ep.records)
            best = min(best, cost)
            return
        for a in np.flatnonzero(ep.mask()):
            rec(ep_actions + [int(a)])

    rec([])
    return best
