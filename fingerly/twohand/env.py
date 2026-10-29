"""Two-hand fingering environments (README §C.3.1, §C.5, §C.6): modes 2a, 2b and 2c.

    2a  --mode lh      the left hand's events of a segment, alone                     31 actions
    2b  --mode given   both hands, merged in onset order, each event's hand given     31 actions
    2c  --mode assign  both hands; single-note events also choose the hand            62 actions
                       (index = hand * 31 + finger set, hand 0 = left, 1 = right)

One episode is one segment, one step per event. Rewards:

    R_t = -(alpha * D_t + c_time + c_shift + c_cross) + lam * B_t   (- penalty with --no-mask)

2a/2b look the rule and time-leap costs up in precomputed tables (costs.py); 2c computes them per
step with a cache, charging next-note rule parts when that hand's next event arrives (§C.6.4).
`Episode` is one environment; `VecEnv` steps many at once; `TwoHandEnv` is a Gymnasium wrapper with
`action_masks()` for sb3-contrib.
"""

from __future__ import annotations

import hashlib
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .actions import N_SETS, SET_BITS, SETS_OF_SIZE, fingers_of
from .costs import SHIFT_COST, SegmentTables, build_segment_tables
from .features import key_of, iv, obs_dim, tclip
from .prepare import Event, Segment
from .rules import CostModel, HandEvent, crossing, make_model, thumb_position, time_leap

MODES = ("lh", "given", "assign")
HAND_IDX = {"L": 0, "R": 1}


@dataclass
class EnvConfig:
    mode: str = "given"
    cost: str = "parncutt12"
    chord_horizontal: str = "all"
    span_scale: float = 1.0
    v_max: float = 38.93
    w_time: float = 1.0
    w_cross: float = 1.0
    cross_margin: float = 2.0
    cross: bool = False  # crossing term in 2b (always on in 2c)
    alpha: float = 1.0
    lam: float = 0.0
    lookahead: int = 4
    hand_pos: bool = True
    use_mask: bool = True
    impossible_penalty: float = 10.0
    occupied: bool = True  # mask fingers holding a key (False only for consistency checks)
    chord_tol: float = 0.03
    overlap_tol: float = 0.05

    @property
    def n_actions(self) -> int:
        return 2 * N_SETS if self.mode == "assign" else N_SETS

    @property
    def obs_dim(self) -> int:
        return obs_dim(self.lookahead, self.hand_pos)

    @property
    def crossing_on(self) -> bool:
        return self.mode == "assign" or (self.mode == "given" and self.cross)


def env_config(args, v_max: float) -> EnvConfig:
    """EnvConfig from the shared command-line flags of train.py, train_bc.py and train_ppo.py."""
    return EnvConfig(mode=args.mode, cost=args.cost, chord_horizontal=args.chord_horizontal,
                     span_scale=args.span_scale, v_max=v_max, w_time=args.w_time, w_cross=args.w_cross,
                     cross=args.cross, lam=args.lam, lookahead=args.lookahead, hand_pos=not args.no_hand_pos,
                     use_mask=not args.no_mask, chord_tol=args.chord_tol, overlap_tol=args.overlap_tol)


# --------------------------------------------------------------------------------------
# Static per-segment data
# --------------------------------------------------------------------------------------
class SegInfo:
    def __init__(self, seg: Segment, cfg: EnvConfig, model: CostModel, bonus: bool,
                 tables: Optional[SegmentTables] = None):
        self.seg = seg
        ev = seg.events
        if cfg.mode == "lh":
            self.order = [i for i, e in enumerate(ev) if e.hand == "L"]
        else:
            self.order = list(range(len(ev)))
        self.bonus = bonus and not seg.synthetic and seg.n_annotators > 0
        self.lo = [e.pitches[0] for e in ev]
        self.hi = [e.pitches[-1] for e in ev]
        self.black = [float(e.pitches[0] % 12 in (1, 3, 6, 8, 10)) for e in ev]
        self.onset = [e.onset for e in ev]
        self.sound_end = [max(e.orig_offset) for e in ev]
        # under a held note of either hand at its onset (static: holds come from the score)
        holds = [(e.onset, u) for e in ev for u in e.hold_until if u > -np.inf]
        self.held_any = [float(any(o0 < e.onset < u - cfg.overlap_tol for o0, u in holds)) for e in ev]
        # imitation targets: (hand, global set) pairs that reproduce some annotator's fingers
        self.matches: List[set] = []
        for e in ev:
            m = set()
            for row in e.labels:
                hands = {h for h, _ in row}
                if len(hands) != 1 or None in hands or any(f == 0 for _, f in row):
                    continue
                h = hands.pop()
                fs = tuple(f for _, f in row)
                for s in SETS_OF_SIZE[e.size]:
                    if fingers_of(int(s), h) == fs:
                        m.add((h, int(s)))
            self.matches.append(m)
        self.tables: Optional[SegmentTables] = tables
        if cfg.mode != "assign" and tables is None:
            hands = ("L",) if cfg.mode == "lh" else ("L", "R")
            self.tables = build_segment_tables(seg, model, cfg.v_max, cfg.w_time, hands)
        self._hev: Dict[Tuple[int, str], List[HandEvent]] = {}
        self._reach: Dict[Tuple[int, str], np.ndarray] = {}
        self.model = model

    def hand_events(self, i: int, hand: str) -> List[HandEvent]:
        k = (i, hand)
        if k not in self._hev:
            e = self.seg.events[i]
            self._hev[k] = [self.model.hand_event(hand, e.pitches, fingers_of(int(s), hand)) for s in SETS_OF_SIZE[e.size]]
            self._reach[k] = np.array([self.model.chord_reach_ok(h) for h in self._hev[k]])
        return self._hev[k]

    def reach(self, i: int, hand: str) -> np.ndarray:
        self.hand_events(i, hand)
        return self._reach[(i, hand)]


@dataclass
class HandState:
    played: int = 0  # events played (index into this hand's cost tables in 2a/2b)
    sets: List[int] = field(default_factory=list)  # global set ids, last two
    local: List[int] = field(default_factory=list)  # local set indices, last two
    hev: List[HandEvent] = field(default_factory=list)  # last two HandEvents
    keys: List[int] = field(default_factory=list)  # last two keys (thumb-side notes)
    last_onset: float = -np.inf
    thumb: Optional[float] = None
    held: List[Tuple[int, float]] = field(default_factory=list)  # (finger, until)
    sounding_until: float = -np.inf  # 2c hand mask: latest original offset of its notes

    def push(self, s, local, hev, key):
        self.sets = (self.sets + [s])[-2:]
        self.local = (self.local + [local])[-2:]
        self.hev = (self.hev + [hev])[-2:]
        self.keys = (self.keys + [key])[-2:]


@dataclass
class StepRecord:
    event: int
    hand: str
    set: int
    rules: float
    extra: float  # time-leap + shift + crossing
    infeasible: bool
    shift: bool
    forced: bool  # every action masked: a fallback was used (occupied override / unreachable)
    bonus: float
    penalty: float


# --------------------------------------------------------------------------------------
# One environment
# --------------------------------------------------------------------------------------
class Episode:
    def __init__(self, cfg: EnvConfig, model: Optional[CostModel] = None):
        self.cfg = cfg
        self.model = model or make_model(cfg.cost, cfg.chord_horizontal, cfg.span_scale)
        self.n_actions = cfg.n_actions
        self.info: Optional[SegInfo] = None
        self._rule_cache: Dict[tuple, Tuple[float, float]] = {}

    # ------------------------------------------------------------------ episode control
    def reset(self, info: SegInfo):
        self.info = info
        self.t = 0
        self.hands = {"L": HandState(), "R": HandState()}
        self.records: List[StepRecord] = []
        self.ret = 0.0
        self._mask_cache = None
        return self.obs(), self.mask()

    @property
    def done(self) -> bool:
        return self.t >= len(self.info.order)

    @property
    def event_index(self) -> int:
        return self.info.order[self.t]

    @property
    def event(self) -> Event:
        return self.info.seg.events[self.event_index]

    # ------------------------------------------------------------------ masks
    def _busy_bits(self, hand: str, onset: float) -> np.ndarray:
        hs = self.hands[hand]
        hs.held = [(f, u) for f, u in hs.held if onset < u - self.cfg.overlap_tol]
        b = np.zeros(5, bool)
        for f, _ in hs.held:
            b[f - 1] = True
        return b

    def _hand_mask(self, hand: str) -> Tuple[np.ndarray, dict]:
        """Allowed global sets for the current event played by `hand`, plus flags."""
        i = self.event_index
        e = self.info.seg.events[i]
        hs = self.hands[hand]
        sets = SETS_OF_SIZE[e.size]
        flags = dict(shift=False, forced=False, infeasible=np.zeros(len(sets), bool))
        if self.info.tables is not None:
            T = self.info.tables.hands[hand].tables[hs.played]
            s1 = hs.local[-1] if hs.local else 0
            allowed = T.allowed[s1].copy()
            flags["shift"] = bool(T.shift[s1])
            flags["infeasible"] = T.infeasible[s1]
            if not T.reach.any():
                flags["forced"] = True
        else:
            H0 = self.info.hand_events(i, hand)
            reach = self.info.reach(i, hand)
            ev1 = hs.hev[-1] if hs.hev else None
            trans = np.array([self.model.transition_ok(ev1, h) for h in H0])
            base = reach & trans
            allowed = base.copy()
            if not base.any():
                flags["shift"] = True
                allowed = reach.copy()
            if not allowed.any():
                flags["forced"] = True
                allowed[:] = True
            flags["infeasible"] = ~base & allowed
        if not self.cfg.use_mask:
            allowed = np.ones(len(sets), bool)
        elif self.cfg.occupied:
            busy = self._busy_bits(hand, e.onset)
            if busy.any():
                ok = allowed & ~(SET_BITS[sets][:, busy].any(1))
                if ok.any():
                    allowed = ok
                else:
                    flags["forced"] = True
        m = np.zeros(N_SETS, bool)
        m[sets[allowed]] = True
        return m, flags

    def _hands_allowed(self) -> List[str]:
        """Hands the current event may take (2c, §C.5.3 item 4)."""
        e = self.event
        if self.cfg.mode != "assign" or e.size > 1:
            return [e.hand]
        if not self.cfg.use_mask:
            return ["L", "R"]
        ok = []
        for h in ("L", "R"):
            if self.hands[h].sounding_until > e.onset + self.cfg.overlap_tol:
                continue
            ok.append(h)
        if self.t > 0:
            prev = self.info.order[self.t - 1]
            if e.onset - self.info.onset[prev] <= self.cfg.chord_tol and self.records:
                other = "R" if self.records[-1].hand == "L" else "L"
                ok = [h for h in ok if h == other] or [other]
        return ok or ["L", "R"]

    def mask(self) -> np.ndarray:
        if self.done:
            return np.ones(self.n_actions, bool)
        if self._mask_cache is not None:
            return self._mask_cache[0]
        flags = {}
        if self.cfg.mode == "assign":
            m = np.zeros(2 * N_SETS, bool)
            for h in self._hands_allowed():
                hm, flags[h] = self._hand_mask(h)
                m[HAND_IDX[h] * N_SETS:(HAND_IDX[h] + 1) * N_SETS] = hm
        else:
            h = self.event.hand
            m, flags[h] = self._hand_mask(h)
        self._mask_cache = (m, flags)
        return m

    def true_mask(self) -> np.ndarray:
        """The feasibility mask even when the agent is unmasked (--no-mask)."""
        if self.cfg.use_mask:
            return self.mask()
        use = self.cfg.use_mask
        self.cfg.use_mask = True
        cache = self._mask_cache
        self._mask_cache = None
        try:
            return self.mask().copy()
        finally:
            self.cfg.use_mask = use
            self._mask_cache = cache

    # ------------------------------------------------------------------ costs (2c)
    def _rules(self, ev2, ev1, ev0) -> float:
        k = (ev2, ev1, ev0)
        v = self._rule_cache.get(k)
        if v is None:
            v = float(self.model.event_rules(ev2, ev1, ev0).sum())
            if ev1 is not None:
                v += float(self.model.next_terms(ev1, ev0).sum())
            if len(self._rule_cache) > 500_000:
                self._rule_cache.clear()
            self._rule_cache[k] = v
        return v

    # ------------------------------------------------------------------ step
    def step(self, action: int) -> Tuple[float, bool]:
        assert not self.done, "step() after the episode ended"
        a = int(action)
        legal_mask = self.mask()
        flags_all = self._mask_cache[1]
        penalty = 0.0
        if not self.cfg.use_mask:
            if not self.true_mask()[a]:
                penalty = self.cfg.impossible_penalty
        elif not legal_mask[a]:
            raise ValueError(f"masked action {a} at event {self.event_index}")
        if self.cfg.mode == "assign":
            hand = "L" if a < N_SETS else "R"
            s = a % N_SETS
        else:
            hand, s = self.event.hand, a
        i = self.event_index
        e = self.info.seg.events[i]
        sets = SETS_OF_SIZE[e.size]
        loc = np.flatnonzero(sets == s)
        if len(loc) == 0:
            raise ValueError(f"set {s} has the wrong size for event {i} (size {e.size})")
        loc = int(loc[0])
        hs = self.hands[hand]
        flags = flags_all.get(hand) or dict(shift=False, forced=False, infeasible=np.zeros(len(sets), bool))
        fingers = fingers_of(s, hand)
        h0 = self.info.hand_events(i, hand)[loc] if self.info.tables is None else None

        if self.info.tables is not None:
            T = self.info.tables.hands[hand].tables[hs.played]
            s1 = hs.local[-1] if hs.local else 0
            s2 = hs.local[-2] if (len(hs.local) == 2 and T.C.shape[0] > 1) else 0
            rules = float(T.D[s2, s1, loc])
            extra = float(T.C[s2, s1, loc]) - rules
        else:
            ev1 = hs.hev[-1] if hs.hev else None
            ev2 = hs.hev[-2] if len(hs.hev) == 2 else None
            rules = self._rules(ev2, ev1, h0)
            extra = 0.0
            if hs.thumb is not None:
                extra += time_leap(thumb_position(hand, e.pitches, fingers) - hs.thumb, e.onset - hs.last_onset,
                                   self.cfg.v_max, self.cfg.w_time)
            if flags["shift"]:
                extra += SHIFT_COST

        # update the hand
        hs.push(s, loc, h0, key_of(e.pitches, hand))
        hs.played += 1
        hs.last_onset = e.onset
        hs.thumb = thumb_position(hand, e.pitches, fingers)
        hs.sounding_until = max(hs.sounding_until, e.offset)
        for f, u in zip(fingers, e.hold_until):
            if u > -np.inf:
                hs.held.append((f, u))
        if self.cfg.crossing_on:
            extra += crossing(self.hands["L"].thumb, self.hands["R"].thumb, self.cfg.cross_margin, self.cfg.w_cross)

        bonus = 0.0
        if self.cfg.lam and self.info.bonus:
            bonus = float((hand, s) in self.info.matches[i])
        reward = -(self.cfg.alpha * rules + extra) + self.cfg.lam * bonus - penalty
        self.records.append(StepRecord(i, hand, s, rules, extra, bool(flags["infeasible"][loc]) or penalty > 0,
                                       bool(flags["shift"]), bool(flags["forced"]), bonus, penalty))
        self.ret += reward
        self.t += 1
        self._mask_cache = None
        return reward, self.done

    def action_cost(self, action: int) -> float:
        """Immediate cost (rules + time-leap + shift) of an action, without taking it (greedy baseline)."""
        a = int(action)
        hand, s = (("L" if a < N_SETS else "R"), a % N_SETS) if self.cfg.mode == "assign" else (self.event.hand, a)
        i = self.event_index
        e = self.info.seg.events[i]
        loc = int(np.flatnonzero(SETS_OF_SIZE[e.size] == s)[0])
        hs = self.hands[hand]
        if self.info.tables is not None:
            T = self.info.tables.hands[hand].tables[hs.played]
            s1 = hs.local[-1] if hs.local else 0
            s2 = hs.local[-2] if (len(hs.local) == 2 and T.C.shape[0] > 1) else 0
            return float(T.C[s2, s1, loc])
        h0 = self.info.hand_events(i, hand)[loc]
        ev1 = hs.hev[-1] if hs.hev else None
        ev2 = hs.hev[-2] if len(hs.hev) == 2 else None
        c = self._rules(ev2, ev1, h0)
        if hs.thumb is not None:
            c += time_leap(thumb_position(hand, e.pitches, fingers_of(s, hand)) - hs.thumb, e.onset - hs.last_onset,
                           self.cfg.v_max, self.cfg.w_time)
        self.mask()
        if self._mask_cache[1].get(hand, {}).get("shift"):
            c += SHIFT_COST
        return c

    # ------------------------------------------------------------------ observation
    def obs(self) -> np.ndarray:
        cfg = self.cfg
        o = np.zeros(cfg.obs_dim, np.float32)
        if self.done:
            return o
        info = self.info
        i = self.event_index
        e = info.seg.events[i]
        mode = cfg.mode
        hand_c = e.hand if mode != "assign" else None
        # current event
        k = key_of(e.pitches, hand_c) if hand_c else e.pitches[0]
        o[0] = (k - 60) / 24.0
        o[1] = float(k % 12 in (1, 3, 6, 8, 10))
        o[2] = e.size / 5.0
        o[3] = (e.pitches[-1] - e.pitches[0]) / 12.0
        o[4] = tclip(e.onset - info.onset[info.order[self.t - 1]]) if self.t > 0 else 1.0
        o[5] = info.held_any[i]
        j = 6
        HL, HR = self.hands["L"], self.hands["R"]
        for d in range(1, cfg.lookahead + 1):
            if self.t + d < len(info.order):
                n = info.order[self.t + d]
                en = info.seg.events[n]
                if HL.keys:
                    o[j] = iv(HL.keys[-1], key_of(en.pitches, "L"), "L")
                if HR.keys:
                    o[j + 1] = iv(HR.keys[-1], key_of(en.pitches, "R"), "R")
                o[j + 2] = info.black[n]
                o[j + 3] = en.size / 5.0
                o[j + 4] = tclip(en.onset - e.onset)
                o[j + 5] = 1.0
                o[j + 6] = 0.0 if mode == "assign" else (1.0 if en.hand == "R" else -1.0)
            j += 7
        for h in ("L", "R"):
            hs = self.hands[h]
            cur = key_of(e.pitches, h)
            if hs.played:
                o[j] = 1.0
                o[j + 1] = iv(hs.keys[-1], cur, h)
                o[j + 2:j + 7] = SET_BITS[hs.sets[-1]]
                if len(hs.sets) == 2:
                    o[j + 7:j + 12] = SET_BITS[hs.sets[-2]]
                    o[j + 12] = iv(hs.keys[-2], cur, h)
                o[j + 13] = tclip(e.onset - hs.last_onset)
                o[j + 14] = float(any(e.onset < u - cfg.overlap_tol for _, u in hs.held))
                if cfg.hand_pos:
                    o[j + 15] = iv(hs.thumb, cur, h)
            j += 16 if cfg.hand_pos else 15
        o[j] = 0.0 if mode == "assign" else (1.0 if e.hand == "R" else -1.0)
        return o

    # ------------------------------------------------------------------ results
    def fingering(self) -> List[Tuple[str, Tuple[int, ...]]]:
        """Per visited event: (hand, fingers aligned with the event's ascending pitches)."""
        return [(r.hand, fingers_of(r.set, r.hand)) for r in self.records]


# --------------------------------------------------------------------------------------
# Many environments
# --------------------------------------------------------------------------------------
TABLE_CACHE = Path("cache/twohand")


def _tables_key(segments: Sequence[Segment], cfg: EnvConfig) -> str:
    """Content hash of everything the cost tables depend on."""
    sig = (cfg.mode, cfg.cost, cfg.chord_horizontal, cfg.span_scale, round(cfg.v_max, 6), cfg.w_time,
           [(s.piece, s.index, tuple((e.hand, e.onset, e.pitches) for e in s.events)) for s in segments])
    return hashlib.sha1(pickle.dumps(sig)).hexdigest()[:20]


class SegmentPool:
    """SegInfo objects built once and shared by the environments of one process. The cost tables
    of 2a/2b are cached on disk (cache/twohand), keyed by a hash of the segments and reward settings."""

    def __init__(self, segments: Sequence[Segment], cfg: EnvConfig, bonus: bool, model: Optional[CostModel] = None,
                 cache: bool = True):
        self.cfg = cfg
        self.model = model or make_model(cfg.cost, cfg.chord_horizontal, cfg.span_scale)
        tables = None
        path = None
        if cache and cfg.mode != "assign" and segments:
            path = TABLE_CACHE / f"tables_{_tables_key(segments, cfg)}.pkl"
            if path.exists():
                try:
                    with open(path, "rb") as f:
                        tables = pickle.load(f)
                except Exception:  # noqa: BLE001 - a broken cache file is rebuilt
                    tables = None
        self.infos = [SegInfo(s, cfg, self.model, bonus, tables[k] if tables else None) for k, s in enumerate(segments)]
        if path is not None and tables is None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            with open(tmp, "wb") as f:
                pickle.dump([x.tables for x in self.infos], f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)
        self.infos = [x for x in self.infos if x.order]

    def __len__(self):
        return len(self.infos)


class VecEnv:
    """n Episodes stepped together; finished episodes restart on a random segment."""

    def __init__(self, pool: SegmentPool, n_envs: int = 16, seed: int = 0):
        self.pool = pool
        self.cfg = pool.cfg
        self.n = n_envs
        self.rng = np.random.default_rng(seed)
        self.eps = [Episode(self.cfg, pool.model) for _ in range(n_envs)]
        self.obs = np.zeros((n_envs, self.cfg.obs_dim), np.float32)
        self.masks = np.zeros((n_envs, self.cfg.n_actions), bool)
        self.seg_id = np.zeros(n_envs, int)
        self.finished: List[dict] = []  # summaries of finished episodes since the last drain
        for k in range(n_envs):
            self._reset(k)

    def _reset(self, k: int):
        sid = int(self.rng.integers(len(self.pool)))
        self.seg_id[k] = sid
        o, m = self.eps[k].reset(self.pool.infos[sid])
        self.obs[k], self.masks[k] = o, m

    def step(self, actions: np.ndarray):
        rew = np.zeros(self.n, np.float32)
        done = np.zeros(self.n, bool)
        next_obs = np.empty_like(self.obs)
        next_masks = np.empty_like(self.masks)
        for k, ep in enumerate(self.eps):
            r, d = ep.step(int(actions[k]))
            rew[k], done[k] = r, d
            if d:
                next_obs[k] = 0.0
                next_masks[k] = True
                self.finished.append(dict(seg=int(self.seg_id[k]), ret=ep.ret, len=len(ep.records),
                                          records=ep.records))
                self._reset(k)
            else:
                next_obs[k] = ep.obs()
                next_masks[k] = ep.mask()
        # self.obs/masks now hold the first state of new episodes where done
        for k, ep in enumerate(self.eps):
            if not done[k]:
                self.obs[k], self.masks[k] = next_obs[k], next_masks[k]
        return next_obs, next_masks, rew, done

    def drain(self) -> List[dict]:
        out, self.finished = self.finished, []
        return out


def run_segment(ep: Episode, info: SegInfo, policy) -> Episode:
    """Finger one segment with policy(obs[1, D], mask[1, A]) -> actions[1]."""
    o, m = ep.reset(info)
    while not ep.done:
        a = int(policy(o[None], m[None])[0])
        ep.step(a)
        if not ep.done:
            o, m = ep.obs(), ep.mask()
    return ep


# --------------------------------------------------------------------------------------
# Gymnasium wrapper (sb3-contrib MaskablePPO)
# --------------------------------------------------------------------------------------
class TwoHandEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, pool: SegmentPool, seed: int = 0):
        super().__init__()
        self.pool = pool
        self.ep = Episode(pool.cfg, pool.model)
        self.action_space = spaces.Discrete(pool.cfg.n_actions)
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(pool.cfg.obs_dim,), dtype=np.float32)
        self._rng = np.random.default_rng(seed)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        sid = int(self._rng.integers(len(self.pool))) if not options or "seg" not in options else options["seg"]
        o, _ = self.ep.reset(self.pool.infos[sid])
        return o, {}

    def action_masks(self) -> np.ndarray:
        return self.ep.mask()

    def step(self, action):
        r, d = self.ep.step(int(action))
        o = self.ep.obs()
        return o, r, d, False, {}
