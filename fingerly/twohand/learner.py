"""Masked Double DQN for the two-hand environments (README §C.9.2–C.9.4, §C.10).

* 16 vectorised environments, one batched forward pass per step.
* Dueling MLP (fingerly.dqn.QNetwork), n-step Double DQN targets with the max taken over the actions
  allowed at the next event, Huber loss, soft target updates, linear epsilon.
* R3 elite replay: the agent's best episode per training segment; up to `elite` of each batch.
* A2 demonstrations: pianists' episodes with a DQfD large-margin loss, share decayed linearly.
* A1 / 2c: start from S1's weights (`bc_init`) or from a 2b trunk (`init`).
* Model selection on validation every `eval_every` steps, early stopping with `patience`.
* Checkpoints (network, optimiser, counters, RNG states, elite memory, replay buffer) every
  `checkpoint_every` steps; an interrupted run resumes from `ckpt.pt` in its output folder.
"""

from __future__ import annotations

import copy
import csv
import json
import random
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..dqn import QNetwork
from .actions import N_SETS, SETS_OF_SIZE, fingers_of
from .env import Episode, SegmentPool, VecEnv

NEG = -1e9


@dataclass
class TrainConfig:
    algo: str = "ddqn"  # "ddqn" (double + dueling, n-step) | "dqn" (plain, 1-step)
    steps: int = 1_000_000
    n_envs: int = 16
    lr: float = 3e-4
    gamma: float = 0.99
    n_step: int = 3
    buffer: int = 500_000
    batch: int = 256
    train_every: int = 4  # environment steps per gradient update
    learning_starts: int = 10_000
    tau: float = 0.005
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_frac: float = 0.2
    hidden: Tuple[int, ...] = (256, 256)
    double: bool = True
    dueling: bool = True
    grad_clip: float = 10.0
    elite: float = 0.0  # share of each batch from elite transitions (R3: 0.25)
    demos: float = 0.0  # A2: initial share of demonstrations (0.25), decayed to demos_end
    demos_end: float = 0.10
    demo_margin: float = 0.8
    demo_weight: float = 1.0
    eval_every: int = 25_000
    patience: int = 50_000
    checkpoint_every: int = 50_000
    save_buffer: bool = True
    cuda_graph: bool = True  # capture the update in a CUDA graph when training on the GPU
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self):
        self.hidden = tuple(self.hidden)
        if self.algo == "dqn":
            self.double, self.dueling, self.n_step = False, False, 1


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------------------
# Buffers
# --------------------------------------------------------------------------------------
class Replay:
    def __init__(self, cap: int, obs_dim: int, n_actions: int, with_cur_mask: bool = False):
        self.cap = cap
        self.o = np.zeros((cap, obs_dim), np.float32)
        self.o2 = np.zeros((cap, obs_dim), np.float32)
        self.a = np.zeros(cap, np.int64)
        self.r = np.zeros(cap, np.float32)
        self.d = np.zeros(cap, np.float32)
        self.m2 = np.zeros((cap, n_actions), bool)
        self.m = np.zeros((cap, n_actions), bool) if with_cur_mask else None
        self.i = 0
        self.n = 0

    def add(self, o, a, r, o2, d, m2, m=None):
        j = self.i
        self.o[j], self.a[j], self.r[j], self.o2[j], self.d[j], self.m2[j] = o, a, r, o2, d, m2
        if self.m is not None:
            self.m[j] = m
        self.i = (self.i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def __len__(self):
        return self.n

    def idx(self, rng, k):
        return rng.integers(0, self.n, size=k)

    def take(self, idx):
        return self.o[idx], self.a[idx], self.r[idx], self.o2[idx], self.d[idx], self.m2[idx]

    def state(self) -> dict:
        n = self.n
        return dict(o=self.o[:n], o2=self.o2[:n], a=self.a[:n], r=self.r[:n], d=self.d[:n], m2=self.m2[:n], i=self.i)

    def load(self, st: dict):
        n = len(st["a"])
        self.o[:n], self.o2[:n], self.a[:n], self.r[:n], self.d[:n], self.m2[:n] = (
            st["o"], st["o2"], st["a"], st["r"], st["d"], st["m2"])
        self.n, self.i = n, int(st["i"])


class NStep:
    """Per-environment n-step accumulator; returns finished transitions."""

    def __init__(self, n: int, gamma: float):
        self.n, self.g = n, gamma
        self.q: deque = deque()

    def push(self, o, a, r, o2, m2, terminal: bool, m=None) -> List[tuple]:
        self.q.append((o, a, r, m))
        out = []
        if terminal:
            while self.q:
                out.append(self._emit(o2, m2, True))
        elif len(self.q) == self.n:
            out.append(self._emit(o2, m2, False))
        return out

    def _emit(self, o2, m2, terminal):
        o, a, _, m = self.q[0]
        R, g = 0.0, 1.0
        for (_, _, r, _) in self.q:
            R += g * r
            g *= self.g
        self.q.popleft()
        return (o, a, R, o2, 0.0 if terminal else g, m2, m)

    def flush_truncated(self, o2, m2) -> List[tuple]:
        """End of a demonstration that stops early: bootstrap from the last state."""
        out = []
        while self.q:
            g = self.g ** len(self.q)
            o, a, _, m = self.q[0]
            R, gg = 0.0, 1.0
            for (_, _, r, _) in self.q:
                R += gg * r
                gg *= self.g
            self.q.popleft()
            out.append((o, a, R, o2, g, m2, m))
        return out


# --------------------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------------------
class Agent:
    def __init__(self, obs_dim: int, n_actions: int, cfg: TrainConfig):
        self.cfg = cfg
        self.obs_dim, self.n_actions = obs_dim, n_actions
        self.dev = torch.device(cfg.device)
        self.q = QNetwork(obs_dim, n_actions, cfg.hidden, cfg.dueling).to(self.dev)
        self.qt = copy.deepcopy(self.q)
        for p in self.qt.parameters():
            p.requires_grad_(False)
        self._graph_mode = self.dev.type == "cuda" and cfg.cuda_graph
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr, foreach=True, capturable=self._graph_mode)
        self._params = list(self.q.parameters())
        self._tparams = list(self.qt.parameters())
        self._graph = None
        self._loss_t = None

    @torch.no_grad()
    def q_values(self, obs: np.ndarray) -> np.ndarray:
        return self.q(torch.as_tensor(obs, device=self.dev)).cpu().numpy()

    def act(self, obs: np.ndarray, masks: np.ndarray, eps: float, rng: np.random.Generator) -> np.ndarray:
        q = self.q_values(obs)
        q[~masks] = NEG
        a = q.argmax(1)
        if eps > 0:
            for k in np.flatnonzero(rng.random(len(a)) < eps):
                a[k] = rng.choice(np.flatnonzero(masks[k]))
        return a

    def greedy(self) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        def pol(obs, masks):
            q = self.q_values(obs)
            q[~masks] = NEG
            return q.argmax(1)
        return pol

    def _step(self, o, a, r, o2, d, m2, dflag=None, mc=None) -> torch.Tensor:
        """One gradient step; graph-safe (no data-dependent Python branches when dflag is given)."""
        q_all = self.q(o)
        q = q_all.gather(1, a[:, None]).squeeze(1)
        with torch.no_grad():
            qt2 = self.qt(o2).masked_fill(~m2, NEG)
            if self.cfg.double:
                a2 = self.q(o2).masked_fill(~m2, NEG).argmax(1, keepdim=True)
                qn = qt2.gather(1, a2).squeeze(1)
            else:
                qn = qt2.max(1).values
            qn = torch.where(d > 0, qn, torch.zeros_like(qn))
            target = r + d * qn
        loss = F.smooth_l1_loss(q, target)
        if dflag is not None:  # DQfD large-margin loss on demonstration samples
            margin = torch.full_like(q_all, self.cfg.demo_margin)
            margin.scatter_(1, a[:, None], 0.0)
            best = (q_all + margin).masked_fill(~mc, NEG).max(1).values
            j_e = ((best - q) * dflag).sum() / dflag.sum().clamp(min=1.0)
            loss = loss + self.cfg.demo_weight * j_e
        loss.backward()
        nn.utils.clip_grad_norm_(self._params, self.cfg.grad_clip, foreach=True)
        self.opt.step()
        self.opt.zero_grad(set_to_none=self._graph_mode is False)
        with torch.no_grad():
            torch._foreach_lerp_(self._tparams, self._params, self.cfg.tau)
        return loss.detach()

    def _capture(self, B: int):
        dev, A, D = self.dev, self.n_actions, self.obs_dim
        self._st = dict(o=torch.zeros(B, D, device=dev), a=torch.zeros(B, dtype=torch.long, device=dev),
                        r=torch.zeros(B, device=dev), o2=torch.zeros(B, D, device=dev), d=torch.zeros(B, device=dev),
                        m2=torch.ones(B, A, dtype=torch.bool, device=dev), dflag=torch.zeros(B, device=dev),
                        mc=torch.ones(B, A, dtype=torch.bool, device=dev))
        s = self._st
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):  # warm-up outside the graph (allocates optimiser state)
            for _ in range(3):
                self._step(s["o"], s["a"], s["r"], s["o2"], s["d"], s["m2"], s["dflag"], s["mc"])
        torch.cuda.current_stream().wait_stream(side)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._loss_t = self._step(s["o"], s["a"], s["r"], s["o2"], s["d"], s["m2"], s["dflag"], s["mc"])

    def update(self, o, a, r, o2, d, m2, demo: Optional[np.ndarray] = None, m_cur: Optional[np.ndarray] = None):
        """One gradient update. On CUDA the step is a captured CUDA graph (README §C.10: speed);
        returns None there (read the loss with last_loss()), else the loss as a float."""
        if self._graph_mode:
            if self._graph is None:
                self._capture(len(a))
            s = self._st
            for k, x in (("o", o), ("a", a), ("r", r), ("o2", o2), ("d", d), ("m2", m2)):
                s[k].copy_(torch.from_numpy(np.ascontiguousarray(x)), non_blocking=True)
            if demo is not None:
                s["dflag"].copy_(torch.from_numpy(demo.astype(np.float32)), non_blocking=True)
                s["mc"].copy_(torch.from_numpy(m_cur), non_blocking=True)
            else:
                s["dflag"].zero_()
            self._graph.replay()
            return None
        t = lambda x: torch.as_tensor(x, device=self.dev)  # noqa: E731
        dflag = mc = None
        if demo is not None and demo.any():
            dflag, mc = t(demo.astype(np.float32)), t(m_cur)
        self._loss_t = self._step(t(o), t(a), t(r), t(o2), t(d), t(m2), dflag, mc)
        return float(self._loss_t.item())

    def last_loss(self) -> Optional[float]:
        return None if self._loss_t is None else float(self._loss_t.item())

    def save(self, path, extra: Optional[dict] = None):
        torch.save(dict(q=self.q.state_dict(), cfg=asdict(self.cfg), obs_dim=self.obs_dim, n_actions=self.n_actions,
                        extra=extra or {}), path)

    @classmethod
    def load(cls, path, device: str = "cpu") -> "Agent":
        ck = torch.load(path, map_location=device, weights_only=False)
        cfg = TrainConfig(**{**ck["cfg"], "device": device})
        ag = cls(ck["obs_dim"], ck["n_actions"], cfg)
        ag.q.load_state_dict(ck["q"])
        ag.qt.load_state_dict(ck["q"])
        ag.extra = ck.get("extra", {})
        return ag

    def load_weights(self, path, trunk_only: bool = False):
        """A1: all weights of a same-shaped network (S1). 2c: only layers whose shapes match (trunk)."""
        sd = torch.load(path, map_location=self.dev, weights_only=False)
        sd = sd.get("q", sd)
        own = self.q.state_dict()
        use = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape
               and (not trunk_only or k.startswith("body."))}
        own.update(use)
        self.q.load_state_dict(own)
        self.qt.load_state_dict(own)
        return sorted(use)


# --------------------------------------------------------------------------------------
# Demonstrations (A2) and elite seeding (A3)
# --------------------------------------------------------------------------------------
def annotator_action(info, i: int, hand_mode_assign: bool, annotator: int = 0) -> Optional[int]:
    e = info.seg.events[i]
    if annotator >= len(e.labels):
        return None
    row = e.labels[annotator]
    hands = {h for h, _ in row}
    if len(hands) != 1 or None in hands:
        return None
    h = hands.pop()
    fs = tuple(f for _, f in row)
    for s in SETS_OF_SIZE[e.size]:
        if fingers_of(int(s), h) == fs:
            return int(s) + (N_SETS if (hand_mode_assign and h == "R") else 0)
    return None


def episode_transitions(pool: SegmentPool, info, actions_fn, n_step: int, gamma: float) -> Tuple[List[tuple], float]:
    """Play forced actions through the environment; stop (and bootstrap) at the first action that
    is unavailable or masked. actions_fn(ep) -> action or None."""
    ep = Episode(pool.cfg, pool.model)
    o, m = ep.reset(info)
    ns = NStep(n_step, gamma)
    out, ret = [], 0.0
    while not ep.done:
        a = actions_fn(ep)
        if a is None or not m[a]:
            out += ns.flush_truncated(o, m)
            break
        r, done = ep.step(a)
        ret += r
        o2, m2 = (ep.obs(), ep.mask()) if not done else (np.zeros_like(o), np.ones_like(m))
        out += ns.push(o, a, r, o2, m2, done, m)
        o, m = o2, m2
    return out, ret


def build_demos(pool: SegmentPool, n_step: int, gamma: float) -> Replay:
    assign = pool.cfg.mode == "assign"
    trans = []
    for info in pool.infos:
        if not info.bonus:
            continue
        tr, _ = episode_transitions(pool, info, lambda ep: annotator_action(ep.info, ep.event_index, assign),
                                    n_step, gamma)
        trans += tr
    buf = Replay(max(1, len(trans)), pool.cfg.obs_dim, pool.cfg.n_actions, with_cur_mask=True)
    for (o, a, R, o2, d, m2, m) in trans:
        buf.add(o, a, R, o2, d, m2, m)
    return buf


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------
class Trainer:
    def __init__(self, pool: SegmentPool, cfg: TrainConfig, out_dir: str,
                 evaluate: Optional[Callable[["Agent"], Dict[str, float]]] = None,
                 select_key: str = "Mgen", select_sign: float = 1.0, meta: Optional[dict] = None,
                 bc_init: Optional[str] = None, init: Optional[str] = None,
                 elite_seed: Optional[Dict[int, Tuple[List[tuple], float]]] = None,
                 pool_schedule: Optional[List[Tuple[int, SegmentPool]]] = None):
        self.pool, self.cfg, self.out = pool, cfg, Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.evaluate, self.select_key, self.select_sign = evaluate, select_key, select_sign
        self.meta = meta or {}
        set_seeds(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        env_cfg = pool.cfg
        self.agent = Agent(env_cfg.obs_dim, env_cfg.n_actions, cfg)
        self.loaded = []
        if bc_init:
            self.loaded = self.agent.load_weights(bc_init)
        if init:
            self.loaded = self.agent.load_weights(init, trunk_only=True)
        self.buf = Replay(cfg.buffer, env_cfg.obs_dim, env_cfg.n_actions)
        self.demo = build_demos(pool, cfg.n_step, cfg.gamma) if cfg.demos > 0 else None
        self.elite: Dict[int, dict] = {}
        self.elite_fixed = set()
        if elite_seed:
            for sid, (tr, ret) in elite_seed.items():
                if tr:
                    self._store_elite(sid, tr, ret)
                    self.elite_fixed.add(sid)
        self.pool_schedule = pool_schedule  # curriculum: [(start_step, pool), ...]
        self.step = 0
        self.episodes = 0
        self.best = -np.inf
        self.best_step = 0
        self.history: List[dict] = []

    # ------------------------------------------------------------------ elite memory
    def _store_elite(self, sid, trans, ret):
        o, a, R, o2, d, m2, _ = map(np.array, zip(*trans))
        self.elite[sid] = dict(ret=ret, o=o.astype(np.float32), a=a.astype(np.int64), r=R.astype(np.float32),
                               o2=o2.astype(np.float32), d=d.astype(np.float32), m2=m2.astype(bool))

    def _elite_sample(self, k):
        sids = list(self.elite)
        pick = self.rng.integers(0, len(sids), size=k)
        rows = []
        for p in pick:
            e = self.elite[sids[p]]
            j = int(self.rng.integers(len(e["a"])))
            rows.append((e["o"][j], e["a"][j], e["r"][j], e["o2"][j], e["d"][j], e["m2"][j]))
        return [np.stack(c) for c in zip(*rows)]

    # ------------------------------------------------------------------ checkpointing
    def _ckpt_path(self):
        return self.out / "ckpt.pt"

    def save_ckpt(self):
        st = dict(q=self.agent.q.state_dict(), qt=self.agent.qt.state_dict(), opt=self.agent.opt.state_dict(),
                  step=self.step, episodes=self.episodes, best=self.best, best_step=self.best_step,
                  history=self.history, elite=self.elite, elite_fixed=list(self.elite_fixed),
                  rng=self.rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                  py_rng=random.getstate(), np_rng=np.random.get_state())
        tmp = self.out / "ckpt.tmp"
        torch.save(st, tmp)
        tmp.replace(self._ckpt_path())
        if self.cfg.save_buffer:
            np.savez(self.out / "buffer.tmp.npz", **self.buf.state())
            (self.out / "buffer.tmp.npz").replace(self.out / "buffer.npz")

    def try_resume(self) -> bool:
        p = self._ckpt_path()
        if not p.exists():
            return False
        st = torch.load(p, map_location=self.agent.dev, weights_only=False)
        self.agent.q.load_state_dict(st["q"])
        self.agent.qt.load_state_dict(st["qt"])
        self.agent.opt.load_state_dict(st["opt"])
        self.step, self.episodes, self.best, self.best_step = st["step"], st["episodes"], st["best"], st["best_step"]
        self.history, self.elite, self.elite_fixed = st["history"], st["elite"], set(st["elite_fixed"])
        self.rng.bit_generator.state = st["rng"]
        torch.set_rng_state(st["torch_rng"])
        random.setstate(st["py_rng"])
        np.random.set_state(st["np_rng"])
        b = self.out / "buffer.npz"
        if b.exists():
            with np.load(b) as z:
                self.buf.load({k: z[k] for k in z.files})
        print(f"resumed at step {self.step} (buffer {len(self.buf)})", flush=True)
        return True

    # ------------------------------------------------------------------ loop
    def eps(self) -> float:
        frac = min(1.0, self.step / max(1, self.cfg.eps_frac * self.cfg.steps))
        return self.cfg.eps_start + (self.cfg.eps_end - self.cfg.eps_start) * frac

    def _pool_now(self) -> SegmentPool:
        if not self.pool_schedule:
            return self.pool
        cur = self.pool_schedule[0][1]
        for start, p in self.pool_schedule:
            if self.step >= start:
                cur = p
        return cur

    def sample(self):
        cfg = self.cfg
        n_el = int(round(cfg.batch * cfg.elite)) if (cfg.elite > 0 and self.elite) else 0
        n_dm = 0
        if self.demo is not None and len(self.demo):
            frac = min(1.0, self.step / max(1, cfg.steps))
            n_dm = int(round(cfg.batch * (cfg.demos + (cfg.demos_end - cfg.demos) * frac)))
        n_b = cfg.batch - n_el - n_dm
        parts = [self.buf.take(self.buf.idx(self.rng, n_b))]
        if n_el:
            parts.append(tuple(self._elite_sample(n_el)))
        demo_flag = m_cur = None
        if n_dm:
            j = self.demo.idx(self.rng, n_dm)
            parts.append(self.demo.take(j))
            demo_flag = np.concatenate([np.zeros(n_b + n_el, bool), np.ones(n_dm, bool)])
            m_cur = np.concatenate([np.ones((n_b + n_el, self.demo.m.shape[1]), bool), self.demo.m[j]])
        cols = [np.concatenate([p[c] for p in parts]) for c in range(6)]
        return cols, demo_flag, m_cur

    def train(self) -> Agent:
        cfg = self.cfg
        resumed = self.try_resume()
        (self.out / "config.json").write_text(json.dumps(dict(train=asdict(cfg), env=self.pool.cfg.__dict__,
                                                              meta=self.meta, loaded_weights=self.loaded), indent=1, default=str))
        tlog = open(self.out / "train_log.csv", "a", newline="")
        tw = csv.writer(tlog)
        if not resumed or tlog.tell() == 0:
            tw.writerow(["step", "episodes", "eps", "train_return_per_event", "loss", "steps_per_sec"])
        elog = open(self.out / "eval_log.csv", "a", newline="")
        ew = None
        cur_pool = self._pool_now()
        vec = VecEnv(cur_pool, cfg.n_envs, seed=cfg.seed * 1000 + self.step)
        ns = [NStep(cfg.n_step, cfg.gamma) for _ in range(cfg.n_envs)]
        ep_tr: List[List[tuple]] = [[] for _ in range(cfg.n_envs)]
        ep_ret = np.zeros(cfg.n_envs)
        ep_len = np.zeros(cfg.n_envs, int)
        recent = deque(maxlen=200)
        losses = deque(maxlen=500)
        upd_acc = 0.0
        t0, s0 = time.time(), self.step
        next_log = (self.step // 5000 + 1) * 5000
        next_eval = (self.step // cfg.eval_every + 1) * cfg.eval_every
        next_ck = (self.step // cfg.checkpoint_every + 1) * cfg.checkpoint_every
        stop = False
        while self.step < cfg.steps and not stop:
            if self.pool_schedule and self._pool_now() is not cur_pool:
                cur_pool = self._pool_now()
                vec = VecEnv(cur_pool, cfg.n_envs, seed=cfg.seed * 1000 + self.step)
                ns = [NStep(cfg.n_step, cfg.gamma) for _ in range(cfg.n_envs)]
                ep_tr = [[] for _ in range(cfg.n_envs)]
                ep_ret[:] = 0
                ep_len[:] = 0
            obs, masks = vec.obs.copy(), vec.masks.copy()
            sids = vec.seg_id.copy()
            acts = self.agent.act(obs, masks, self.eps(), self.rng)
            o2, m2, rew, done = vec.step(acts)
            for k in range(cfg.n_envs):
                ep_ret[k] += rew[k]
                ep_len[k] += 1
                for tr in ns[k].push(obs[k], acts[k], rew[k], o2[k], m2[k], bool(done[k])):
                    self.buf.add(*tr[:6])
                    if cfg.elite > 0:
                        ep_tr[k].append(tr)
                if done[k]:
                    self.episodes += 1
                    recent.append(ep_ret[k] / max(1, ep_len[k]))
                    sid = int(sids[k])
                    if cfg.elite > 0 and cur_pool is self.pool and sid not in self.elite_fixed:
                        if sid not in self.elite or ep_ret[k] > self.elite[sid]["ret"]:
                            self._store_elite(sid, ep_tr[k], float(ep_ret[k]))
                    ep_tr[k] = []
                    ep_ret[k] = 0.0
                    ep_len[k] = 0
            vec.drain()
            self.step += cfg.n_envs
            if len(self.buf) >= cfg.learning_starts:
                upd_acc += cfg.n_envs / cfg.train_every
                while upd_acc >= 1:
                    cols, demo_flag, m_cur = self.sample()
                    lv = self.agent.update(*cols, demo=demo_flag, m_cur=m_cur)
                    if lv is not None:
                        losses.append(lv)
                    upd_acc -= 1
            if self.step >= next_log:
                next_log += 5000
                sps = (self.step - s0) / max(1e-9, time.time() - t0)
                tw.writerow([self.step, self.episodes, round(self.eps(), 4),
                             round(float(np.mean(recent)), 4) if recent else "",
                             round(float(np.mean(losses)), 5) if losses else
                             (round(self.agent.last_loss(), 5) if self.agent.last_loss() is not None else ""),
                             round(sps, 1)])
                tlog.flush()
            if self.evaluate is not None and (self.step >= next_eval or self.step >= cfg.steps):
                next_eval += cfg.eval_every
                m = {"step": self.step, **self.evaluate(self.agent)}
                self.history.append(m)
                if ew is None:
                    ew = csv.DictWriter(elog, fieldnames=list(m.keys()), extrasaction="ignore")
                    if elog.tell() == 0:
                        ew.writeheader()
                ew.writerow(m)
                elog.flush()
                score = self.select_sign * m.get(self.select_key, -np.inf)
                tag = ""
                if score > self.best:
                    self.best, self.best_step = score, self.step
                    self.agent.save(self.out / "best.pt", {**self.meta, "step": self.step, "eval": m})
                    tag = "  <- best"
                print(f"step {self.step:>8} eps {self.eps():.3f} " + " ".join(
                    f"{k} {v:.4f}" for k, v in m.items() if k != "step" and isinstance(v, float)) + tag, flush=True)
                if cfg.patience and self.step - self.best_step >= cfg.patience and self.step >= cfg.eps_frac * cfg.steps:
                    print(f"early stop at {self.step}: no improvement in {cfg.patience} steps", flush=True)
                    stop = True
            if self.step >= next_ck:
                next_ck += cfg.checkpoint_every
                self.save_ckpt()
        self.agent.save(self.out / "last.pt", {**self.meta, "step": self.step})
        save_buffer, self.cfg.save_buffer = self.cfg.save_buffer, False
        self.save_ckpt()
        self.cfg.save_buffer = save_buffer
        (self.out / "buffer.npz").unlink(missing_ok=True)  # only needed to resume an unfinished run
        (self.out / "done.json").write_text(json.dumps(dict(step=self.step, episodes=self.episodes,
                                                            best_step=self.best_step, best=self.best,
                                                            seconds=time.time() - t0), indent=1, default=float))
        tlog.close()
        elog.close()
        return self.agent


# --------------------------------------------------------------------------------------
# Speed benchmark (§C.10)
# --------------------------------------------------------------------------------------
def bench(pool: SegmentPool, cfg: TrainConfig, steps: int = 1000, warmup: int = 4000) -> Dict[str, float]:
    """Env steps per second: environments alone (random masked actions), acting with the network,
    and full training (acting + replay + updates every `train_every` env steps)."""
    set_seeds(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    vec = VecEnv(pool, cfg.n_envs, cfg.seed)
    agent = Agent(pool.cfg.obs_dim, pool.cfg.n_actions, cfg)
    buf = Replay(max(cfg.buffer, warmup * 2), pool.cfg.obs_dim, pool.cfg.n_actions)
    ns = [NStep(cfg.n_step, cfg.gamma) for _ in range(cfg.n_envs)]

    def run(n, mode):
        done_steps = 0
        acc = 0.0
        t = time.time()
        while done_steps < n:
            obs, masks = vec.obs.copy(), vec.masks.copy()
            if mode == "env":
                acts = np.array([rng.choice(np.flatnonzero(m)) for m in masks])
            else:
                acts = agent.act(obs, masks, 0.1, rng)
            o2, m2, rew, done = vec.step(acts)
            vec.drain()
            for k in range(cfg.n_envs):
                for tr in ns[k].push(obs[k], acts[k], rew[k], o2[k], m2[k], bool(done[k])):
                    buf.add(*tr[:6])
            done_steps += cfg.n_envs
            if mode == "train":
                acc += cfg.n_envs / cfg.train_every
                while acc >= 1:
                    idx = buf.idx(rng, cfg.batch)
                    agent.update(*buf.take(idx))
                    acc -= 1
        if agent.dev.type == "cuda":
            torch.cuda.synchronize()
        return done_steps / (time.time() - t)

    run(warmup, "env")
    run(cfg.n_envs * 8, "train")  # warm-up: allocations and (on CUDA) the graph capture
    return dict(env_only=run(steps, "env"), act=run(steps, "act"), train=run(steps, "train"))
