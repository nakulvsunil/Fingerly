"""DQN / Double DQN with invalid-action masking and n-step returns (PyTorch).

Stable-Baselines3's DQN has no action masking, so the agent is written out here. It follows
Mnih et al. (2015) plus the usual fixes: Double DQN targets (van Hasselt et al., 2016),
optional dueling head, Huber loss, hard target-network sync and n-step returns. Masking is
applied both when acting (invalid fingers are never chosen) and in the bootstrap target
(max over valid next actions only), as in Huang & Ontañón (2022).
"""

from __future__ import annotations

import csv
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NEG_INF = -1e9


@dataclass
class DQNConfig:
    total_steps: int = 200_000
    lr: float = 3e-4
    batch_size: int = 128
    gamma: float = 0.99
    n_step: int = 3
    buffer_size: int = 200_000
    learning_starts: int = 2_000
    train_freq: int = 1
    gradient_steps: int = 1
    target_update: int = 1_000  # hard sync every N env steps
    eps_start: float = 1.0
    eps_end: float = 0.02
    eps_decay_frac: float = 0.3  # fraction of total_steps over which epsilon decays linearly
    double: bool = True
    dueling: bool = True
    hidden: List[int] = field(default_factory=lambda: [256, 256])
    grad_clip: float = 10.0
    eval_every: int = 10_000
    log_every: int = 1_000
    seed: int = 0
    device: str = "cpu"


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int = 5, hidden=(256, 256), dueling: bool = True):
        super().__init__()
        layers, d = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.body = nn.Sequential(*layers)
        self.dueling = dueling
        if dueling:
            self.value = nn.Linear(d, 1)
            self.adv = nn.Linear(d, n_actions)
        else:
            self.head = nn.Linear(d, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.body(x)
        if self.dueling:
            a = self.adv(z)
            return self.value(z) + a - a.mean(dim=1, keepdim=True)
        return self.head(z)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, n_actions: int = 5, seed: int = 0):
        self.cap = capacity
        self.obs = np.zeros((capacity, obs_dim), np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), np.float32)
        self.act = np.zeros(capacity, np.int64)
        self.rew = np.zeros(capacity, np.float32)
        self.disc = np.zeros(capacity, np.float32)  # gamma^n, or 0 if the episode ended
        self.next_mask = np.ones((capacity, n_actions), bool)
        self.i = 0
        self.full = False
        self.rng = np.random.default_rng(seed)

    def add(self, o, a, r, o2, disc, mask2):
        j = self.i
        self.obs[j], self.act[j], self.rew[j] = o, a, r
        self.next_obs[j], self.disc[j], self.next_mask[j] = o2, disc, mask2
        self.i = (self.i + 1) % self.cap
        self.full = self.full or self.i == 0

    def __len__(self):
        return self.cap if self.full else self.i

    def sample(self, n: int, device: str):
        idx = self.rng.integers(0, len(self), size=n)
        t = lambda x: torch.as_tensor(x[idx], device=device)  # noqa: E731
        return t(self.obs), t(self.act), t(self.rew), t(self.next_obs), t(self.disc), t(self.next_mask)


class NStepCollector:
    """Turns 1-step transitions into n-step ones before they go into the buffer."""

    def __init__(self, n: int, gamma: float, buffer: ReplayBuffer):
        self.n, self.gamma, self.buf = n, gamma, buffer
        self.q: deque = deque()

    def _emit(self, next_obs, next_mask, terminal: bool):
        o, a, _ = self.q[0]
        R, g = 0.0, 1.0
        for (_, _, r) in self.q:
            R += g * r
            g *= self.gamma
        self.buf.add(o, a, R, next_obs, 0.0 if terminal else g, next_mask)
        self.q.popleft()

    def push(self, o, a, r, o2, mask2, terminal: bool):
        self.q.append((o, a, r))
        if terminal:
            while self.q:
                self._emit(o2, mask2, True)
        elif len(self.q) == self.n:
            self._emit(o2, mask2, False)


class DQNAgent:
    def __init__(self, obs_dim: int, cfg: DQNConfig, n_actions: int = 5):
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        torch.manual_seed(cfg.seed)
        self.q = QNetwork(obs_dim, n_actions, cfg.hidden, cfg.dueling).to(cfg.device)
        self.q_target = QNetwork(obs_dim, n_actions, cfg.hidden, cfg.dueling).to(cfg.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.rng = np.random.default_rng(cfg.seed)

    @torch.no_grad()
    def act(self, obs: np.ndarray, mask: np.ndarray, eps: float = 0.0) -> int:
        valid = np.flatnonzero(mask)
        if eps > 0 and self.rng.random() < eps:
            return int(self.rng.choice(valid))
        q = self.q(torch.as_tensor(obs, device=self.cfg.device).unsqueeze(0))[0].cpu().numpy()
        q[~mask] = NEG_INF
        return int(q.argmax())

    def policy(self) -> Callable[[np.ndarray, np.ndarray], int]:
        return lambda obs, mask: self.act(obs, mask, 0.0)

    def update(self, batch) -> float:
        o, a, r, o2, disc, m2 = batch
        q = self.q(o).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next_t = self.q_target(o2).masked_fill(~m2, NEG_INF)
            if self.cfg.double:
                a2 = self.q(o2).masked_fill(~m2, NEG_INF).argmax(1, keepdim=True)
                q_next = q_next_t.gather(1, a2).squeeze(1)
            else:
                q_next = q_next_t.max(1).values
            q_next = torch.where(disc > 0, q_next, torch.zeros_like(q_next))
            target = r + disc * q_next
        loss = F.smooth_l1_loss(q, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip)
        self.opt.step()
        return float(loss.item())

    def sync_target(self):
        self.q_target.load_state_dict(self.q.state_dict())

    # -------------------------------------------------------------- checkpointing
    def save(self, path: str | Path, extra: Optional[dict] = None):
        torch.save(
            {"q": self.q.state_dict(), "cfg": asdict(self.cfg), "obs_dim": self.obs_dim, "extra": extra or {}},
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "DQNAgent":
        ck = torch.load(path, map_location=device, weights_only=False)
        cfg = DQNConfig(**{**ck["cfg"], "device": device})
        agent = cls(ck["obs_dim"], cfg)
        agent.q.load_state_dict(ck["q"])
        agent.sync_target()
        agent.extra = ck.get("extra", {})
        return agent


def train_dqn(
    env,
    cfg: DQNConfig,
    out_dir: str | Path,
    evaluate: Optional[Callable[[DQNAgent], Dict[str, float]]] = None,
    select_key: str = "val_reward",
    extra: Optional[dict] = None,
) -> DQNAgent:
    """Train on `env`; every cfg.eval_every steps call evaluate(agent) and keep the checkpoint
    with the highest evaluate()[select_key] as best.pt. Logs go to train_log.csv / eval_log.csv."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    agent = DQNAgent(env.observation_space.shape[0], cfg)
    buf = ReplayBuffer(cfg.buffer_size, agent.obs_dim, seed=cfg.seed)
    nstep = NStepCollector(cfg.n_step, cfg.gamma, buf)

    train_f = open(out / "train_log.csv", "w", newline="")
    train_w = csv.writer(train_f)
    train_w.writerow(["step", "episodes", "eps", "mean_return_per_note", "mean_loss", "steps_per_sec"])
    eval_f = open(out / "eval_log.csv", "w", newline="")
    eval_w = None

    obs, info = env.reset(seed=cfg.seed)
    ep_ret, ep_len, episodes = 0.0, 0, 0
    recent_ret = deque(maxlen=100)
    losses = deque(maxlen=1000)
    best = -float("inf")
    t0 = time.time()
    decay_steps = max(1, int(cfg.eps_decay_frac * cfg.total_steps))

    for step in range(1, cfg.total_steps + 1):
        eps = max(cfg.eps_end, cfg.eps_start - (cfg.eps_start - cfg.eps_end) * step / decay_steps)
        mask = info["action_mask"]
        a = agent.act(obs, mask, eps)
        obs2, r, term, trunc, info = env.step(a)
        nstep.push(obs, a, r, obs2, info["action_mask"], term)
        ep_ret += r
        ep_len += 1
        obs = obs2
        if term or trunc:
            recent_ret.append(ep_ret / max(1, ep_len))
            episodes += 1
            ep_ret, ep_len = 0.0, 0
            obs, info = env.reset()

        if step >= cfg.learning_starts and step % cfg.train_freq == 0:
            for _ in range(cfg.gradient_steps):
                losses.append(agent.update(buf.sample(cfg.batch_size, cfg.device)))
        if step % cfg.target_update == 0:
            agent.sync_target()

        if step % cfg.log_every == 0:
            sps = step / (time.time() - t0)
            train_w.writerow(
                [step, episodes, round(eps, 4),
                 round(float(np.mean(recent_ret)), 4) if recent_ret else "",
                 round(float(np.mean(losses)), 5) if losses else "", round(sps, 1)]
            )
            train_f.flush()

        if evaluate is not None and (step % cfg.eval_every == 0 or step == cfg.total_steps):
            m = evaluate(agent)
            m = {"step": step, **m}
            if eval_w is None:
                eval_w = csv.DictWriter(eval_f, fieldnames=list(m.keys()))
                eval_w.writeheader()
            eval_w.writerow(m)
            eval_f.flush()
            score = m.get(select_key, -float("inf"))
            tag = ""
            if score > best:
                best = score
                agent.save(out / "best.pt", extra)
                tag = "  <- best"
            print(
                f"step {step:>8}  eps {eps:.3f}  "
                + "  ".join(f"{k} {v:.4f}" for k, v in m.items() if k != "step" and isinstance(v, float))
                + tag,
                flush=True,
            )

    agent.save(out / "last.pt", extra)
    train_f.close()
    eval_f.close()
    return agent
