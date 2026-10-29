"""The four agents of Iman & Ahn (2024), with the settings of sections III-C and IV.

All networks: input 274 -> 512 -> 256 -> 64 (ReLU) -> output. Adam, learning rate 1e-3,
discount 0.99, MSE loss, 500 episodes (one episode = the whole piece).

DQN / DDQN   Q-network with 5 outputs; replay memory of the latest 10,000 steps, batch 32,
             target network synced every 100 steps, one gradient step per environment step.
             DDQN picks the next action with the online network and evaluates it with the
             target network (van Hasselt et al., 2016).
PG           REINFORCE with a learned baseline V(s; w) (5-way softmax policy, 1-output value
             network), updated once per episode with Monte-Carlo returns.
A2C          one-step advantage actor-critic: A = r + gamma V(s') - V(s), updated every step.

Not given in the paper, so chosen here (stated in the report):
* epsilon-greedy for DQN / DDQN decays linearly from 1.0 to 0.01 over the first 100 episodes;
* learning starts as soon as the replay memory holds one batch;
* the fingering we score after training is the greedy one (argmax Q, or argmax pi).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .env import OBS_DIM, PianoFingeringEnv

ALGOS = ("DQN", "DDQN", "PG", "A2C")


@dataclass
class PaperConfig:
    episodes: int = 500
    gamma: float = 0.99
    lr: float = 1e-3
    buffer: int = 10_000
    batch: int = 32
    target_update: int = 100
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_episodes: int = 100


def mlp(out: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(OBS_DIM, 512), nn.ReLU(),
        nn.Linear(512, 256), nn.ReLU(),
        nn.Linear(256, 64), nn.ReLU(),
        nn.Linear(64, out),
    )


@torch.no_grad()
def greedy_fingering(env: PianoFingeringEnv, net: nn.Module) -> List[int]:
    obs, _ = env.reset()
    done = False
    while not done:
        a = int(net(torch.as_tensor(obs).unsqueeze(0)).argmax(1))
        obs, _, done, _, _ = env.step(a)
    return list(env.fingers)


def _run_dqn(env, cfg: PaperConfig, rng, double: bool):
    q, q_t = mlp(5), mlp(5)
    q_t.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=cfg.lr)
    n = cfg.buffer
    S = np.zeros((n, OBS_DIM), np.float32)
    S2 = np.zeros((n, OBS_DIM), np.float32)
    A = np.zeros(n, np.int64)
    R = np.zeros(n, np.float32)
    D = np.zeros(n, np.float32)
    size = ptr = steps = 0
    returns = []
    for ep in range(cfg.episodes):
        eps = max(cfg.eps_end, cfg.eps_start - (cfg.eps_start - cfg.eps_end) * ep / cfg.eps_episodes)
        obs, _ = env.reset()
        done, G = False, 0.0
        while not done:
            if rng.random() < eps:
                a = int(rng.integers(5))
            else:
                with torch.no_grad():
                    a = int(q(torch.as_tensor(obs).unsqueeze(0)).argmax(1))
            obs2, r, done, _, _ = env.step(a)
            S[ptr], A[ptr], R[ptr], S2[ptr], D[ptr] = obs, a, r, obs2, float(done)
            ptr, size = (ptr + 1) % n, min(size + 1, n)
            obs, G, steps = obs2, G + r, steps + 1
            if size >= cfg.batch:
                idx = rng.integers(0, size, cfg.batch)
                s, a_, r_, s2, d_ = (torch.as_tensor(x[idx]) for x in (S, A, R, S2, D))
                with torch.no_grad():
                    if double:
                        a2 = q(s2).argmax(1, keepdim=True)
                        nq = q_t(s2).gather(1, a2).squeeze(1)
                    else:
                        nq = q_t(s2).max(1).values
                    target = r_ + cfg.gamma * (1 - d_) * nq
                pred = q(s).gather(1, a_.unsqueeze(1)).squeeze(1)
                loss = F.mse_loss(pred, target)
                opt.zero_grad()
                loss.backward()
                opt.step()
            if steps % cfg.target_update == 0:
                q_t.load_state_dict(q.state_dict())
        returns.append(G)
    return q, returns


def _run_pg(env, cfg: PaperConfig, rng):
    pi, v = mlp(5), mlp(1)
    opt_pi = torch.optim.Adam(pi.parameters(), lr=cfg.lr)
    opt_v = torch.optim.Adam(v.parameters(), lr=cfg.lr)
    returns = []
    for ep in range(cfg.episodes):
        obs, _ = env.reset()
        done = False
        S, A, R = [], [], []
        while not done:
            with torch.no_grad():
                p = torch.softmax(pi(torch.as_tensor(obs).unsqueeze(0)), 1)[0].numpy()
            a = int(rng.choice(5, p=p / p.sum()))
            obs2, r, done, _, _ = env.step(a)
            S.append(obs), A.append(a), R.append(r)
            obs = obs2
        G, g = np.zeros(len(R), np.float32), 0.0
        for t in range(len(R) - 1, -1, -1):
            g = R[t] + cfg.gamma * g
            G[t] = g
        s = torch.as_tensor(np.array(S))
        a = torch.as_tensor(A)
        G_t = torch.as_tensor(G)
        value = v(s).squeeze(1)
        adv = (G_t - value).detach()
        logp = torch.log_softmax(pi(s), 1).gather(1, a.unsqueeze(1)).squeeze(1)
        loss_pi = -(logp * adv).sum()
        loss_v = F.mse_loss(value, G_t)
        opt_pi.zero_grad(), loss_pi.backward(), opt_pi.step()
        opt_v.zero_grad(), loss_v.backward(), opt_v.step()
        returns.append(float(sum(R)))
    return pi, returns


def _run_a2c(env, cfg: PaperConfig, rng):
    pi, v = mlp(5), mlp(1)
    opt_pi = torch.optim.Adam(pi.parameters(), lr=cfg.lr)
    opt_v = torch.optim.Adam(v.parameters(), lr=cfg.lr)
    returns = []
    for ep in range(cfg.episodes):
        obs, _ = env.reset()
        done, G = False, 0.0
        while not done:
            s = torch.as_tensor(obs).unsqueeze(0)
            logits = pi(s)
            p = torch.softmax(logits, 1)[0].detach().numpy()
            a = int(rng.choice(5, p=p / p.sum()))
            obs2, r, done, _, _ = env.step(a)
            value = v(s).squeeze()
            with torch.no_grad():
                nv = 0.0 if done else v(torch.as_tensor(obs2).unsqueeze(0)).squeeze()
                target = r + cfg.gamma * nv
            adv = (target - value).detach()
            loss_pi = -torch.log_softmax(logits, 1)[0, a] * adv
            loss_v = F.mse_loss(value, torch.as_tensor(target, dtype=torch.float32))
            opt_pi.zero_grad(), loss_pi.backward(), opt_pi.step()
            opt_v.zero_grad(), loss_v.backward(), opt_v.step()
            obs, G = obs2, G + r
        returns.append(G)
    return pi, returns


def train(algo: str, env: PianoFingeringEnv, seed: int, cfg: PaperConfig = PaperConfig()) -> Dict:
    """Train one agent on one environment; returns the episode returns, the greedy fingering after
    training and the wall-clock run time."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    if algo in ("DQN", "DDQN"):
        net, returns = _run_dqn(env, cfg, rng, double=algo == "DDQN")
    elif algo == "PG":
        net, returns = _run_pg(env, cfg, rng)
    elif algo == "A2C":
        net, returns = _run_a2c(env, cfg, rng)
    else:
        raise ValueError(f"unknown algorithm {algo}")
    runtime = time.time() - t0
    return {"returns": returns, "fingering": greedy_fingering(env, net), "runtime": runtime}
