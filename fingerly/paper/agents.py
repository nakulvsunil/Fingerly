"""The paper's four agents (README §A.1): DQN, Double DQN, REINFORCE with a state-value baseline
("PG") and A2C, all on the 512-256-64 ReLU MLP, Adam lr 1e-3, MSE losses, gamma 0.99,
500 episodes.

Settings the paper leaves open (README §A.7): epsilon-greedy decayed per step by 0.999 from
1.0 to 0.01 (item 4); PG updates once per episode on Monte-Carlo returns minus V(s), A2C
every step on the one-step TD advantage (item 8). Actor and critic are separate networks.
"""

from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .env import OBS_DIM, PaperFingeringEnv

ALGOS = ("dqn", "ddqn", "pg", "a2c")


@dataclass
class PaperConfig:
    episodes: int = 500
    gamma: float = 0.99
    lr: float = 1e-3
    buffer: int = 10_000
    batch: int = 32
    target_sync: int = 100
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay: float = 0.999
    device: str = "cpu"


def mlp(out: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(OBS_DIM, 512), nn.ReLU(),
        nn.Linear(512, 256), nn.ReLU(),
        nn.Linear(256, 64), nn.ReLU(),
        nn.Linear(64, out),
    )


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rollout(env: PaperFingeringEnv, act: Callable[[np.ndarray], int]) -> List[int]:
    """One greedy pass; returns the fingering (1..5)."""
    obs, _ = env.reset()
    done = False
    while not done:
        obs, _, done, _, _ = env.step(act(obs))
    return list(env.fingers)


def _result(rewards, fingers_by_ep, final, t0, net):
    best = int(np.argmax(rewards))
    return {
        "episode_rewards": rewards,
        "best_episode": best,
        "best_episode_reward": rewards[best],
        "best_episode_fingering": fingers_by_ep[best],
        "final_fingering": final,
        "seconds": time.time() - t0,
        "state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
    }


# ---------------------------------------------------------------------------------------
# DQN / Double DQN
# ---------------------------------------------------------------------------------------
def train_dqn(env: PaperFingeringEnv, cfg: PaperConfig, seed: int, double: bool = False) -> Dict:
    seed_all(seed)
    dev = torch.device(cfg.device)
    q, q_target = mlp(5).to(dev), mlp(5).to(dev)
    q_target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=cfg.lr)
    buf: deque = deque(maxlen=cfg.buffer)
    eps, steps = cfg.eps_start, 0
    rewards, fingers_by_ep = [], []
    t0 = time.time()

    def greedy(o):
        with torch.no_grad():
            return int(q(torch.as_tensor(o, device=dev).unsqueeze(0)).argmax(1).item())

    for _ in range(cfg.episodes):
        obs, _ = env.reset()
        done, total = False, 0.0
        while not done:
            a = random.randrange(5) if random.random() < eps else greedy(obs)
            nxt, r, done, _, _ = env.step(a)
            buf.append((obs, a, r, nxt, float(done)))
            obs, total = nxt, total + r
            steps += 1
            eps = max(cfg.eps_end, eps * cfg.eps_decay)
            if len(buf) >= cfg.batch:
                o, a_, r_, n_, d_ = zip(*random.sample(buf, cfg.batch))
                o = torch.as_tensor(np.stack(o), device=dev)
                n_ = torch.as_tensor(np.stack(n_), device=dev)
                a_ = torch.as_tensor(a_, device=dev)
                r_ = torch.as_tensor(r_, dtype=torch.float32, device=dev)
                d_ = torch.as_tensor(d_, dtype=torch.float32, device=dev)
                with torch.no_grad():
                    if double:
                        a_next = q(n_).argmax(1, keepdim=True)
                        q_next = q_target(n_).gather(1, a_next).squeeze(1)
                    else:
                        q_next = q_target(n_).max(1).values
                    target = r_ + cfg.gamma * (1 - d_) * q_next
                pred = q(o).gather(1, a_.unsqueeze(1)).squeeze(1)
                loss = F.mse_loss(pred, target)
                opt.zero_grad()
                loss.backward()
                opt.step()
            if steps % cfg.target_sync == 0:
                q_target.load_state_dict(q.state_dict())
        rewards.append(total)
        fingers_by_ep.append(list(env.fingers))
    return _result(rewards, fingers_by_ep, rollout(env, greedy), t0, q)


# ---------------------------------------------------------------------------------------
# REINFORCE with baseline ("PG") and A2C
# ---------------------------------------------------------------------------------------
def _policy_greedy(pi, dev):
    def act(o):
        with torch.no_grad():
            return int(pi(torch.as_tensor(o, device=dev).unsqueeze(0)).argmax(1).item())
    return act


def train_pg(env: PaperFingeringEnv, cfg: PaperConfig, seed: int) -> Dict:
    seed_all(seed)
    dev = torch.device(cfg.device)
    pi, v = mlp(5).to(dev), mlp(1).to(dev)
    opt_pi = torch.optim.Adam(pi.parameters(), lr=cfg.lr)
    opt_v = torch.optim.Adam(v.parameters(), lr=cfg.lr)
    rewards, fingers_by_ep = [], []
    t0 = time.time()
    for _ in range(cfg.episodes):
        obs, _ = env.reset()
        done, states, acts, rs = False, [], [], []
        while not done:
            with torch.no_grad():
                logits = pi(torch.as_tensor(obs, device=dev).unsqueeze(0))[0]
            a = int(torch.distributions.Categorical(logits=logits).sample().item())
            nxt, r, done, _, _ = env.step(a)
            states.append(obs), acts.append(a), rs.append(r)
            obs = nxt
        G, ret = 0.0, []
        for r in reversed(rs):
            G = r + cfg.gamma * G
            ret.append(G)
        ret = torch.as_tensor(ret[::-1], dtype=torch.float32, device=dev)
        s = torch.as_tensor(np.stack(states), device=dev)
        a = torch.as_tensor(acts, device=dev)
        values = v(s).squeeze(1)
        adv = (ret - values).detach()
        logp = torch.log_softmax(pi(s), 1).gather(1, a.unsqueeze(1)).squeeze(1)
        opt_pi.zero_grad()
        (-(logp * adv).mean()).backward()
        opt_pi.step()
        opt_v.zero_grad()
        F.mse_loss(values, ret).backward()
        opt_v.step()
        rewards.append(float(sum(rs)))
        fingers_by_ep.append(list(env.fingers))
    return _result(rewards, fingers_by_ep, rollout(env, _policy_greedy(pi, dev)), t0, pi)


def train_a2c(env: PaperFingeringEnv, cfg: PaperConfig, seed: int) -> Dict:
    seed_all(seed)
    dev = torch.device(cfg.device)
    pi, v = mlp(5).to(dev), mlp(1).to(dev)
    opt_pi = torch.optim.Adam(pi.parameters(), lr=cfg.lr)
    opt_v = torch.optim.Adam(v.parameters(), lr=cfg.lr)
    rewards, fingers_by_ep = [], []
    t0 = time.time()
    for _ in range(cfg.episodes):
        obs, _ = env.reset()
        done, total = False, 0.0
        while not done:
            s = torch.as_tensor(obs, device=dev).unsqueeze(0)
            logits = pi(s)[0]
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            nxt, r, done, _, _ = env.step(int(a.item()))
            value = v(s)[0, 0]
            with torch.no_grad():
                nv = 0.0 if done else v(torch.as_tensor(nxt, device=dev).unsqueeze(0))[0, 0]
                target = r + cfg.gamma * nv
            adv = (target - value).detach()
            opt_pi.zero_grad()
            (-dist.log_prob(a) * adv).backward()
            opt_pi.step()
            opt_v.zero_grad()
            F.mse_loss(value, torch.as_tensor(target, dtype=torch.float32, device=dev)).backward()
            opt_v.step()
            obs, total = nxt, total + r
        rewards.append(total)
        fingers_by_ep.append(list(env.fingers))
    return _result(rewards, fingers_by_ep, rollout(env, _policy_greedy(pi, dev)), t0, pi)


def train(algo: str, env: PaperFingeringEnv, cfg: PaperConfig, seed: int) -> Dict:
    if algo == "dqn":
        return train_dqn(env, cfg, seed, double=False)
    if algo == "ddqn":
        return train_dqn(env, cfg, seed, double=True)
    if algo == "pg":
        return train_pg(env, cfg, seed)
    if algo == "a2c":
        return train_a2c(env, cfg, seed)
    raise ValueError(f"unknown algo {algo!r}; choose from {ALGOS}")


def config_dict(cfg: PaperConfig) -> Dict:
    return asdict(cfg)
