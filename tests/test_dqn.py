"""DQN tests. Needs torch; the short training test takes about a minute on a laptop CPU."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fingerly.baselines import OptimalCost  # noqa: E402
from fingerly.cost import sequence_cost  # noqa: E402
from fingerly.dqn import DQNAgent, DQNConfig, NStepCollector, ReplayBuffer, train_dqn  # noqa: E402
from fingerly.env import FingeringEnv, run_policy  # noqa: E402
from fingerly.synth import synthetic_passages  # noqa: E402


def test_nstep_returns():
    buf = ReplayBuffer(100, 2)
    col = NStepCollector(3, 0.5, buf)
    o = np.zeros(2, np.float32)
    m = np.ones(5, bool)
    for t, r in enumerate([1.0, 2.0, 4.0, 8.0]):
        col.push(o + t, 0, r, o + t + 1, m, terminal=(t == 3))
    # first transition: 1 + .5*2 + .25*4 = 3, bootstraps from state 3 with gamma^3
    assert buf.rew[0] == pytest.approx(3.0) and buf.disc[0] == pytest.approx(0.125)
    assert buf.next_obs[0][0] == 3
    # at the end the remaining ones are flushed without bootstrapping
    assert list(buf.rew[1:4]) == pytest.approx([2 + 2 + 2, 4 + 4, 8])
    assert list(buf.disc[1:4]) == [0, 0, 0]


def test_masked_action_never_invalid():
    agent = DQNAgent(24, DQNConfig(seed=0))
    obs = np.random.randn(24).astype(np.float32)
    for _ in range(50):
        mask = np.random.rand(5) < 0.4
        mask[np.random.randint(5)] = True
        assert mask[agent.act(obs, mask, eps=0.0)]
        assert mask[agent.act(obs, mask, eps=1.0)]


def test_short_training_approaches_optimal_cost(tmp_path):
    train = synthetic_passages(300, seed=0)
    val = synthetic_passages(40, seed=1)
    env = FingeringEnv(train)
    cfg = DQNConfig(total_steps=30_000, learning_starts=1_000, eval_every=10_000, seed=0)
    agent = train_dqn(env, cfg, tmp_path)
    eval_env = FingeringEnv(val)
    pol = agent.policy()
    agent_cost = np.mean([sequence_cost(p.pitches, run_policy(eval_env, p, pol))[0] / len(p) for p in val])
    dp = OptimalCost()
    dp_cost = np.mean([sequence_cost(p.pitches, dp.predict(p))[0] / len(p) for p in val])
    rnd = np.random.default_rng(0)
    rand_cost = np.mean([sequence_cost(p.pitches, run_policy(
        eval_env, p, lambda o, m: int(rnd.choice(np.flatnonzero(m)))))[0] / len(p) for p in val])
    # learning: much closer to the optimum than to random
    assert agent_cost < dp_cost + 0.3 * (rand_cost - dp_cost)
    assert (tmp_path / "last.pt").exists()
    DQNAgent.load(tmp_path / "last.pt")
