"""Numerical and legality checks that do not require Pokemon Showdown."""

from dataclasses import replace

import numpy as np
import pytest
import torch

import config
from DQN.agent import DQNAgent, DQNSettings, greedy_action, masked_td_targets
from DQN.model import QNetwork
from DQN.replay import ReplayBuffer


def test_model_shape_and_raw_outputs():
    model = QNetwork(7, 5, (8, 8))
    assert model(torch.zeros(3, 7)).shape == (3, 5)
    with torch.no_grad():
        model.layers[-1].bias.fill_(-2)
    values = model(torch.zeros(3, 7))
    assert torch.isfinite(values).all()
    assert (values < 0).all()  # Q-values are not probabilities or a softmax.


def test_greedy_is_legal_and_breaks_ties_by_index():
    rng = np.random.default_rng(3)
    for _ in range(100):
        mask = rng.random(26) < 0.3
        mask[rng.integers(26)] = True
        q = rng.normal(size=26)
        q[~mask] = 1e6
        assert mask[greedy_action(q, mask)]
    assert greedy_action([3, 3, 99], [True, True, False]) == 0
    with pytest.raises(ValueError, match="empty"):
        greedy_action([0, 1], [False, False])
    with pytest.raises(ValueError, match="matching"):
        greedy_action([0, 1], [True])
    with pytest.raises(FloatingPointError):
        greedy_action([np.nan, 1], [True, True])


def test_epsilon_schedule_and_update_timing():
    settings = DQNSettings()
    assert settings.epsilon(0) == 1
    assert settings.epsilon(10_000) == pytest.approx(0.525)
    assert settings.epsilon(20_000) == pytest.approx(0.05)
    assert settings.epsilon(200_000) == pytest.approx(0.05)
    assert not settings.should_update(5000, 5000)
    assert not settings.should_update(5001, 5001)
    assert settings.should_update(5004, 5004)
    assert not settings.should_update(5004, 1)
    smoke = DQNSettings.for_profile("smoke")
    assert (smoke.total_steps, smoke.opponent, smoke.suffix) == (20_000, "maxbp", "smoke")
    assert (smoke.learning_starts, smoke.exploration_steps) == (1000, 10_000)
    with pytest.raises(ValueError):
        DQNSettings.for_profile("double_dqn")


def test_exploration_and_evaluation_rng_are_independent(monkeypatch):
    monkeypatch.setattr(config, "NET_ARCH", [8, 8])
    seed_a, seed_b = np.random.SeedSequence(0).spawn(2)
    agent = DQNAgent(3, 5, DQNSettings(), np.random.default_rng(seed_a))
    mask = np.array([False, True, False, True, False])
    obs = np.zeros(3, dtype=np.float32)
    expected = np.random.default_rng(seed_a)
    for _ in range(100):
        expected.random()
        action = int(expected.choice(np.flatnonzero(mask)))
        assert agent.act(obs, mask, epsilon=1) == action
        assert mask[agent.act(obs, mask, epsilon=0)]
        expected.random()  # act consumes one epsilon draw even when epsilon is zero.
    state = agent.exploration_rng.bit_generator.state
    for _ in range(10):
        agent.greedy(obs, mask)
    assert state == agent.exploration_rng.bit_generator.state
    replay_rng = np.random.default_rng(seed_b)
    assert replay_rng.integers(1000) == np.random.default_rng(seed_b).integers(1000)


def test_targets_ignore_illegal_actions_and_handle_terminal_and_time_limit():
    rewards = torch.tensor([0.1, -1.0, 0.2])
    next_q = torch.tensor([[2.0, 1e6], [1e6, 1e6], [3.0, 1e6]], requires_grad=True)
    masks = torch.tensor([[True, False], [False, False], [True, False]])
    # Last row represents a time-limit truncation: it still has terminated=False.
    terminal = torch.tensor([False, True, False])
    targets = masked_td_targets(rewards, next_q, masks, terminal, 0.99)
    torch.testing.assert_close(targets, torch.tensor([2.08, -1.0, 3.17]))
    assert not targets.requires_grad
    changed = next_q.detach().clone()
    changed[:, 1] = -1e6
    torch.testing.assert_close(targets, masked_td_targets(rewards, changed, masks, terminal, 0.99))
    masks[0] = False
    with pytest.raises(ValueError, match="nonterminal"):
        masked_td_targets(rewards, next_q, masks, terminal, 0.99)


def test_all_terminal_targets_are_finite():
    result = masked_td_targets(torch.tensor([1.0, -1.0]), torch.full((2, 3), float("nan")),
                               torch.zeros(2, 3, dtype=torch.bool),
                               torch.ones(2, dtype=torch.bool), 0.99)
    torch.testing.assert_close(result, torch.tensor([1.0, -1.0]))


def test_replay_copies_wraps_and_samples_only_populated_rows():
    replay = ReplayBuffer(3, 2, 2, np.random.default_rng(4))
    obs = np.array([1, 2], dtype=np.float32)
    mask = np.array([True, False])
    replay.add(obs, 0, 1, obs + 1, mask, False)
    obs[:] = 99
    mask[:] = False
    batch = replay.sample(1)
    np.testing.assert_array_equal(batch.obs[0], [1, 2])
    np.testing.assert_array_equal(batch.next_masks[0], [True, False])
    batch.obs[:] = 90
    assert replay.obs[0, 0] == 1
    with pytest.raises(ValueError):
        replay.sample(2)
    for value in range(2, 6):
        replay.add([value, value], 0, value, [value + 1, value + 1], [True, False], False)
    batch = replay.sample(3)
    assert len(replay) == 3
    assert sorted(batch.rewards.tolist()) == [3, 4, 5]
    assert batch.obs.dtype == batch.next_obs.dtype == batch.rewards.dtype == np.float32
    assert batch.actions.dtype == np.int64
    assert batch.terminated.dtype == batch.next_masks.dtype == bool


def test_seeded_replay_sampling():
    buffers = [ReplayBuffer(10, 2, 2, np.random.default_rng(7)) for _ in range(2)]
    for replay in buffers:
        for i in range(10):
            replay.add([i, i], 0, i, [i, i], [True, False], False)
    np.testing.assert_array_equal(buffers[0].sample(4).rewards, buffers[1].sample(4).rewards)


def test_update_changes_online_but_not_target_and_clips_gradients(monkeypatch):
    monkeypatch.setattr(config, "NET_ARCH", [8, 8])
    settings = replace(DQNSettings(), max_grad_norm=0.01)
    agent = DQNAgent(2, 3, settings, np.random.default_rng(0))
    before = {k: v.clone() for k, v in agent.online.state_dict().items()}
    for key, value in agent.target.state_dict().items():
        torch.testing.assert_close(value, before[key])
    replay = ReplayBuffer(4, 2, 3, np.random.default_rng(0))
    for _ in range(4):
        replay.add([1, 1], 1, 10, [0, 0], [False, False, False], True)
    metrics = agent.update(replay.sample(4))
    assert all(np.isfinite(list(metrics.values())))
    assert agent.update_count == 1
    assert any(not torch.equal(value, before[key]) for key, value in agent.online.state_dict().items())
    for key, value in agent.target.state_dict().items():
        torch.testing.assert_close(value, before[key])
    assert all(p.grad is None for p in agent.target.parameters())
    norm = torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in agent.online.parameters()]))
    assert norm <= settings.max_grad_norm + 1e-6
    # Only the sampled action's output row participates in the regression loss.
    assert agent.online.layers[-1].weight.grad[0].abs().sum() == 0
    agent.sync_target()
    for online, target in zip(agent.online.parameters(), agent.target.parameters()):
        torch.testing.assert_close(online, target)
