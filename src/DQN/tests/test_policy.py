"""Inference checkpoint round trips and compatibility checks, without a server."""

import numpy as np
import pytest
import torch

import config
from DQN.agent import DQNSettings, greedy_action
from DQN.model import QNetwork
from DQN.policy import load_policy, save_checkpoint
from encoder import OBS_SIZE


@pytest.fixture
def checkpoint(tmp_path):
    # Real Gymnasium spaces expose NumPy integer dimensions. They must not leak
    # into metadata, or weights_only=True rejects an otherwise valid checkpoint.
    model = QNetwork(np.int64(OBS_SIZE), np.int64(config.ACTION_SPACE_SIZE), config.NET_ARCH)
    path = save_checkpoint(tmp_path / "step_0", model, np.int64(0), "lowvar", np.int64(0), DQNSettings())
    return model, path


def test_roundtrip_is_exact_deterministic_and_legal(checkpoint):
    model, path = checkpoint
    obs = np.random.default_rng(5).random(OBS_SIZE, dtype=np.float32)
    mask = np.zeros(config.ACTION_SPACE_SIZE, dtype=bool)
    mask[[1, 3, 6, 9]] = True
    with torch.no_grad():
        expected = greedy_action(model(torch.from_numpy(obs)[None])[0].numpy(), mask)
    policies = [load_policy(path), load_policy(path)]
    assert all(policy(obs, mask) == expected for policy in policies for _ in range(3))
    saved = torch.load(path, weights_only=True)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(saved["q_net"][key], value)
    assert saved["format_version"] == 1
    assert saved["step"] == 0
    assert not path.with_suffix(".pt.tmp").exists()
    with pytest.raises(ValueError, match="empty"):
        policies[0](obs, np.zeros_like(mask))
    with pytest.raises(ValueError, match="observation shape"):
        policies[0](obs[:2], mask)


@pytest.mark.parametrize("key,value", [("format_version", 99), ("algo", "ppo"), ("obs_size", 2),
                                      ("action_size", 3), ("net_arch", [16]), ("gamma", 0.5)])
def test_incompatible_metadata_fails(checkpoint, key, value):
    _, path = checkpoint
    data = torch.load(path, weights_only=True)
    data[key] = value
    torch.save(data, path)
    with pytest.raises(ValueError, match=key):
        load_policy(path)


def test_missing_weights_and_corruption_fail(checkpoint):
    _, path = checkpoint
    data = torch.load(path, weights_only=True)
    del data["q_net"]
    torch.save(data, path)
    with pytest.raises(ValueError, match="q_net"):
        load_policy(path)
    path.write_bytes(b"not a checkpoint")
    with pytest.raises(Exception):
        load_policy(path)


def test_failed_save_does_not_replace_previous_checkpoint(checkpoint, monkeypatch):
    model, path = checkpoint
    original = path.read_bytes()

    def fail(data, temporary):
        temporary.write_bytes(b"partial")
        raise OSError("simulated disk error")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="disk error"):
        save_checkpoint(path, model, 5, "lowvar", 0, DQNSettings())
    assert path.read_bytes() == original
    assert not path.with_suffix(".pt.tmp").exists()
