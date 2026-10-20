"""Exercise decision boundaries and shared integration without long training jobs."""

import csv
from dataclasses import replace
import io
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import config
from DQN.agent import DQNAgent, DQNSettings
from DQN import run_batch as dqn_batch
from DQN import train_dqn
from DQN.replay import ReplayBuffer


class FakeEnv:
    """Terminal observations differ from reset observations, exposing reset mistakes."""

    def __init__(self, truncate=False, episode_steps=3):
        self.truncate = truncate
        self.episode_steps = episode_steps
        self.steps = 0
        self.resets = 0
        self.closed = False
        self.finished = False

    def get_wrapper_attr(self, name):
        assert name == "action_masks"
        return self.action_masks

    def action_masks(self):
        return np.array([False, False, False] if self.finished and not self.truncate
                        else [True, False, True])

    def reset(self, **kwargs):
        self.resets += 1
        self.local_step = 0
        self.finished = False
        return np.array([-5, -5], dtype=np.float32), {}

    def step(self, action):
        assert self.action_masks()[action]
        self.steps += 1
        self.local_step += 1
        self.finished = self.local_step == self.episode_steps
        obs = np.array([99 if self.finished else self.local_step, 1], dtype=np.float32)
        info = {"battle_end": {"opp_team": "A"}} if self.finished else {}
        return obs, 1.0, self.finished and not self.truncate, self.finished and self.truncate, info

    def close(self):
        self.closed = True


class FakeRecorder:
    def __init__(self, directory):
        self.dir = directory
        self.battles = 0
        self.evaluations = []
        self.final_state = None
        self.closed = False

    def evaluate(self, step, policy, save):
        self.evaluations.append(step)
        policy(np.zeros(2, dtype=np.float32), np.array([True, False, True]))
        save(self.dir / "ckpt" / f"step_{step}")
        return dict(win_rate=0.5, wilson_low=0.4, wilson_high=0.6)

    def episode_end(self, step, end):
        self.battles += 1

    def log_scalars(self, step, values):
        pass

    def final(self, step, policy):
        self.final_state = torch.load(self.dir / "ckpt" / f"step_{step}.pt", weights_only=True)["q_net"]
        return [dict(opponent="heuristic", win_rate=0.5)]

    def close(self):
        self.closed = True


@pytest.mark.parametrize("truncate", [False, True])
def test_loop_stores_pre_reset_states_and_exact_budget(tmp_path, monkeypatch, truncate):
    monkeypatch.setattr(config, "NET_ARCH", [8, 8])
    monkeypatch.setattr(config, "EVAL_FREQ", 4)
    settings = replace(DQNSettings(), total_steps=8, buffer_size=20, batch_size=2,
                       learning_starts=0, train_freq=2, target_update_interval=4)
    agent = DQNAgent(2, 3, settings, np.random.default_rng(2))
    replay = ReplayBuffer(20, 2, 3, np.random.default_rng(3))
    env, rec = FakeEnv(truncate=truncate), FakeRecorder(tmp_path)
    diagnostic = io.StringIO()
    before_eval = []
    original_eval = rec.evaluate

    def evaluate(step, policy, save):
        before = agent.exploration_rng.bit_generator.state
        replay_before = replay.rng.bit_generator.state
        result = original_eval(step, policy, save)
        assert agent.exploration_rng.bit_generator.state == before
        assert replay.rng.bit_generator.state == replay_before
        before_eval.append(agent.update_count)
        return result

    rec.evaluate = evaluate
    train_dqn.run_training(env, rec, agent, replay, settings, "lowvar", 0, diagnostic)
    assert env.steps == len(replay) == 8
    assert env.resets == 3  # two completed episodes, then a budget-cut unfinished one
    assert rec.evaluations == [0, 4, 8]
    assert before_eval == [0, 2, 4]  # evaluation/checkpoint comes after the scheduled update
    assert rec.battles == 2
    assert replay.next_obs[2, 0] == replay.next_obs[5, 0] == 99
    assert replay.obs[3, 0] == -5
    assert bool(replay.terminated[2]) is not truncate
    assert replay.next_masks[2].any() == truncate
    for key, tensor in agent.online.state_dict().items():
        torch.testing.assert_close(tensor, rec.final_state[key])
    rows = list(csv.DictReader(io.StringIO(diagnostic.getvalue())))
    assert len(rows) == 2  # no fabricated row for the partial final episode
    assert float(rows[0]["discounted_return"]) == pytest.approx(1 + 0.99 + 0.99**2)


def test_no_extra_reset_at_final_episode_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "NET_ARCH", [4])
    monkeypatch.setattr(config, "EVAL_FREQ", 3)
    settings = replace(DQNSettings(), total_steps=6)
    env, rec = FakeEnv(), FakeRecorder(tmp_path)
    agent = DQNAgent(2, 3, settings, np.random.default_rng(0))
    replay = ReplayBuffer(10, 2, 3, np.random.default_rng(0))
    train_dqn.run_training(env, rec, agent, replay, settings, "lowvar", 0, io.StringIO(), True)
    assert env.resets == 2
    assert rec.final_state is None


def test_train_refuses_existing_run_before_connecting(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path)
    directory = tmp_path / "dqn_lowvar_s0"
    directory.mkdir()
    (directory / "keep.txt").write_text("existing result")
    monkeypatch.setattr(train_dqn, "make_env", lambda *a, **kw: pytest.fail("must not connect"))
    with pytest.raises(FileExistsError):
        train_dqn.train("lowvar", 0, DQNSettings())
    assert (directory / "keep.txt").read_text() == "existing result"


def test_train_closes_env_when_recorder_creation_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path)
    env = FakeEnv()
    env.observation_space = SimpleNamespace(shape=(train_dqn.OBS_SIZE,))
    env.action_space = SimpleNamespace(n=config.ACTION_SPACE_SIZE)
    monkeypatch.setattr(train_dqn, "make_env", lambda *a, **kw: env)

    def fail_recorder(*args, **kwargs):
        raise OSError("recorder unavailable")

    monkeypatch.setattr(train_dqn, "RunRecorder", fail_recorder)
    with pytest.raises(OSError, match="unavailable"):
        train_dqn.train("lowvar", 0, replace(DQNSettings(), buffer_size=10))
    assert env.closed


def test_train_closes_all_resources_on_loop_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path)
    env, rec = FakeEnv(), FakeRecorder(tmp_path)
    env.observation_space = SimpleNamespace(shape=(train_dqn.OBS_SIZE,))
    env.action_space = SimpleNamespace(n=config.ACTION_SPACE_SIZE)
    rec.write_run_config = lambda *a, **kw: None
    monkeypatch.setattr(train_dqn, "make_env", lambda *a, **kw: env)
    monkeypatch.setattr(train_dqn, "RunRecorder", lambda *a, **kw: rec)

    def fail_loop(*args, **kwargs):
        raise RuntimeError("battle disconnected")

    monkeypatch.setattr(train_dqn, "run_training", fail_loop)
    with pytest.raises(RuntimeError, match="disconnected"):
        train_dqn.train("lowvar", 0, replace(DQNSettings(), buffer_size=10))
    assert env.closed and rec.closed
    # The file handle has also been closed by ExitStack.
    (tmp_path / "q_returns.csv").unlink()


def test_batch_adapter_delegates_and_propagates_failure(monkeypatch):
    calls = []
    monkeypatch.setattr("sys.argv", ["run_batch.py", "--parallel", "2"])
    monkeypatch.setattr(dqn_batch.shared_batch, "TRAINERS", dict(dqn_batch.shared_batch.TRAINERS))
    monkeypatch.setattr(dqn_batch.shared_batch, "launch", lambda *args: calls.append(args) or 1)
    assert dqn_batch.main() == 1
    assert calls == [("dqn", 2, False)]
    assert dqn_batch.shared_batch.TRAINERS["dqn"].name == "train_dqn.py"
    assert len(dqn_batch.shared_batch.JOBS) == 2 * len(config.SEEDS)
    monkeypatch.setattr("sys.argv", ["run_batch.py", "--status"])
    monkeypatch.setattr(dqn_batch.shared_batch, "status", lambda algo: calls.append(algo))
    assert dqn_batch.main() == 0 and calls[-1] == "dqn"
    monkeypatch.setattr("sys.argv", ["run_batch.py", "--parallel", "0"])
    with pytest.raises(SystemExit):
        dqn_batch.main()


def test_shared_recorder_outputs_are_accepted_by_existing_analyzer(tmp_path, monkeypatch):
    import analyze
    import recorder as shared_recorder
    from DQN.model import QNetwork
    from DQN.policy import load_policy, save_checkpoint
    from evaluate import SUMMARY_FIELDS

    monkeypatch.setattr(config, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(config, "TOTAL_STEPS", 10)
    monkeypatch.setattr(config, "EVAL_FREQ", 5)
    # This test exercises the actual recorder/CSV contract, without playing thousands of battles.
    account = SimpleNamespace(username="fake")
    eval_env = SimpleNamespace(env=SimpleNamespace(env=SimpleNamespace(agent1=account, agent2=account)),
                               close=lambda: None)
    monkeypatch.setattr(shared_recorder, "make_eval_env", lambda *a: eval_env)

    def summary(n):
        result = {key: 0.0 for key in SUMMARY_FIELDS}
        result.update(eval_n=n, wins=n, losses=0, ties=0, win_rate=1.0)
        return result

    monkeypatch.setattr(shared_recorder, "evaluate", lambda policy, env, n, opponent: summary(n))
    monkeypatch.setattr(shared_recorder, "final_evaluation", lambda policy, env: [
        dict(opponent=opp, **summary(n)) for opp, n in analyze.FINAL_EVAL])
    monkeypatch.setattr(shared_recorder, "plot_run", lambda directory: None)
    model = QNetwork(train_dqn.OBS_SIZE, config.ACTION_SPACE_SIZE, config.NET_ARCH)
    settings = replace(DQNSettings(), total_steps=10)
    for seed in (3, 4):
        rec = shared_recorder.RunRecorder("dqn", "lowvar", seed)
        try:
            rec.write_run_config(settings.hyperparameters())
            for step in (0, 5, 10):
                rec.evaluate(step, lambda obs, mask: 0,
                             lambda path: save_checkpoint(path, model, step, "lowvar", seed, settings))
            rec.episode_end(10, dict(opp_team="A", outcome=1, len_steps=10, len_turns=8,
                                    disc_shaping_sum=0, ep_return_shaped=1))
            rec.final(10, lambda obs, mask: 0)
        finally:
            rec.close()
        result = analyze.load_run(rec.dir)
        assert result.errors == [] and result.status == "complete"
        assert len(list((rec.dir / "ckpt").glob("*.pt"))) == 3
        for checkpoint in (rec.dir / "ckpt").glob("*.pt"):
            policy = load_policy(checkpoint)
            assert policy(np.zeros(train_dqn.OBS_SIZE, dtype=np.float32),
                          np.ones(config.ACTION_SPACE_SIZE, dtype=bool)) in range(config.ACTION_SPACE_SIZE)
        assert list(pd.read_csv(rec.dir / "eval.csv").columns) == shared_recorder.EVAL_FIELDS
