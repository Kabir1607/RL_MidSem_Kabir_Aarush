"""Phase 4: metrics, plotting, evaluation summary, spec test 8 (deterministic evaluation), and a live
check that evaluation is stratified."""

import gymnasium as gym
import numpy as np
import pandas as pd
import pytest
from sb3_contrib import MaskablePPO

import config
from callbacks import masked_greedy
from encoder import Encoder
from evaluate import evaluate, make_eval_env, summarise
from metrics import auc, baseline_win_rate, stability, threshold_step, wilson
from plotting import plot_run
from recorder import EPISODE_FIELDS, EVAL_FIELDS


# ---------- metrics ----------

def test_baseline_win_rates():
    assert baseline_win_rate("lowvar") == pytest.approx(0.4287)
    assert baseline_win_rate("highvar") == pytest.approx(0.4373)


def test_wilson_contains_p():
    lo, hi = wilson(60, 120)
    assert lo < 0.5 < hi and hi - lo == pytest.approx(0.176, abs=0.002)


def test_auc_is_average_win_rate():
    assert auc([0, 10, 20], [0.2, 0.2, 0.2]) == pytest.approx(0.2)
    assert auc([0, 10, 20], [0.0, 0.5, 1.0]) == pytest.approx(0.5)
    assert np.isnan(auc([0], [0.3]))


def test_threshold_needs_consecutive_and_is_nan_if_never():
    steps = [0, 10, 20, 30, 40]
    assert threshold_step(steps, [0.1, 0.6, 0.2, 0.6, 0.6], 0.5) == 30  # single spike at 10 doesn't count
    assert np.isnan(threshold_step(steps, [0.1, 0.6, 0.2, 0.6, 0.2], 0.5))


def test_stability():
    drop, std = stability([0.2, 0.5, 0.3, 0.4, 0.4, 0.4])
    assert drop == pytest.approx(0.2)
    assert std == pytest.approx(np.std([0.5, 0.3, 0.4, 0.4, 0.4], ddof=1))


# ---------- evaluation summary ----------

def end(opp, outcome, steps=15):
    return dict(opp_team=opp, outcome=outcome, len_steps=steps, len_turns=steps - 3)


def test_summarise():
    ends = [end("A", 1), end("A", -1), end("B", 1), end("B", 1), end("C", 0), end("C", -1)]
    s = summarise(ends)
    assert (s["wins"], s["losses"], s["ties"]) == (3, 2, 1)
    assert (s["wr_vs_A"], s["wr_vs_B"], s["wr_vs_C"]) == (0.5, 1.0, 0.0)


def test_summarise_rejects_unstratified():
    with pytest.raises(RuntimeError):
        summarise([end("A", 1), end("A", 1), end("B", 1)])


# ---------- plotting ----------

def test_plot_run(tmp_path):
    steps = np.arange(0, 60_000, 10_000)
    wr = np.array([0.3, 0.4, 0.55, 0.6, 0.58, 0.62])
    ev = pd.DataFrame({f: 0 for f in EVAL_FIELDS}, index=range(len(steps)))
    ev = ev.assign(run_id="ppo_lowvar_s0", team="lowvar", step=steps, win_rate=wr, wilson_low=wr - 0.08,
                   wilson_high=wr + 0.08)
    ev.to_csv(tmp_path / "eval.csv", index=False)
    n = 3000
    episodes = pd.DataFrame(0, index=range(n), columns=EPISODE_FIELDS)
    episodes.assign(step=np.linspace(16, 50_000, n).astype(int), battle_idx=np.arange(1, n + 1), len_steps=16) \
        .to_csv(tmp_path / "train_episodes.csv", index=False)
    plot_run(tmp_path)
    assert (tmp_path / "learning_curve.png").stat().st_size > 10_000
    assert not (tmp_path / "learning_curve.tmp.png").exists()


# ---------- spec test 8: evaluating a checkpoint is deterministic ----------

class MaskedDummyEnv(gym.Env):
    """Same spaces as the real env, random legal-action masks; no server needed."""
    observation_space = Encoder("lowvar").observation_space
    action_space = gym.spaces.Discrete(config.ACTION_SPACE_SIZE)

    def reset(self, *, seed=None, options=None):
        return np.zeros(self.observation_space.shape, np.float32), {}

    def step(self, action):
        return np.zeros(self.observation_space.shape, np.float32), 0.0, True, False, {}

    def action_masks(self):
        m = np.zeros(config.ACTION_SPACE_SIZE, bool)
        m[[0, 6, 7]] = True
        return m


def test_checkpoint_evaluation_is_deterministic(tmp_path):
    model = MaskablePPO("MlpPolicy", MaskedDummyEnv(), seed=0, device="cpu", **config.PPO_KWARGS)
    model.save(tmp_path / "ckpt")
    rng = np.random.default_rng(0)
    space = MaskedDummyEnv.observation_space
    obs = rng.uniform(space.low, space.high, size=(200, *space.shape)).astype(np.float32)
    masks = rng.random((200, config.ACTION_SPACE_SIZE)) < 0.3
    masks[:, 6] = True
    runs = []
    for _ in range(2):
        policy = masked_greedy(MaskablePPO.load(tmp_path / "ckpt", device="cpu"))
        runs.append([policy(o, m) for o, m in zip(obs, masks)])
    assert runs[0] == runs[1]
    assert all(m[a] for a, m in zip(runs[0], masks))


# ---------- live: stratified evaluation ----------

def test_evaluate_live_is_stratified(showdown):
    env = make_eval_env("lowvar", seed=2, algo="tst")
    rng = np.random.default_rng(2)
    try:
        s = evaluate(lambda obs, mask: int(rng.choice(np.flatnonzero(mask))), env, 6)
        assert env.opp_teams.sent == ["A", "A", "B", "B", "C", "C"]
        s2 = evaluate(lambda obs, mask: int(np.flatnonzero(mask)[0]), env, 3, opponent="random")
        assert env.opp_teams.sent[6:] == ["A", "B", "C"]
    finally:
        env.close()
    assert s["eval_n"] == 6 and s["wins"] + s["losses"] + s["ties"] == 6
    assert s2["eval_n"] == 3
