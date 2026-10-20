"""analyze.py: the output contract (reference.md Section 7) is enforced, and the outputs are produced."""

import json

import numpy as np
import pandas as pd
import pytest

import analyze
import config
from evaluate import FINAL_EVAL
from recorder import EPISODE_FIELDS, EVAL_FIELDS, FINAL_FIELDS

STEPS = np.arange(0, config.TOTAL_STEPS + 1, config.EVAL_FREQ)


def make_run(runs_dir, algo, team, seed, hyper=None, last_step=config.TOTAL_STEPS, final=True):
    """A synthetic run directory exactly as RunRecorder writes it."""
    d = runs_dir / config.run_id(algo, team, seed)
    (d / "ckpt").mkdir(parents=True)
    cfg = {k: json.loads(json.dumps(v, default=str)) for k, v in vars(config).items() if k.isupper()}
    (d / "run_config.json").write_text(json.dumps(dict(
        run_id=d.name, algo=algo, team=team, seed=seed, eval_opponent="heuristic",
        hyperparameters=hyper or {"lr": 1e-4}, config=cfg, git={"commit": "abc", "dirty": False})))
    steps = STEPS[STEPS <= last_step]
    rng = np.random.default_rng(seed)
    wins = np.minimum(config.N_EVAL, (np.linspace(0.3, 1.0, len(steps)) * config.N_EVAL).astype(int))
    ev = pd.DataFrame(0, index=range(len(steps)), columns=EVAL_FIELDS).assign(
        run_id=d.name, algo=algo, team=team, seed=seed, step=steps, battles_trained=steps // 15,
        eval_n=config.N_EVAL, wins=wins, losses=config.N_EVAL - wins, win_rate=wins / config.N_EVAL,
        wilson_low=wins / config.N_EVAL - 0.05, wilson_high=wins / config.N_EVAL + 0.05)
    ev.to_csv(d / "eval.csv", index=False)
    n = 200
    pd.DataFrame(0, index=range(n), columns=EPISODE_FIELDS).assign(
        step=np.linspace(15, last_step, n).astype(int), battle_idx=np.arange(1, n + 1),
        len_steps=rng.integers(10, 20, n), opp_team="A").to_csv(d / "train_episodes.csv", index=False)
    fe = pd.DataFrame(columns=FINAL_FIELDS)
    if final:
        fe = pd.DataFrame([dict(run_id=d.name, algo=algo, team=team, seed=seed, step=config.TOTAL_STEPS, opponent=o,
                                eval_n=nb, wins=nb, losses=0, ties=0, win_rate=1.0, wilson_low=0.95, wilson_high=1.0,
                                wr_vs_A=1.0, wr_vs_B=1.0, wr_vs_C=1.0, mean_eval_len_steps=12, mean_eval_len_turns=10,
                                b_team=0.43, rl_lift=0.57 if o == "heuristic" else None) for o, nb in FINAL_EVAL],
                          columns=FINAL_FIELDS)
    fe.to_csv(d / "final_eval.csv", index=False)
    (d / "ckpt" / f"step_{config.TOTAL_STEPS}.pt").write_text("x")
    return d


def test_complete_run_is_valid(tmp_path):
    run = analyze.load_run(make_run(tmp_path, "dqn", "lowvar", 0))
    assert run.errors == [] and run.status == "complete"


def test_in_progress_run(tmp_path):
    run = analyze.load_run(make_run(tmp_path, "dqn", "lowvar", 0, last_step=50_000, final=False))
    assert run.status == "in progress"


@pytest.mark.parametrize("breakage", ["columns", "gap", "eval_n", "config", "past_budget", "final_rows"])
def test_contract_violations_are_invalid(tmp_path, breakage):
    d = make_run(tmp_path, "dqn", "highvar", 1)
    ev = pd.read_csv(d / "eval.csv")
    if breakage == "columns":
        ev.rename(columns={"win_rate": "winrate"}).to_csv(d / "eval.csv", index=False)
    elif breakage == "gap":
        ev.drop(index=3).to_csv(d / "eval.csv", index=False)
    elif breakage == "eval_n":
        ev.assign(eval_n=100).to_csv(d / "eval.csv", index=False)
    elif breakage == "config":
        cfg = json.loads((d / "run_config.json").read_text())
        cfg["config"]["GAMMA"] = 0.95
        (d / "run_config.json").write_text(json.dumps(cfg))
    elif breakage == "past_budget":
        ep = pd.read_csv(d / "train_episodes.csv")
        ep.loc[len(ep) - 1, "step"] = config.TOTAL_STEPS + 500
        ep.to_csv(d / "train_episodes.csv", index=False)
    elif breakage == "final_rows":
        pd.read_csv(d / "final_eval.csv").iloc[:2].to_csv(d / "final_eval.csv", index=False)
    assert analyze.load_run(d).status == "invalid"


def test_hyperparameters_must_match_within_algorithm(tmp_path):
    make_run(tmp_path, "dqn", "lowvar", 0, hyper={"lr": 1e-4})
    make_run(tmp_path, "dqn", "highvar", 0, hyper={"lr": 3e-4})
    runs, _ = analyze.find_runs(tmp_path)
    assert analyze.check_across_runs(runs)
    assert all(r.status == "invalid" for r in runs)


def test_smoke_and_other_dirs_are_skipped(tmp_path):
    make_run(tmp_path, "dqn", "lowvar", 0)
    (tmp_path / "dqn_lowvar_s0_smoke").mkdir()
    (tmp_path / "logs").mkdir()
    runs, skipped = analyze.find_runs(tmp_path)
    assert [r.run_id for r in runs] == ["dqn_lowvar_s0"]
    assert sorted(skipped) == ["dqn_lowvar_s0_smoke", "logs"]


def test_end_to_end_outputs(tmp_path, monkeypatch):
    runs_dir, out = tmp_path / "runs", tmp_path / "figures"
    for algo in analyze.ALGOS:
        for team in config.TEAMS:
            for seed in config.SEEDS:
                make_run(runs_dir, algo, team, seed, hyper={"algo": algo})
    monkeypatch.setattr("sys.argv", ["analyze.py", "--runs-dir", str(runs_dir), "--out", str(out)])
    analyze.main()
    for name in ("runs_index.csv", "summary.csv", "contrasts.csv", "curves_ppo.png", "curves_dqn.png",
                 "threshold_steps.png", "interaction_auc.png", "interaction_lift.png", "ep_len.png"):
        assert (out / name).exists(), name
    summary = pd.read_csv(out / "summary.csv")
    assert len(summary) == 12 + 8  # 12 runs + mean/std for 4 cells
    contrasts = pd.read_csv(out / "contrasts.csv")
    assert list(contrasts["contrast"]) == ["ppo", "dqn", "dqn_minus_ppo (RQ2 interaction)"]
