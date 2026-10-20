"""RunRecorder: everything a training run writes (spec Sections 6 and 7), independent of the algorithm.

PPO uses it through the SB3 callbacks in callbacks.py; the DQN arm calls it directly
(see src/DQN/DQN_Reference/reference.md, Section 6).
"""

import csv
import json
import platform
import random
import shutil
import subprocess
import time
from importlib import metadata
from pathlib import Path

import numpy as np

import config
from evaluate import SUMMARY_FIELDS, evaluate, final_evaluation, make_eval_env
from metrics import baseline_win_rate, run_metrics
from plotting import plot_run

EPISODE_FIELDS = ["step", "battle_idx", "opp_team", "outcome", "len_steps", "len_turns", "disc_shaping_sum",
                  "ep_return_shaped"]
EVAL_FIELDS = ["run_id", "algo", "team", "seed", "step", "battles_trained", *SUMMARY_FIELDS, "wall_time_s"]
FINAL_FIELDS = ["run_id", "algo", "team", "seed", "step", "opponent", *SUMMARY_FIELDS, "b_team", "rl_lift"]
PACKAGES = ["poke_env", "stable_baselines3", "sb3_contrib", "gymnasium", "torch", "numpy", "pandas"]


def seed_everything(seed):
    """Seed random, numpy and torch from the run seed."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config.TORCH_THREADS)


def git_state():
    def git(*args):
        return subprocess.run(["git", *args], cwd=config.ROOT, capture_output=True, text=True).stdout.strip()
    return dict(commit=git("rev-parse", "HEAD"), dirty=bool(git("status", "--porcelain", "--", "src", "Teams")))


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value if isinstance(value, (str, int, float, bool, type(None))) else repr(value)


class RunRecorder:
    """Owns runs/<run_id>/ and the evaluation env of one run.

    algo, team, seed: identify the run (run_id = config.run_id(algo, team, seed), plus _<suffix> if given).
    opponent: scripted opponent for the in-training evaluations ("random" only for smoke runs).
    overwrite: delete an existing non-empty run directory instead of refusing to start.
    """

    def __init__(self, algo, team, seed, suffix="", opponent="heuristic", overwrite=False):
        self.algo, self.team, self.seed = algo, team, seed
        self.run_id = config.run_id(algo, team, seed) + (f"_{suffix}" if suffix else "")
        self.dir = config.RUNS_DIR / self.run_id
        if self.dir.exists() and any(self.dir.iterdir()):
            if not overwrite:
                raise FileExistsError(f"{self.dir} already has results; pass overwrite=True (--overwrite) to replace them")
            shutil.rmtree(self.dir)
        (self.dir / "ckpt").mkdir(parents=True)
        self.b_team = baseline_win_rate(team)
        self.opponent = opponent
        self.eval_env = make_eval_env(team, seed, algo, opponent)
        self.battles = 0
        self.t0 = time.time()
        self._episodes_file = (self.dir / "train_episodes.csv").open("w", newline="")
        self._episodes = csv.DictWriter(self._episodes_file, EPISODE_FIELDS, extrasaction="ignore")
        self._episodes.writeheader()
        for name, fields in (("eval.csv", EVAL_FIELDS), ("final_eval.csv", FINAL_FIELDS)):
            with (self.dir / name).open("w", newline="") as f:
                csv.DictWriter(f, fields).writeheader()
        self._diagnostics = None

    # ---------- what the trainer calls ----------

    def write_run_config(self, hyperparameters, env=None, extra=None):
        """run_config.json: every config value, the algorithm's hyperparameters, git state, versions, accounts."""
        accounts = {"eval": [self.eval_env.env.env.agent1.username, self.eval_env.env.env.agent2.username]}
        if env is not None:
            poke = env.unwrapped.env
            accounts["train"] = [poke.agent1.username, poke.agent2.username]
        cfg = {k: v for k, v in vars(config).items() if k.isupper()}
        data = dict(run_id=self.run_id, algo=self.algo, team=self.team, seed=self.seed, eval_opponent=self.opponent,
                    b_team=self.b_team, hyperparameters=hyperparameters, config=cfg, extra=extra or {},
                    git=git_state(), accounts=accounts, python=platform.python_version(),
                    versions={p: metadata.version(p) for p in PACKAGES},
                    started=time.strftime("%Y-%m-%d %H:%M:%S"))
        (self.dir / "run_config.json").write_text(json.dumps(jsonable(data), indent=2))
        if data["git"]["dirty"]:
            print(f"WARNING: src/ or Teams/ has uncommitted changes; {self.run_id} can't be traced to a commit. "
                  "Commit before the main runs.", flush=True)

    def episode_end(self, step, battle_end):
        """Call at the end of every training battle with info["battle_end"]."""
        self.battles += 1
        self._episodes.writerow(dict(battle_end, step=step, battle_idx=self.battles))
        self._episodes_file.flush()

    def evaluate(self, step, policy, save_checkpoint=None):
        """In-training evaluation: N_EVAL stratified battles, then eval.csv row, checkpoint, metrics, plot.

        save_checkpoint(path_without_suffix) saves the model; the trainer picks the file extension.
        Returns the eval.csv row plus the running metrics (auc, threshold_step, ...).
        """
        summary = evaluate(policy, self.eval_env, config.N_EVAL, self.opponent)
        row = dict(run_id=self.run_id, algo=self.algo, team=self.team, seed=self.seed, step=step,
                   battles_trained=self.battles, **summary, wall_time_s=round(time.time() - self.t0, 1))
        with (self.dir / "eval.csv").open("a", newline="") as f:
            csv.DictWriter(f, EVAL_FIELDS).writerow(row)
        if save_checkpoint is not None:
            save_checkpoint(self.dir / "ckpt" / f"step_{step}")
        metrics = run_metrics(self._read_eval(), self.b_team)
        plot_run(self.dir)
        return dict(row, **metrics)

    def final(self, step, policy):
        """Final evaluation (spec Section 6.5) of the last policy; writes final_eval.csv."""
        rows = [dict(run_id=self.run_id, algo=self.algo, team=self.team, seed=self.seed, step=step, **r,
                     b_team=self.b_team, rl_lift=r["win_rate"] - self.b_team if r["opponent"] == "heuristic" else None)
                for r in final_evaluation(policy, self.eval_env)]
        with (self.dir / "final_eval.csv").open("a", newline="") as f:
            csv.DictWriter(f, FINAL_FIELDS).writerows(rows)
        return rows

    def log_scalars(self, step, values):
        """Optional algorithm diagnostics, long format: diagnostics.csv with step, key, value."""
        if self._diagnostics is None:
            self._diagnostics_file = (self.dir / "diagnostics.csv").open("w", newline="")
            self._diagnostics = csv.writer(self._diagnostics_file)
            self._diagnostics.writerow(["step", "key", "value"])
        self._diagnostics.writerows([step, k, v] for k, v in values.items())
        self._diagnostics_file.flush()

    def close(self):
        self._episodes_file.close()
        if self._diagnostics is not None:
            self._diagnostics_file.close()
        self.eval_env.close()

    def _read_eval(self):
        import pandas as pd
        return pd.read_csv(self.dir / "eval.csv")
