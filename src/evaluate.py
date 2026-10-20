"""Evaluation protocol (spec Sections 6.1 and 6.5), shared by both algorithms.

A policy is any function policy(obs, mask) -> action that is deterministic (greedy over legal actions).
"""

import numpy as np

import config
from env import OPPONENTS, make_env
from metrics import wilson

FINAL_EVAL = (("heuristic", config.FINAL_EVAL_N_HEURISTIC), ("random", config.FINAL_EVAL_N_SANITY),
              ("maxbp", config.FINAL_EVAL_N_SANITY))
SUMMARY_FIELDS = ["eval_n", "wins", "losses", "ties", "win_rate", "wilson_low", "wilson_high",
                  *(f"wr_vs_{k}" for k in config.EVAL_OPP_ORDER), "mean_eval_len_steps", "mean_eval_len_turns"]


def make_eval_env(team, seed, algo, opponent="heuristic"):
    """The evaluation env: own Showdown accounts, stratified opponent teams, no Monitor."""
    return make_env(team, seed, role="eval", opponent=opponent, algo=algo,
                    opp_per_team=config.N_EVAL // len(config.EVAL_OPP_ORDER), monitor=False)


_opponents = {}


def set_opponent(env, opponent):
    """Swap the scripted opponent of an env from make_env(..., monitor=False)."""
    if opponent not in _opponents:
        _opponents[opponent] = OPPONENTS[opponent](battle_format=config.BATTLE_FORMAT, start_listening=False)
    env.env.opponent = _opponents[opponent]


def evaluate(policy, env, n, opponent="heuristic"):
    """Play n battles, n / 3 against each opponent team in a fixed order; return SUMMARY_FIELDS."""
    keys = config.EVAL_OPP_ORDER
    if n % len(keys):
        raise ValueError(f"n = {n} is not divisible by {len(keys)} opponent teams")
    set_opponent(env, opponent)
    env.opp_teams.restart(n // len(keys))
    ends = []
    for _ in range(n):
        obs, _ = env.reset()
        done = False
        while not done:
            obs, _, terminated, truncated, info = env.step(policy(obs, env.action_masks()))
            done = terminated or truncated
        ends.append(info["battle_end"])
    return summarise(ends)


def summarise(ends):
    keys = config.EVAL_OPP_ORDER
    n = len(ends)
    outcomes = np.array([e["outcome"] for e in ends])
    wins, losses = int((outcomes == 1).sum()), int((outcomes == -1).sum())
    by_opp = {k: [e["outcome"] == 1 for e in ends if e["opp_team"] == k] for k in keys}
    counts = {k: len(v) for k, v in by_opp.items()}
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"evaluation not stratified: {counts}")
    lo, hi = wilson(wins, n)
    return dict(
        eval_n=n, wins=wins, losses=losses, ties=n - wins - losses, win_rate=wins / n, wilson_low=lo, wilson_high=hi,
        **{f"wr_vs_{k}": float(np.mean(v)) for k, v in by_opp.items()},
        mean_eval_len_steps=float(np.mean([e["len_steps"] for e in ends])),
        mean_eval_len_turns=float(np.mean([e["len_turns"] for e in ends])),
    )


def final_evaluation(policy, env):
    """Spec Section 6.5: 1,020 battles vs the heuristic, 204 each vs RandomPlayer and MaxBasePowerPlayer."""
    return [dict(opponent=opponent, **evaluate(policy, env, n, opponent)) for opponent, n in FINAL_EVAL]
