"""Sample-efficiency and stability metrics (spec Section 6.3). Pure functions over evaluation results,
shared by the recorder, the plots and the aggregation script."""

import numpy as np
import pandas as pd

import config
from summarize_baselines import wilson  # same Wilson interval as the baseline table

__all__ = ["wilson", "baseline_win_rate", "auc", "threshold_step", "stability", "run_metrics"]


def baseline_win_rate(team):
    """b_team: the heuristic pilot's win rate for this team version vs the heuristic, pooled over opponents."""
    df = pd.read_csv(config.BASELINE_CSV)
    c = config.BASELINE_COLUMNS
    row = df[(df[c["team"]] == config.BASELINE_TEAM_KEY[team]) & (df[c["pilot"]] == config.BASELINE_PILOT)
             & (df[c["opponent"]] == config.BASELINE_OPPONENT)]
    if len(row) != 1:
        raise ValueError(f"expected one baseline row for {team}, found {len(row)}")
    return float(row[c["win_rate"]].iloc[0])


def auc(steps, win_rates):
    """Trapezoidal area under (step, win rate) from 0 to the last step, divided by the last step."""
    steps, win_rates = np.asarray(steps, float), np.asarray(win_rates, float)
    if len(steps) < 2 or steps[-1] <= 0:
        return float("nan")
    return float(np.trapezoid(win_rates, steps) / steps[-1])


def threshold_step(steps, win_rates, threshold, consecutive=config.THRESHOLD_CONSECUTIVE):
    """First step whose win rate and the next (consecutive - 1) are all >= threshold; NaN if never."""
    win_rates = np.asarray(win_rates, float)
    for i in range(len(win_rates) - consecutive + 1):
        if np.all(win_rates[i:i + consecutive] >= threshold):
            return float(steps[i])
    return float("nan")


def stability(win_rates, last_k=config.STABILITY_LAST_K):
    """(largest drop between consecutive evaluations, std of the last k evaluations)."""
    win_rates = np.asarray(win_rates, float)
    drops = win_rates[:-1] - win_rates[1:]
    largest_drop = float(max(drops.max(), 0.0)) if len(drops) else float("nan")
    tail = win_rates[-last_k:]
    std_last = float(np.std(tail, ddof=1)) if len(tail) >= 2 else float("nan")
    return largest_drop, std_last


def run_metrics(eval_df, b_team):
    """All per-run metrics from an eval.csv dataframe."""
    steps, wr = eval_df["step"].to_numpy(), eval_df["win_rate"].to_numpy()
    largest_drop, std_last = stability(wr)
    threshold = b_team + config.THRESHOLD_MARGIN
    return dict(auc=auc(steps, wr), threshold=threshold, threshold_step=threshold_step(steps, wr, threshold),
                largest_drop=largest_drop, std_last=std_last)
