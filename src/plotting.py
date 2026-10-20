"""Per-run learning-curve plot (spec Section 6.4), redrawn after every evaluation."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from metrics import baseline_win_rate, run_metrics  # noqa: E402

EV_COLUMN = "train/explained_variance"


def read_csv(path):
    return pd.read_csv(path) if path.exists() and path.stat().st_size > 0 else pd.DataFrame()


def plot_run(run_dir):
    """Write run_dir/learning_curve.png from eval.csv, train_episodes.csv and (if present) tb/progress.csv."""
    ev = read_csv(run_dir / "eval.csv")
    if ev.empty:
        return
    episodes = read_csv(run_dir / "train_episodes.csv")
    progress = read_csv(run_dir / "tb" / "progress.csv")
    b_team = baseline_win_rate(ev["team"].iloc[0])
    m = run_metrics(ev, b_team)

    fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(9, 7), height_ratios=[2, 1])
    top.fill_between(ev["step"], ev["wilson_low"], ev["wilson_high"], alpha=0.25, label="Wilson 95%")
    top.plot(ev["step"], ev["win_rate"], marker="o", ms=3, label="eval win rate")
    top.axhline(b_team, ls="--", color="grey", label=f"heuristic baseline ({b_team:.3f})")
    top.axhline(m["threshold"], ls=":", color="black", label=f"threshold ({m['threshold']:.3f})")
    thr = "not reached"
    if not np.isnan(m["threshold_step"]):
        top.axvline(m["threshold_step"], color="tab:red", lw=1, label="threshold step")
        thr = f"{m['threshold_step']:,.0f}"
    top.set_ylim(0, 1)
    top.set_ylabel("evaluation win rate")
    top.set_title(f"{ev['run_id'].iloc[0]}   AUC {m['auc']:.3f}   threshold step {thr}")
    top.legend(loc="lower right", fontsize=8)
    if len(episodes) >= 2:
        s, b = episodes["step"].to_numpy(float), episodes["battle_idx"].to_numpy(float)
        sec = top.secondary_xaxis("top", functions=(lambda x: np.interp(x, s, b), lambda x: np.interp(x, b, s)))
        ticks = MaxNLocator(6).tick_values(0, b[-1])
        sec.set_xticks(ticks[ticks <= b[-1]])  # ticks past the data would all map onto the last step
        sec.set_xlabel("battles trained")

    if len(episodes):
        rolling = episodes["len_steps"].rolling(config.EP_LEN_ROLLING_WINDOW, min_periods=1).mean()
        bottom.plot(episodes["step"], rolling, color="tab:green")
    bottom.set_ylabel(f"episode length (steps,\nrolling {config.EP_LEN_ROLLING_WINDOW})", color="tab:green")
    if EV_COLUMN in progress:
        ev_ax = bottom.twinx()
        p = progress.dropna(subset=[EV_COLUMN])
        ev_ax.plot(p["time/total_timesteps"], p[EV_COLUMN], color="tab:purple", lw=1)
        ev_ax.set_ylabel("explained variance", color="tab:purple")
    bottom.set_xlabel("decision steps")

    fig.tight_layout()
    out = run_dir / "learning_curve.png"
    tmp = run_dir / "learning_curve.tmp.png"
    fig.savefig(tmp, dpi=110)
    plt.close(fig)
    os.replace(tmp, out)  # a viewer never sees a half-written PNG
