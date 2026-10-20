"""Aggregate all runs into figures and tables (spec Section 6.6).

Reads runs/<algo>_<team>_s<seed>/ for algo in {ppo, dqn}, validates every run against the output contract
(src/DQN/DQN_Reference/reference.md, Section 7), and writes to figures/:

  runs_index.csv          every run found: status (complete / in progress / invalid), problems, commit
  summary.csv             one row per run, plus mean and std rows per (algo, team) cell
  contrasts.csv           per algorithm: HighVar - LowVar for AUC, threshold steps and lift; DQN - PPO interaction
  curves_<algo>.png       LowVar vs HighVar learning curves, every seed plus the mean, baselines dashed (RQ1)
  threshold_steps.png     steps to b_team + 0.10 per seed; "not reached" drawn at the top (RQ1)
  interaction_auc.png     AUC at the budget: team version x algorithm, every seed (RQ2)
  interaction_lift.png    final win rate - b_team: team version x algorithm, every seed (RQ3)
  ep_len.png              training episode length per cell, every seed

  python src/analyze.py              # complete runs only
  python src/analyze.py --partial    # also runs still training (metrics up to their last evaluation)
"""

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from evaluate import FINAL_EVAL  # noqa: E402
from metrics import baseline_win_rate, run_metrics  # noqa: E402
from recorder import EPISODE_FIELDS, EVAL_FIELDS, FINAL_FIELDS  # noqa: E402

ALGOS = ("ppo", "dqn")
RUN_DIR_RE = re.compile(r"^(?P<algo>[a-z]+)_(?P<team>[a-z]+)_s(?P<seed>\d+)$")
# config values that must be identical in every run of both algorithms (spec Sections 2 and 5.4)
FAIRNESS_KEYS = ("BATTLE_FORMAT", "N_TEAM", "GAMMA", "MAX_EPISODE_STEPS", "NET_ARCH", "TOTAL_STEPS", "EVAL_FREQ",
                 "N_EVAL", "EVAL_OPP_ORDER", "FINAL_EVAL_N_HEURISTIC", "FINAL_EVAL_N_SANITY", "THRESHOLD_MARGIN",
                 "THRESHOLD_CONSECUTIVE")
SHAPING_TOL = 1e-6

# Reference palette (dataviz skill), validated: team pair passes all checks; algorithm pair passes with
# direct labels (aqua < 3:1 contrast). The two pairs never share a figure (violet ~ blue).
TEAM_COLOR = {"lowvar": "#2a78d6", "highvar": "#eb6834"}
ALGO_COLOR = {"ppo": "#1baf7a", "dqn": "#6250d6"}
TEAM_LABEL = {"lowvar": "LowVar", "highvar": "HighVar"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


@dataclass
class Run:
    algo: str
    team: str
    seed: int
    dir: Path
    errors: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    cfg: dict = field(default_factory=dict)
    eval: pd.DataFrame = None
    episodes: pd.DataFrame = None
    final: pd.DataFrame = None

    @property
    def run_id(self):
        return self.dir.name

    @property
    def last_step(self):
        return int(self.eval["step"].iloc[-1]) if self.eval is not None and len(self.eval) else -1

    @property
    def complete(self):
        return not self.errors and self.last_step == config.TOTAL_STEPS and self.final is not None and len(self.final)

    @property
    def status(self):
        return "invalid" if self.errors else "complete" if self.complete else "in progress"


# ---------- loading and validation (the output contract) ----------

def read_table(run, name, fields):
    path = run.dir / name
    if not path.exists():
        run.errors.append(f"missing {name}")
        return None
    df = pd.read_csv(path)
    if list(df.columns) != fields:
        run.errors.append(f"{name}: columns {list(df.columns)} != contract {fields}")
        return None
    return df


def load_run(path):
    m = RUN_DIR_RE.match(path.name)
    run = Run(m["algo"], m["team"], int(m["seed"]), path)

    cfg_path = path / "run_config.json"
    if not cfg_path.exists():
        run.errors.append("missing run_config.json")
    else:
        run.cfg = json.loads(cfg_path.read_text())
        if (run.cfg.get("algo"), run.cfg.get("team"), run.cfg.get("seed")) != (run.algo, run.team, run.seed):
            run.errors.append("run_config.json algo/team/seed do not match the directory name")
        for key in FAIRNESS_KEYS:
            ours, theirs = getattr(config, key), run.cfg.get("config", {}).get(key)
            if json.loads(json.dumps(ours)) != theirs:
                run.errors.append(f"config {key} = {theirs}, current config has {ours}")
        if run.cfg.get("eval_opponent") != "heuristic":
            run.errors.append(f"evaluated against {run.cfg.get('eval_opponent')}, not heuristic")
        if run.cfg.get("git", {}).get("dirty"):
            run.notes.append("started from uncommitted code")

    run.eval = read_table(run, "eval.csv", EVAL_FIELDS)
    if run.eval is not None:
        ev = run.eval
        ids = ev[["run_id", "algo", "team", "seed"]].drop_duplicates().values.tolist()
        if ids != [[run.run_id, run.algo, run.team, run.seed]]:
            run.errors.append(f"eval.csv identifies the run as {ids}")
        expected = np.arange(len(ev)) * config.EVAL_FREQ
        if not np.array_equal(ev["step"].to_numpy(), expected):
            run.errors.append("eval.csv steps are not 0, EVAL_FREQ, 2*EVAL_FREQ, ...")
        if (ev["eval_n"] != config.N_EVAL).any() or (ev["wins"] + ev["losses"] + ev["ties"] != ev["eval_n"]).any():
            run.errors.append(f"eval.csv: every evaluation must have {config.N_EVAL} battles")
        if len(ev) == 0:
            run.errors.append("eval.csv is empty")

    run.episodes = read_table(run, "train_episodes.csv", EPISODE_FIELDS)
    if run.episodes is not None and len(run.episodes):
        ep = run.episodes
        if (ep["step"].diff().dropna() < 0).any() or not np.array_equal(ep["battle_idx"], np.arange(1, len(ep) + 1)):
            run.errors.append("train_episodes.csv: steps must be non-decreasing and battle_idx 1, 2, 3, ...")
        if ep["step"].max() > config.TOTAL_STEPS:
            run.errors.append(f"trained past TOTAL_STEPS ({ep['step'].max()})")
        n_bad = int((ep["disc_shaping_sum"].abs() > SHAPING_TOL).sum())
        if n_bad:
            run.notes.append(f"{n_bad} battles with discounted shaping sum != 0 (truncated battles?)")

    final = read_table(run, "final_eval.csv", FINAL_FIELDS)
    if final is not None and len(final):
        want = {opp: n for opp, n in FINAL_EVAL}
        got = dict(zip(final["opponent"], final["eval_n"]))
        if got != want or (final["step"] != config.TOTAL_STEPS).any():
            run.errors.append(f"final_eval.csv: expected {want} at step {config.TOTAL_STEPS}, got {got}")
        else:
            run.final = final
    if run.last_step == config.TOTAL_STEPS and run.final is None and not run.errors:
        run.notes.append("training finished but final evaluation missing")
    if run.last_step == config.TOTAL_STEPS and not list((path / "ckpt").glob(f"step_{config.TOTAL_STEPS}.*")):
        run.notes.append("final checkpoint missing (needed for the move-usage replay)")
    return run


def find_runs(runs_dir):
    runs, skipped = [], []
    for path in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        m = RUN_DIR_RE.match(path.name)
        if m and m["algo"] in ALGOS and m["team"] in config.TEAMS and int(m["seed"]) in config.SEEDS:
            runs.append(load_run(path))
        else:
            skipped.append(path.name)
    return runs, skipped


def check_across_runs(runs):
    """Within an algorithm, all runs must share hyperparameters (and should share a commit)."""
    problems = []
    for algo in ALGOS:
        rs = [r for r in runs if r.algo == algo and r.cfg]
        hps = {json.dumps(r.cfg.get("hyperparameters"), sort_keys=True) for r in rs}
        if len(hps) > 1:
            problems.append(f"{algo}: hyperparameters differ between runs")
            for r in rs:
                r.errors.append("hyperparameters differ from other runs of the same algorithm")
        commits = {r.cfg.get("git", {}).get("commit") for r in rs}
        if len(commits) > 1:
            problems.append(f"{algo}: runs come from {len(commits)} different commits")
    return problems


# ---------- tables ----------

def run_row(run):
    b = baseline_win_rate(run.team)
    m = run_metrics(run.eval, b)
    row = dict(algo=run.algo, team=run.team, seed=str(run.seed), status=run.status, last_step=run.last_step,
               battles_trained=int(run.eval["battles_trained"].iloc[-1]), b_team=b, auc=m["auc"],
               threshold_step=m["threshold_step"], largest_drop=m["largest_drop"], std_last5=m["std_last"])
    heur = run.final[run.final["opponent"] == "heuristic"].iloc[0] if run.final is not None else None
    row.update(
        final_win_rate=heur["win_rate"] if heur is not None else np.nan,
        final_wilson_low=heur["wilson_low"] if heur is not None else np.nan,
        final_wilson_high=heur["wilson_high"] if heur is not None else np.nan,
        rl_lift=heur["win_rate"] - b if heur is not None else np.nan,
        **{f"final_wr_vs_{k}": heur[f"wr_vs_{k}"] if heur is not None else np.nan for k in config.EVAL_OPP_ORDER},
        **{f"final_wr_{o}": (run.final.loc[run.final["opponent"] == o, "win_rate"].iloc[0]
                             if run.final is not None else np.nan) for o in ("random", "maxbp")},
        mean_len_steps_last=float(run.episodes["len_steps"].tail(config.EP_LEN_ROLLING_WINDOW).mean())
        if run.episodes is not None and len(run.episodes) else np.nan,
    )
    return row


def summarise(runs):
    per_run = pd.DataFrame([run_row(r) for r in runs])
    numeric = [c for c in per_run.columns if c not in ("algo", "team", "seed", "status")]
    cells = []
    for (algo, team), g in per_run.groupby(["algo", "team"], sort=False):
        for stat in ("mean", "std"):
            vals = getattr(g[numeric], stat)() if stat == "mean" else g[numeric].std(ddof=1)
            cells.append(dict(algo=algo, team=team, seed=stat, status=f"{len(g)} seeds", **vals.to_dict()))
        cells[-2]["threshold_reached"] = f"{int(g['threshold_step'].notna().sum())}/{len(g)}"
    return per_run, pd.concat([per_run, pd.DataFrame(cells)], ignore_index=True)


def contrasts(per_run):
    rows = []
    for algo in ALGOS:
        g = per_run[per_run["algo"] == algo]
        means = g.groupby("team")[["auc", "threshold_step", "rl_lift", "final_win_rate"]].mean()
        if not {"lowvar", "highvar"} <= set(means.index):
            continue
        low, high = means.loc["lowvar"], means.loc["highvar"]
        rows.append(dict(contrast=algo, auc_lowvar=low["auc"], auc_highvar=high["auc"],
                         delta_auc=high["auc"] - low["auc"], delta_threshold_step=high["threshold_step"] - low["threshold_step"],
                         lift_lowvar=low["rl_lift"], lift_highvar=high["rl_lift"],
                         lift_difference=high["rl_lift"] - low["rl_lift"],
                         seed_std_final_wr_lowvar=g.loc[g["team"] == "lowvar", "final_win_rate"].std(ddof=1),
                         seed_std_final_wr_highvar=g.loc[g["team"] == "highvar", "final_win_rate"].std(ddof=1)))
    out = pd.DataFrame(rows)
    if {"ppo", "dqn"} <= set(out.get("contrast", [])):
        p, d = out.set_index("contrast").loc["ppo"], out.set_index("contrast").loc["dqn"]
        diff = {k: d[k] - p[k] for k in ("delta_auc", "delta_threshold_step", "lift_difference")}
        out = pd.concat([out, pd.DataFrame([dict(contrast="dqn_minus_ppo (RQ2 interaction)", **diff)])], ignore_index=True)
    return out


# ---------- figures ----------

def style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 160, "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "axes.titlesize": 11, "axes.titlecolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2,
    })


def save(fig, path):
    tmp = path.with_suffix(".tmp.png")
    fig.savefig(tmp, bbox_inches="tight")
    plt.close(fig)
    tmp.replace(path)


def mean_curve(curves):
    """Mean win rate over seeds, only at steps every seed has reached."""
    wide = pd.concat({seed: c.set_index("step")["win_rate"] for seed, c in curves.items()}, axis=1).dropna()
    return wide.index.to_numpy(), wide.mean(axis=1).to_numpy()


def end_labels(ax, labels, min_gap=0.06):
    """Direct labels at line ends, nudged apart vertically so they never overlap."""
    labels = sorted(labels, key=lambda l: l[2])
    ys = [y for _, _, y in labels]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + min_gap)
    for (text, x, _), y in zip(labels, ys):
        ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points", color=INK, va="center", fontsize=9,
                    annotation_clip=False)


def plot_curves(runs, algo, out):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    labels = []
    for team in config.TEAMS:
        rs = [r for r in runs if r.algo == algo and r.team == team]
        if not rs:
            continue
        color = TEAM_COLOR[team]
        for r in rs:
            ax.plot(r.eval["step"], r.eval["win_rate"], color=color, lw=1, alpha=0.45)
        steps, mean = mean_curve({r.seed: r.eval for r in rs})
        ax.plot(steps, mean, color=color, lw=2.2, label=f"{TEAM_LABEL[team]} mean of {len(rs)} seeds (thin: each seed)")
        if len(steps):
            labels.append((TEAM_LABEL[team], steps[-1], mean[-1]))
        b = baseline_win_rate(team)
        ax.axhline(b, color=color, ls="--", lw=1.2, label=f"{TEAM_LABEL[team]} heuristic baseline ({b:.3f})")
    end_labels(ax, labels)
    ax.set(xlim=(0, config.TOTAL_STEPS * 1.06), ylim=(0, 1), xlabel="decision steps",
           ylabel="evaluation win rate vs heuristic (120 battles)",
           title=f"{algo.upper()}: LowVar vs HighVar learning curves")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x / 1e6:g}M" if x else "0")
    ax.legend(loc="lower right", fontsize=8)
    save(fig, out / f"curves_{algo}.png")


def plot_interaction(per_run, value, ylabel, title, out_name, out, ref=None):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    x = {team: i for i, team in enumerate(config.TEAMS)}
    for k, algo in enumerate(ALGOS):
        g = per_run[per_run["algo"] == algo].dropna(subset=[value])
        if g.empty:
            continue
        color = ALGO_COLOR[algo]
        offset = (k - 0.5) * 0.08
        ax.scatter([x[t] + offset for t in g["team"]], g[value], s=40, color=color, edgecolor="white", lw=1.5,
                   zorder=3, label=f"{algo.upper()} (dots: seeds, line: mean)")
        means = g.groupby("team")[value].mean().reindex(config.TEAMS)
        ax.plot([x[t] + offset for t in config.TEAMS], means.to_numpy(), color=color, lw=2, zorder=2)
        last = means.dropna()
        if len(last):
            ax.annotate(algo.upper(), (x[last.index[-1]] + offset, last.iloc[-1]), xytext=(8, 0),
                        textcoords="offset points", color=INK, va="center", fontsize=9)
    if ref is not None:
        ax.axhline(ref, color=MUTED, lw=1, ls=":")
    ax.set_xticks(list(x.values()), [TEAM_LABEL[t] for t in config.TEAMS])
    ax.set(xlim=(-0.4, len(x) - 0.4), ylabel=ylabel, title=title)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best", fontsize=8)
    else:
        ax.text(0.5, 0.5, "no complete runs yet", transform=ax.transAxes, ha="center", color=MUTED)
    save(fig, out / out_name)


def plot_thresholds(per_run, out):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cells = [(a, t) for a in ALGOS for t in config.TEAMS]
    top = config.TOTAL_STEPS * 1.12
    for i, (algo, team) in enumerate(cells):
        g = per_run[(per_run["algo"] == algo) & (per_run["team"] == team)]
        if g.empty:
            continue
        reached = g["threshold_step"].dropna()
        missed = len(g) - len(reached)
        jitter = np.linspace(-0.08, 0.08, len(g))
        ax.scatter(i + jitter[:len(reached)], reached, s=40, color=TEAM_COLOR[team], edgecolor="white", lw=1.5, zorder=3)
        if missed:
            ax.scatter(i + jitter[len(reached):], [top] * missed, s=50, marker="x", color=TEAM_COLOR[team], zorder=3)
    ax.axhline(config.TOTAL_STEPS, color=MUTED, ls="--", lw=1)
    ax.annotate("budget", (len(cells) - 0.5, config.TOTAL_STEPS), xytext=(0, 3), textcoords="offset points",
                color=MUTED, fontsize=8, ha="right")
    ax.annotate("× = not reached", (len(cells) - 0.5, top), xytext=(0, 6), textcoords="offset points",
                color=MUTED, fontsize=8, ha="right")
    ax.set_xticks(range(len(cells)), [f"{a.upper()}\n{TEAM_LABEL[t]}" for a, t in cells])
    ax.set(xlim=(-0.5, len(cells) - 0.5), ylim=(0, top * 1.06), ylabel="decision steps to threshold",
           title=f"Steps to heuristic baseline + {config.THRESHOLD_MARGIN:.2f} (2 evaluations in a row)")
    ax.yaxis.set_major_formatter(lambda y, _: f"{y / 1e6:g}M" if y else "0")
    save(fig, out / "threshold_steps.png")


def plot_ep_len(runs, out):
    algos = [a for a in ALGOS if any(r.algo == a for r in runs)]
    fig, axes = plt.subplots(1, len(algos), figsize=(6 * len(algos), 4), sharey=True, squeeze=False)
    for ax, algo in zip(axes[0], algos):
        for team in config.TEAMS:
            for j, r in enumerate(r for r in runs if r.algo == algo and r.team == team):
                ep = r.episodes
                if ep is None or not len(ep):
                    continue
                rolling = ep["len_steps"].rolling(config.EP_LEN_ROLLING_WINDOW, min_periods=1).mean()
                ax.plot(ep["step"], rolling, color=TEAM_COLOR[team], lw=1.2, alpha=0.8,
                        label=TEAM_LABEL[team] if j == 0 else None)
        ax.set(title=algo.upper(), xlabel="decision steps")
        ax.xaxis.set_major_formatter(lambda x, _: f"{x / 1e6:g}M" if x else "0")
        ax.legend(fontsize=8)
    axes[0][0].set_ylabel(f"training battle length (steps, rolling {config.EP_LEN_ROLLING_WINDOW} battles)")
    save(fig, out / "ep_len.png")


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", type=Path, default=config.RUNS_DIR)
    ap.add_argument("--out", type=Path, default=config.FIGURES_DIR)
    ap.add_argument("--partial", action="store_true", help="include runs still in progress")
    args = ap.parse_args()

    runs, skipped = find_runs(args.runs_dir)
    across = check_across_runs(runs)
    args.out.mkdir(parents=True, exist_ok=True)
    index = pd.DataFrame([dict(run_id=r.run_id, algo=r.algo, team=r.team, seed=r.seed, status=r.status,
                               last_step=r.last_step, commit=r.cfg.get("git", {}).get("commit", "")[:10],
                               errors="; ".join(r.errors), notes="; ".join(r.notes)) for r in runs])
    index.to_csv(args.out / "runs_index.csv", index=False)
    print(index.drop(columns=["errors", "notes"]).to_string(index=False) if len(index) else "no runs found")
    for r in runs:
        for e in r.errors:
            print(f"  ERROR {r.run_id}: {e}")
        for n in r.notes:
            print(f"  note  {r.run_id}: {n}")
    for p in across:
        print(f"  ACROSS RUNS: {p}")
    if skipped:
        print(f"  skipped (not main runs): {', '.join(skipped)}")

    use = [r for r in runs if r.status == "complete" or (args.partial and r.status == "in progress" and len(r.eval))]
    if not use:
        print("nothing to analyse" + ("" if args.partial else " (use --partial for runs still training)"))
        return
    per_run, summary = summarise(use)
    summary.to_csv(args.out / "summary.csv", index=False, float_format="%.4f")
    contrast = contrasts(per_run)
    contrast.to_csv(args.out / "contrasts.csv", index=False, float_format="%.4f")

    style()
    for algo in ALGOS:
        if any(r.algo == algo for r in use):
            plot_curves(use, algo, args.out)
    plot_thresholds(per_run, args.out)
    plot_interaction(per_run, "auc", f"AUC: mean eval win rate over {config.TOTAL_STEPS / 1e6:g}M steps",
                     "Sample efficiency (RQ2)", "interaction_auc.png", args.out)
    plot_interaction(per_run, "rl_lift", "final win rate − heuristic baseline", "RL lift (RQ3)",
                     "interaction_lift.png", args.out, ref=0)
    plot_ep_len(use, args.out)

    cols = ["algo", "team", "seed", "status", "last_step", "auc", "threshold_step", "final_win_rate", "rl_lift"]
    print("\n" + summary[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if len(contrast):
        print("\n" + contrast.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if args.partial and any(r.status == "in progress" for r in use):
        print("\nPARTIAL: some runs are still training; their AUC and thresholds cover only the steps so far.")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
