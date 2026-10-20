"""Run the main runs of one algorithm (config.TEAMS x config.SEEDS), N at a time (spec Section 5.2).

Each run is its own process with its own log, runs/logs/<run_id>.log; a new run starts as soon as one
finishes. A failed run does not stop the others. Refuses to start from uncommitted code, so every
run_config.json points to a real commit.

  # detached, laptop kept awake (see NOTES.md):
  setsid nohup systemd-inhibit --what=sleep:idle:handle-lid-switch --who=pokemon-rl --why="RL runs" \\
      venv/bin/python src/run_batch.py --algo ppo --parallel 3 > runs/logs/batch_ppo.log 2>&1 &
  python src/run_batch.py --status          # progress of every run
  pkill -f train_ppo.py                     # stop everything (runs then have to start over)
"""

import argparse
import subprocess
import sys
import time

import pandas as pd

import config
from recorder import git_state

TRAINERS = {"ppo": config.ROOT / "src" / "train_ppo.py"}
JOBS = [(team, seed) for seed in config.SEEDS for team in config.TEAMS]  # lowvar s0, highvar s0, lowvar s1, ...
LOG_DIR = config.RUNS_DIR / "logs"
POLL_SECONDS = 30


def stamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def launch(algo, parallel, allow_dirty):
    git = git_state()
    if git["dirty"] and not allow_dirty:
        raise SystemExit("src/ or Teams/ has uncommitted changes: commit first (or pass --allow-dirty for tests)")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{stamp()} batch {algo}: {len(JOBS)} runs, {parallel} at a time, commit {git['commit'][:10]}", flush=True)
    pending, running, failed = list(JOBS), {}, []
    while pending or running:
        while pending and len(running) < parallel:
            team, seed = pending.pop(0)
            run_id = config.run_id(algo, team, seed)
            log = (LOG_DIR / f"{run_id}.log").open("w")
            cmd = [sys.executable, str(TRAINERS[algo]), "--team", team, "--seed", str(seed)]
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=config.ROOT)
            running[proc] = (run_id, log, time.time())
            print(f"{stamp()} started  {run_id} (pid {proc.pid})", flush=True)
        time.sleep(POLL_SECONDS)
        for proc in [p for p in running if p.poll() is not None]:
            run_id, log, t0 = running.pop(proc)
            log.close()
            hours = (time.time() - t0) / 3600
            print(f"{stamp()} finished {run_id}: exit {proc.returncode} after {hours:.2f} h", flush=True)
            if proc.returncode != 0:
                failed.append(run_id)
    print(f"{stamp()} batch done; failed: {failed or 'none'}", flush=True)
    return 1 if failed else 0


def status(algo):
    for team, seed in JOBS:
        run_id = config.run_id(algo, team, seed)
        run_dir = config.RUNS_DIR / run_id
        ev_path, final_path = run_dir / "eval.csv", run_dir / "final_eval.csv"
        if not ev_path.exists():
            print(f"{run_id:18} not started")
            continue
        ev = pd.read_csv(ev_path)
        final = pd.read_csv(final_path) if final_path.exists() else pd.DataFrame()
        if len(final):
            wr = final.loc[final["opponent"] == "heuristic", "win_rate"]
            print(f"{run_id:18} done: final win rate vs heuristic {wr.iloc[0]:.3f}")
            continue
        if ev.empty:
            print(f"{run_id:18} starting (no evaluation yet)")
            continue
        last = ev.iloc[-1]
        eta = (config.TOTAL_STEPS - last["step"]) * last["wall_time_s"] / last["step"] / 3600 if last["step"] else float("nan")
        print(f"{run_id:18} step {int(last['step']):>9,} / {config.TOTAL_STEPS:,}  eval win rate {last['win_rate']:.3f}  "
              f"[{last['wilson_low']:.3f}, {last['wilson_high']:.3f}]  {last['wall_time_s'] / 3600:.2f} h elapsed, "
              f"~{eta:.1f} h left")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algo", choices=list(TRAINERS), default="ppo")
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--status", action="store_true", help="print the progress of every run and exit")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    if args.status:
        status(args.algo)
    else:
        sys.exit(launch(args.algo, args.parallel, args.allow_dirty))


if __name__ == "__main__":
    main()
