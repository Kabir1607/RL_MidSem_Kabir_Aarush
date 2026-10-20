"""Throughput benchmark for choosing config.TOTAL_STEPS (spec Section 5.2).

Runs --procs processes at once against the local Showdown server (as the real runs will), each with
its own account names, alternating lowvar/highvar, and reports decision steps per second.

  --mode random : random legal actions, environment cost only
  --mode ppo    : MaskablePPO.learn with config.PPO_KWARGS, i.e. environment + inference + updates

Example: python src/bench_env.py --procs 6 --mode ppo --steps 8192 --budget-hours 24
"""

import argparse
import multiprocessing as mp
import time
import traceback

import numpy as np

import config


def run(i, mode, steps, device, threads):
    import torch
    from env import make_env

    torch.set_num_threads(threads)
    team = config.TEAMS[i % 2]
    env = make_env(team, seed=i, algo="bch")
    t0 = time.perf_counter()
    if mode == "random":
        rng = np.random.default_rng(i)
        action_masks = env.get_wrapper_attr("action_masks")  # gymnasium 1.x wrappers don't forward attributes
        env.reset()
        for _ in range(steps):
            _, _, terminated, truncated, _ = env.step(rng.choice(np.flatnonzero(action_masks())))
            if terminated or truncated:
                env.reset()
    else:
        from sb3_contrib import MaskablePPO
        model = MaskablePPO("MlpPolicy", env, seed=i, device=device, verbose=0, **config.PPO_KWARGS)
        model.learn(steps)
    dt = time.perf_counter() - t0
    lengths = env.get_episode_lengths()
    env.close()
    return dict(proc=i, team=team, steps=steps, seconds=dt, battles=len(lengths),
                mean_len=float(np.mean(lengths)) if lengths else float("nan"))


def worker(i, mode, steps, device, threads, out):
    try:
        out.put(run(i, mode, steps, device, threads))
    except BaseException:
        out.put(dict(proc=i, error=traceback.format_exc()))
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--steps", type=int, default=4096, help="decision steps per process")
    ap.add_argument("--mode", choices=["random", "ppo"], default="random")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1, help="torch threads per process")
    ap.add_argument("--budget-hours", type=float, help="print the TOTAL_STEPS this throughput allows")
    ap.add_argument("--runs", type=int, default=len(config.TEAMS) * len(config.SEEDS), help="runs in the batch")
    args = ap.parse_args()

    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(i, args.mode, args.steps, args.device, args.threads, out))
             for i in range(args.procs)]
    t0 = time.perf_counter()
    for p in procs:
        p.start()
    results = sorted((out.get() for _ in procs), key=lambda r: r["proc"])
    errors = [r for r in results if "error" in r]
    if errors:
        for p in procs:
            p.terminate()
        raise SystemExit("\n".join(f"proc {r['proc']} failed:\n{r['error']}" for r in errors))
    for p in procs:
        p.join()
    wall = time.perf_counter() - t0

    for r in results:
        print(f"proc {r['proc']} {r['team']:7}: {r['steps']} steps in {r['seconds']:.1f}s = "
              f"{r['steps'] / r['seconds']:.1f} steps/s, {r['battles']} battles, {r['mean_len']:.1f} steps/battle")
    per_proc = np.mean([r["steps"] / r["seconds"] for r in results])
    print(f"\nmode={args.mode} procs={args.procs} device={args.device} threads={args.threads}: "
          f"{per_proc:.1f} steps/s per process, {args.procs * args.steps / wall:.1f} steps/s total ({wall:.0f}s wall)")
    if args.budget_hours:
        # Each run also spends time on evaluation: N_EVAL battles every EVAL_FREQ steps, at about the same speed.
        mean_len = np.nanmean([r["mean_len"] for r in results])
        eval_overhead = 1 + config.N_EVAL * mean_len / config.EVAL_FREQ
        waves = -(-args.runs // args.procs)
        steps = per_proc * args.budget_hours * 3600 / waves / eval_overhead
        print(f"{args.runs} runs in {waves} wave(s) of {args.procs}, eval overhead x{eval_overhead:.2f}: "
              f"about {steps:,.0f} training steps per run fit in {args.budget_hours} h")


if __name__ == "__main__":
    main()
