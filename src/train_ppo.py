"""Train one PPO run (spec Section 5) with in-training evaluation, then the final evaluation.

  python src/train_ppo.py --team lowvar --seed 0
  python src/train_ppo.py --team lowvar --seed 0 --steps 20000 --opponent random --suffix smoke   # smoke test

Needs the local Showdown server. Writes runs/ppo_<team>_s<seed>[_<suffix>]/.
"""

import argparse

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.logger import configure

import config
from callbacks import EvalCurveCallback, TrainEpisodeLogger, masked_greedy
from env import OPPONENTS, make_env
from recorder import RunRecorder, seed_everything


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--team", choices=config.TEAMS, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=config.TOTAL_STEPS, help="decision steps (default: config.TOTAL_STEPS)")
    ap.add_argument("--opponent", choices=list(OPPONENTS), default="heuristic",
                    help="training and in-training evaluation opponent (not heuristic only for smoke tests)")
    ap.add_argument("--suffix", default="", help="appended to the run id, e.g. smoke")
    ap.add_argument("--overwrite", action="store_true", help="delete an existing run directory first")
    ap.add_argument("--skip-final-eval", action="store_true")
    args = ap.parse_args()

    seed_everything(args.seed)
    rec = RunRecorder("ppo", args.team, args.seed, args.suffix, args.opponent, args.overwrite)
    env = make_env(args.team, args.seed, opponent=args.opponent, algo="ppo")
    model = MaskablePPO("MlpPolicy", env, seed=args.seed, device=config.DEVICE, verbose=1, **config.PPO_KWARGS)
    model.set_logger(configure(str(rec.dir / "tb"), ["stdout", "csv", "tensorboard"]))
    rec.write_run_config(hyperparameters=config.PPO_KWARGS, env=env, extra=vars(args))

    callbacks = CallbackList([TrainEpisodeLogger(rec), EvalCurveCallback(rec, args.steps, verbose=1)])
    model.learn(total_timesteps=args.steps, callback=callbacks)  # the callback stops at exactly args.steps
    if not args.skip_final_eval:
        for row in rec.final(args.steps, masked_greedy(model)):
            print(f"[final] vs {row['opponent']}: {row['wins']}/{row['eval_n']} = {row['win_rate']:.3f}")
    rec.close()
    env.close()


if __name__ == "__main__":
    main()
