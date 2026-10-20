"""Train one DQN run using the same battles, observations and reward as PPO.

    python src/DQN/train_dqn.py --team lowvar --seed 0 --profile smoke
    python src/DQN/train_dqn.py --team highvar --seed 0

Main: 200k decisions against the heuristic. Smoke: 20k against MaxBasePower,
shorter warmup/exploration, a separate output directory, and no final evaluation.
All shared evaluation and CSV writing is delegated to RunRecorder.
"""

import argparse
from contextlib import ExitStack
import csv
from pathlib import Path
import sys

# Direct script execution starts with src/DQN on sys.path. Add src so the shared
# modules and the DQN package resolve identically to imports used in tests.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import config
from DQN.agent import DQNAgent, DQNSettings
from DQN.policy import save_checkpoint
from DQN.replay import ReplayBuffer
from encoder import OBS_SIZE
from env import make_env
from recorder import RunRecorder, seed_everything

Q_RETURN_FIELDS = ["step", "battle_idx", "opp_team", "start_step", "start_max_q", "start_action_q",
                   "discounted_return", "terminated", "truncated"]


def run_training(env, recorder, agent, replay, settings, team, seed, q_returns_file,
                 skip_final_eval=False):
    """The decision loop, separated from resource creation so it can be tested.

    The caller owns and closes the environment, recorder, and diagnostic file.
    Every counter here measures training decisions; evaluation uses a separate env.
    """
    action_masks = env.get_wrapper_attr("action_masks")
    q_returns = csv.DictWriter(q_returns_file, Q_RETURN_FIELDS)
    q_returns.writeheader()

    def save(path):
        save_checkpoint(path, agent.online, step, team, seed, settings)

    def evaluate():
        row = recorder.evaluate(step, agent.greedy, save)
        print(f"[eval] step {step:,}: {row['win_rate']:.3f} "
              f"[{row['wilson_low']:.3f}, {row['wilson_high']:.3f}]", flush=True)

    step = 0
    evaluate()  # Record an untrained policy, rather than starting the curve after learning.
    obs, _ = env.reset(seed=seed)
    first_decision = True
    episode_length = 0
    discounted_return = 0.0
    last_update = {}

    for step in range(1, settings.total_steps + 1):
        # Take ownership before step/reset: Gym environments may reuse arrays.
        current_obs = np.array(obs, dtype=np.float32, copy=True)
        mask = np.array(action_masks(), dtype=bool, copy=True)
        epsilon = settings.epsilon(step - 1)
        action = agent.act(current_obs, mask, epsilon)
        if first_decision:
            initial_q = agent.q_values(current_obs)
            start_step = step
            start_max_q = float(initial_q[mask].max())
            start_action_q = float(initial_q[action])
            first_decision = False

        next_obs, reward, terminated, truncated, info = env.step(action)
        next_mask = action_masks()
        # Store the pre-reset next state. Only termination turns off bootstrapping;
        # a time-limit truncation still has a meaningful next-state value.
        replay.add(current_obs, action, reward, next_obs, next_mask, terminated)
        discounted_return += settings.gamma ** episode_length * reward
        episode_length += 1

        if settings.should_update(step, len(replay)):
            last_update = agent.update(replay.sample(settings.batch_size))
        if step % settings.target_update_interval == 0:
            agent.sync_target()  # after the online update if schedules coincide

        if terminated or truncated:
            end = info["battle_end"]
            recorder.episode_end(step, end)
            q_returns.writerow(dict(
                step=step, battle_idx=recorder.battles, opp_team=end["opp_team"],
                start_step=start_step, start_max_q=start_max_q, start_action_q=start_action_q,
                discounted_return=discounted_return, terminated=terminated, truncated=truncated,
            ))
            q_returns_file.flush()
            first_decision = True
            episode_length = 0
            discounted_return = 0.0
            if step < settings.total_steps:
                next_obs, _ = env.reset()
        obs = np.array(next_obs, dtype=np.float32, copy=True)

        if step % 1_000 == 0:
            recorder.log_scalars(step, {**last_update, "train/epsilon": epsilon,
                                       "train/replay_size": len(replay),
                                       "train/updates": agent.update_count})
        if step % config.EVAL_FREQ == 0:
            evaluate()  # weights now include this decision's scheduled update

    if settings.profile == "main" and not skip_final_eval:
        for row in recorder.final(step, agent.greedy):
            print(f"[final] vs {row['opponent']}: {row['win_rate']:.3f}", flush=True)


def train(team, seed, settings, overwrite=False, skip_final_eval=False):
    """Construct resources, run training, and close initialized resources on exit."""
    if team not in config.TEAMS or seed not in config.SEEDS:
        raise ValueError("Team/seed must belong to the shared experiment grid")
    if settings.total_steps <= 0 or settings.total_steps % config.EVAL_FREQ:
        raise ValueError("Decision budget must be a positive multiple of EVAL_FREQ")
    run_id = config.run_id("dqn", team, seed)
    if settings.suffix:
        run_id += "_" + settings.suffix
    directory = config.RUNS_DIR / run_id
    # Refuse existing output before creating a server connection or allocating replay.
    if directory.exists() and any(directory.iterdir()) and not overwrite:
        raise FileExistsError(f"{directory} already has results; use --overwrite to replace them")

    seed_everything(seed)
    exploration_seed, replay_seed = np.random.SeedSequence(seed).spawn(2)
    with ExitStack() as resources:
        env = make_env(team, seed, algo="dqn", opponent=settings.opponent)
        resources.callback(env.close)
        if env.observation_space.shape != (OBS_SIZE,) or env.action_space.n != config.ACTION_SPACE_SIZE:
            raise ValueError("Environment dimensions do not match the shared encoder/action contract")
        agent = DQNAgent(OBS_SIZE, env.action_space.n, settings,
                         np.random.default_rng(exploration_seed))
        replay = ReplayBuffer(settings.buffer_size, OBS_SIZE, env.action_space.n,
                              np.random.default_rng(replay_seed))
        recorder = RunRecorder("dqn", team, seed, suffix=settings.suffix,
                               opponent=settings.opponent, overwrite=overwrite)
        resources.callback(recorder.close)
        recorder.write_run_config(settings.hyperparameters(), env=env,
                                  extra=dict(profile=settings.profile, skip_final_eval=skip_final_eval))
        q_returns_file = resources.enter_context((recorder.dir / "q_returns.csv").open("w", newline=""))
        run_training(env, recorder, agent, replay, settings, team, seed, q_returns_file,
                     skip_final_eval=skip_final_eval)
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--team", choices=config.TEAMS, required=True)
    parser.add_argument("--seed", type=int, choices=config.SEEDS, required=True)
    parser.add_argument("--profile", choices=("main", "smoke"), default="main")
    parser.add_argument("--overwrite", action="store_true", help="replace this run's existing output")
    parser.add_argument("--skip-final-eval", action="store_true", help="leave main run final evaluation incomplete")
    args = parser.parse_args()
    train(args.team, args.seed, DQNSettings.for_profile(args.profile), args.overwrite, args.skip_final_eval)


if __name__ == "__main__":
    main()
