# Implementation notes (spec Section 3.1 VERIFY items)

Checked against the installed versions on 2026-10-20: Python 3.14.8, poke-env 0.16.1, stable-baselines3 2.9.0,
sb3-contrib 2.9.0, gymnasium 1.3.0. Paths below are relative to `venv/lib/python3.14/site-packages/poke_env/`.
The spec asks for Python 3.11; the venv is 3.14, and the real versions go into each `run_config.json`.

## Battle format

The baselines are **singles**: `gen9ounotera` = [Gen 9] OU + Terastal Clause
(`pokemon-showdown/config/custom-formats.ts`). Team preview exists; `src/baselines.py` brings all 4 in file
order (`/team 1234`). Spec Appendix A (doubles) does not apply.

## VERIFY answers

1. **Observation from `SingleAgentWrapper.step/reset`.** A dict `{"observation": embed_battle(battle),
   "action_mask": int8 array of length 26}` (`environment/env.py`, `PokeEnv.step`/`reset`;
   `observation_spaces` is wrapped into a `spaces.Dict` by `PokeEnv.__setattr__`). SB3 needs a Box, so our
   wrapper returns `obs["observation"]` and keeps the mask for `action_masks()`.

2. **Current battle.** `SingleAgentWrapper.env.battle1` (our side), `.battle2` (opponent side).

3. **`calc_reward` calls.** Once per `PokeEnv.step`, for **both** `battle1` and `battle2`; never on reset.
   Both battle objects have the **same `battle_tag`** (it is the Showdown room id), so the spec's reference
   `calc_reward` keyed by `battle_tag` would mix our shaping with the opponent's.
   **Decision:** use the spec's fallback. `PPOSinglesEnv.calc_reward` returns 0; the shaped reward is
   computed in a Gymnasium wrapper from `battle1` only, exactly once per agent `step()`.

4. **`battle.opponent_team`.** Returns `_opponent_team` (revealed Pokémon only) once anything is revealed;
   it falls back to team preview only while `_opponent_team` is empty (`battle/abstract_battle.py`,
   `opponent_team`). `battle.teampreview_opponent_team` always holds all 4 species.
   **Decision:** OTS rows and Φ key the opponent by species from team preview; unrevealed = full HP, alive.

5. **Masks in the official example** (`examples/reinforcement_learning.py`, poke-env master, commit
   cd167a3, 2026-03-23). It uses plain `PPO` with a custom `ActorCriticPolicy` that reads
   `obs["action_mask"]` in both `forward` and `evaluate_actions` and adds −inf to illegal logits.
   **Decision:** the spec requires `MaskablePPO`, so our wrapper flattens the observation and exposes
   `action_masks()`. sb3-contrib finds it through `env.get_wrapper_attr("action_masks")` /
   `VecEnv.env_method`, so it works through `Monitor`. `MaskablePPO.predict(obs, action_masks=...)` exists.
   (The example also uses `reward_computing_helper`, which the spec forbids.)

6. **Heuristic opponent.** `from poke_env.player import SimpleHeuristicsPlayer`
   (`player/baselines.py`). It is deterministic except when its or our active Pokémon is `None`
   (falls back to `choose_random_singles_move`). Construct it with `start_listening=False`: in
   `SingleAgentWrapper` only its `choose_move` is called, and the env's own `agent2` plays on Showdown.

**Stats utility.** `poke_env.stats.compute_raw_stats(species, evs, ivs, level, nature, GenData)` returns
`[hp, atk, def, spa, spd, spe]` with the standard Gen 3+ formulas. Used in `teamsheet.py`. Test 2 checks it
against Showdown's own simulator (`tests/showdown_stats.js` runs `pokemon-showdown/dist/sim` in Node, no
server needed) for all 20 team slots; they match.

**MaskablePPO keywords.** All names in spec Section 5.2 exist in sb3-contrib 2.9.0. Note the defaults differ
from the spec for `batch_size` (64) and `ent_coef` (0.0), so both must be passed explicitly (they are, in
`config.PPO_KWARGS`).

**Showdown usernames.** The full display name, underscores included, must be ≤ 18 characters: longer names
are cut to 18 by `Chat.namefilter` (`pokemon-showdown/server/chat.ts`), and poke-env then fails its login
check ("Expected <name> to be logged in"). Format: `ppo_lv0_evopp_3f9a` (see item 9 for the tag).

## Other things the spec does not cover

1. **Team preview is random.** `_EnvPlayer._teampreview` calls `random_teampreview` for singles
   (`environment/env.py`). Override `teampreview` on both `env.agent1` and `env.agent2` to return
   `/team 1234` and mark all mons selected, like `FixedLeadMixin` in `src/baselines.py`.

2. **One team for both env players.** `SinglesEnv(team=...)` passes the same team to `agent1` and `agent2`.
   After construction, call `env.agent2.update_team(<opponent teambuilder>)`. `agent2` accepts the
   challenge with its own `next_team`, so a per-battle teambuilder works.

3. **Wait steps.** When we get a "wait" request (only the opponent must act, e.g. its forced switch),
   `PokeEnv.step` still returns to the agent: `battle1` is stale, `battle._wait` is True, the mask is `[0]`,
   and the next action is ignored (`agent1_to_move` is False). These are not decision requests (spec 4.1).
   **Decision:** the wrapper steps through them internally while `agent1_to_move` is False. Confirmed in
   Phase 2: 0–3 such steps per battle, and `tests/test_env_live.py` checks none reach the agent.

4. **Ties come back as `truncated`.** `PokeEnv.calc_term_trunc` sets `truncated=True` when a finished battle
   doesn't have exactly one side wiped (ties, forfeits). **Decision:** the wrapper sets
   `terminated = battle.finished` and uses `truncated` only for the `MAX_EPISODE_STEPS` cap.

5. **Action index ↔ Pokémon/move.** Actions 0–5 index `list(battle.team.values())`; actions 6–9 index
   `list(battle.active_pokemon.moves.values())[:4]` (`environment/singles_env.py`). The encoder's move
   slots for the active Pokémon must follow `active.moves` order. Phase 2 tests assert both orders match the
   team file.

6. **Observation row size.** The spec's table gives 70 + (I + 1) + B per row, not 70 + I + B. With the
   current files, I = 6 items and B = 14 abilities, so a row is 91. Computed from the files at startup.

7. **Swap moves.** The spec's table lists Thunderbolt→Thunder, but the files swap Flamethrower→Fire Blast and
   Air Slash→Hurricane (Moltres), Surf→Hydro Pump and Ice Beam→Blizzard (Primarina). The files are the
   source of truth; `check_teams.py` derives the swaps from them.

8. **Heuristic baseline** (`baselines_summary.csv`, pilot `heuristic`, opponent `all`):
   LowVar 0.4287, HighVar 0.4373; threshold = b_team + 0.10. Per opponent it is very uneven
   (B ≈ 1.0, A ≈ 0.05–0.08, C ≈ 0.24–0.25), which is why evaluation is stratified.
   Mean battle length for the heuristic pilot is ≈ 10.4 turns (the spec's ≈ 13 was the random pilot).

9. **Never reuse Showdown names across process launches.** When a process dies mid-battle, Showdown keeps
   the battle open. The next login with the same name gets that battle re-attached a moment after login,
   and the env either treats the half-played battle as a new episode (seen: Phi(s_0) = 0.07) or fails on
   the following reset ("Can not reset player's battles while they are still running").
   **Decision:** names end in `config.LAUNCH_TAG`, 4 random hex characters per process launch.
   `OTSWrapper.reset` also raises if Phi(s_0) ≠ 0, which catches a re-attached battle.
   Orphaned battles stay on the server until it restarts; restart Showdown before each batch of runs.

## Throughput and step budget

`src/bench_env.py` on the 12-core dev machine, local Showdown server, full encoder (763 features),
MaskablePPO with `config.PPO_KWARGS`, CPU, 1 torch thread per process, random-ish early policy
(about 18 decision steps per battle):

| Processes | Steps/s per process | Total steps/s |
|---|---|---|
| 1 | 347 (placeholder encoder) | 347 |
| 6 | 124 | 709 |

With 6 runs in parallel, Showdown is the bottleneck. In-training evaluation adds about 22%.
**`TOTAL_STEPS = 2_000_000`** decision steps per run: about 4.5 h of training + 1 h of evaluation, so
about 5.5–6 h for all 6 runs of one algorithm in parallel. The DQN arm uses the same value.

## Running a batch

`src/run_batch.py` (see its docstring for the exact command): 6 runs, 3 at a time on the laptop (each run
≈ 1 GB RAM and < 1 core; 184 steps/s per process at 3 parallel → ≈ 3.7 h per wave, ≈ 7.5 h total).
Launch detached with `setsid nohup` and under `systemd-inhibit --what=sleep:idle:handle-lid-switch`, with
the Showdown server also started detached, so neither a closed terminal, a closed lid nor the end of an
editor session kills the batch. Keep the laptop plugged in. No resume: an interrupted run starts over.
Each batch starts from a committed, tagged state (`run_batch.py` refuses uncommitted `src/` or `Teams/`).

## More findings

10. **Gymnasium 1.x wrappers don't forward attributes.** `Monitor(env).action_masks()` raises
    AttributeError. Use `env.get_wrapper_attr("action_masks")()`, which is what sb3-contrib does.

11. **poke-env needs numpy integer actions.** `SinglesEnv.action_to_order` calls `action.item()`, so a plain
    `int` move action crashes. `OTSWrapper.step` converts with `np.int64(action)`.

12. **Two envs in one process work.** The training env and the evaluation env each run their own poke-env
    event-loop thread; in-training evaluation in the SB3 callback works while the training battle waits.
    The spec's `eval_watch.py` fallback is not needed.

13. **Smoke test (spec test 7) needs MaxBasePower, not Random.** An untrained greedy policy already beats
    `RandomPlayer` ~87% (eval 0.867 → 0.867 → 0.908 over 20k steps: learning is visible only in training
    battles, win rate 0.71 → 0.96). Against `MaxBasePowerPlayer`: **0.458 → 0.617 → 0.908**, with the
    first and last Wilson intervals disjoint. PPO diagnostics in the smoke runs: approx_kl ≈ 0.01,
    clip fraction 0.16–0.22, explained variance rising to 0.17–0.40.
