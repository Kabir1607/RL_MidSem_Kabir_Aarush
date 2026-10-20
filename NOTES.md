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
`[hp, atk, def, spa, spd, spe]` with the standard Gen 3+ formulas. Use it in `teamsheet.py`.

**MaskablePPO keywords.** All names in spec Section 5.2 exist in sb3-contrib 2.9.0. Note the defaults differ
from the spec for `batch_size` (64) and `ent_coef` (0.0), so both must be passed explicitly (they are, in
`config.PPO_KWARGS`).

**Showdown usernames.** userid (lowercase, non-alphanumerics stripped) must be ≤ 18 characters
(`pokemon-showdown/server/users.ts`). `config.account_name` asserts this; the longest is
`dqn_hv_s2_evopp`.

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
   **Decision:** the wrapper steps through them internally with action 0 and adds up their rewards.
   Confirm with a smoke test in Phase 2.

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
