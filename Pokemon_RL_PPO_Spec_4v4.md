# Pokémon RL Midsem — PPO Implementation Spec (4v4, open team sheets)

This document is written so that an engineer **or an LLM coding agent** can implement the PPO part of the project without other context. It contains: the research context (Section 1), the experimental design that must not be broken (Section 2), the environment and information setting (Section 3), the MDP with the **corrected reward** (Section 4), the PPO algorithm and its exact configuration (Section 5), how sample efficiency and stability are **measured and plotted during training** (Section 6), the code layout and interfaces (Section 7), verification tests (Section 8) and known pitfalls (Section 9).

Words in **MUST / MUST NOT** are hard requirements. Items marked **VERIFY** are facts that have not been checked against the installed library version; the implementer must read the installed source and confirm them before relying on them.

## Changes from the previous plan

| Area | Previous plan | This spec |
| :---- | :---- | :---- |
| Battle size | 6v6 singles | **4v4 singles** (the format the baselines were run in). Every "6" in team sizes, normalisers and the state is now **N = 4**. |
| Shaping potential | Φ nonzero at the end of a battle | **Φ(terminal) = 0** (otherwise the "policy unchanged" guarantee is false in episodic tasks) |
| Reward helper | Option to use poke-env's `reward_computing_helper` | **Must not be used**: it returns Φ(s′) − Φ(s) without γ |
| Shaping check | — | Discounted shaping sum per battle must be ≈ 0 (Section 8) |
| Horizon | "30–60 turns, γ = 0.99" | Measured mean ≈ 13 turns; γ = 0.99 kept; episode length now logged in decision steps |
| PPO loss | Only L^CLIP written | Full loss: clip + value + entropy; masking at update time stated explicitly |
| Action space | "4 moves + 5 switches" | 26 discrete actions; 4v4 means switch indices 4–5 are always masked |
| Evaluation | Separate script after training | **In-training evaluation callback** that writes a CSV and redraws the sample-efficiency plot at every evaluation |
| Schedule, task tables, owners | Included | Removed |

**Assumption to confirm with the user:** "4v4" here means **singles** with 4-Pokémon team files, no team-preview selection (all 4 are brought), slot 1 always leads. If the baselines were actually doubles (VGC-style), see Appendix A; the action space and masking change substantially.

---

## 1. Research context

We study how an RL algorithm's learning is affected by **outcome variance** in Pokémon battles, as a first step toward comparing playing styles (standard vs Perish Trap) in doubles VGC in the final project.

**The manipulation.** One team exists in two versions, identical in species, items, abilities, EVs, IVs, natures and all other moves, except that a few accurate moves are swapped for inaccurate, higher-power ones:

| LowVar | HighVar | Expected BP (power × accuracy) |
| :---- | :---- | :---- |
| Surf 90/100 | Hydro Pump 110/80 | 90 → 88 |
| Flamethrower 90/100 | Fire Blast 110/85 | 90 → 93.5 |
| Thunderbolt 90/100 | Thunder 110/70 | 90 → 77 |
| Ice Beam 90/100 | Blizzard 110/70 | 90 → 77 |

The actual swaps used are whatever is in the existing team files `Teams/lowvar.txt` and `Teams/highvar.txt`. Do not change them.

**Research questions.**

* **RQ1:** Does the HighVar team need more environment steps to reach the same performance, and do runs disagree more across seeds?
* **RQ2:** Does DQN degrade more than PPO on HighVar? (DQN is implemented separately; this spec covers PPO but constrains it so the comparison is fair, see Section 5.4.)
* **RQ3 (secondary):** Does final strength differ? Secondary because 110-BP moves also change KO thresholds.

**Why PPO should be relatively robust to variance (the hypothesis to test).** PPO uses on-policy data, a learned baseline (the critic), advantage normalisation and clipped updates, which all reduce gradient variance. DQN's `max` over noisy Q estimates is biased upward, and the bias grows with noise. Prediction: the LowVar→HighVar drop in sample efficiency is smaller for PPO than for DQN.

**What "variance" means here.** A win/loss outcome always has variance p(1−p). The manipulation makes each **transition** noisier (same state and action → more spread-out next states). That noise enters PPO through the advantages A_t and the critic targets.

---

## 2. Experimental invariants (MUST hold for every run)

1. The **only** difference between conditions is the team file (lowvar vs highvar) and the seed. Code, hyperparameters, step budget, opponent, encoder and reward are identical.
2. **Tera disabled for both players** (and mega/Z/dynamax, which are not in Gen 9). Use exactly the battle format string and opponent setup used for the baselines, so RL results are comparable with them. **VERIFY** from the baseline script and record the format string in `config.py`.
3. Opponent policy: poke-env's `SimpleHeuristicsPlayer` only, piloting one of the 3 opponent team files `teams/opp_a.txt`, `opp_b.txt`, `opp_c.txt`, chosen uniformly at random per training battle.
4. Team order is fixed; slot 1 always leads.
5. No weather setters, accuracy/evasion modifiers, Illusion, Transform/Imposter, Trick/Switcheroo on any team (these break either the variance manipulation or the open-team-sheet lookup). The existing files already satisfy this; the diff script (Section 8) re-checks it.
6. Never change code, team files or settings in the middle of a batch of runs. If a bug fix is needed, restart all affected runs.

---

## 3. Environment and information setting

### 3.1 Stack

* Pokémon Showdown server running locally (`node pokemon-showdown start --no-security`).
* Python 3.11, `poke-env`, `stable-baselines3`, `sb3-contrib`, `torch`, `matplotlib`, `pandas`, `numpy`. Pin versions in `requirements.txt`; the same file on every machine that trains.
* Environment class: a subclass of poke-env's `SinglesEnv`, wrapped with `SingleAgentWrapper(env, opponent=SimpleHeuristicsPlayer(...))` to give a Gymnasium single-agent env.

**VERIFY before writing code** (read the installed poke-env source and its `examples/` folder, write the answers in `NOTES.md`):

1. What `SingleAgentWrapper.step/reset` return as the observation (the raw `embed_battle` output, or a dict that also contains an action mask).
2. How to get the current `battle` object from the wrapped env (needed for the action mask and the reward).
3. Whether `calc_reward` is called exactly once per agent step, and whether it is also called on reset.
4. Whether `battle.opponent_team` contains all 4 opponent Pokémon from team preview or only revealed ones.
5. How the official poke-env RL example passes action masks to `MaskablePPO`. Copy that mechanism.
6. The exact class name and import path of `SimpleHeuristicsPlayer`, and whether it has random tie-breaks.

### 3.2 Open team sheets (OTS)

Our agent knows the **full set** (species, item, ability, 4 moves, nature, EVs, IVs, level, and therefore the 6 actual stats) of all 8 Pokémon from turn 1. Only our agent gets this; the heuristic opponent plays with revealed-only information. This asymmetry is identical in all conditions, so it cancels in the LowVar vs HighVar comparison but inflates absolute RL lift (Section 6.3).

Implementation (Route A: lookup from our own files):

* `teamsheet.py` parses all 5 team files into per-Pokémon dicts. Showdown exports omit IVs of 31 and EVs of 0, so fill defaults (IV 31, EV 0, level 100 unless stated).
* Actual stats: use poke-env's stats utilities if available (**VERIFY**), otherwise the Gen 3+ formulas:
  * HP = ⌊(2·Base + IV + ⌊EV/4⌋) · L / 100⌋ + L + 10
  * Other = ⌊(⌊(2·Base + IV + ⌊EV/4⌋) · L / 100⌋ + 5) · nature⌋, nature ∈ {1.1, 1.0, 0.9}
* Opponent identification: match the opponent's species set (from team preview) against the 3 opponent files; fall back to the lead species. Raise an error if zero or more than one file matches. Cache the result per `battle.battle_tag`.

---

## 4. MDP specification

### 4.1 Time step

One time step = **one decision request** from Showdown to our agent. This is not the same as a game turn: a forced switch after a faint is an extra decision with no new turn. All x-axes, budgets and episode lengths are measured in **decision steps**; battles are reported alongside.

Measured so far: mean ≈ 13 turns per battle in the baselines, so expect roughly 14–18 decision steps per battle. The real number is logged during training (Section 6).

### 4.2 State s (observation vector)

A fixed-length `float32` vector, every feature scaled to roughly [0, 1] (boosts to [−1, 1]). Declare the observation space with these bounds.

**Indexing rule.** Our 4 Pokémon occupy rows 0–3 in team-file order. The opponent's 4 occupy rows 4–7 in *their* sheet's order. Rows are keyed **by species**, never by the order of `battle.team` or of Showdown requests, which moves the active Pokémon around.

**Block 1 — per-Pokémon rows (8 rows).**

| Feature | Size | Source | Scaling |
| :---- | :---- | :---- | :---- |
| HP fraction | 1 | battle | [0, 1] |
| Fainted | 1 | battle | 0/1 |
| Status one-hot (none, brn, par, psn, tox, slp, frz) | 7 | battle | 0/1 |
| Active | 1 | battle | 0/1 |
| Actual stats HP, Atk, Def, SpA, SpD, Spe | 6 | sheet | each ÷ max of that stat over all 16 Pokémon in the 5 files |
| Types multi-hot (18) | 18 | sheet | 0/1 |
| Current item one-hot over the item vocabulary of the files, + "none/removed" | I + 1 | sheet, updated by battle (Knock Off, berries consumed) | 0/1 |
| Ability one-hot over the ability vocabulary of the files | B | sheet | 0/1 |
| Per move (×4): base power/150, accuracy (never-miss → 1.0), category one-hot (3), priority/5, type effectiveness vs the **opposing active**/4, STAB flag, PP fraction | 4 × 9 = 36 | sheet + battle | as listed |

Row size = 70 + I + B. Total for block 1 = 8 × (70 + I + B). Compute exactly from the files at startup; do not hard-code.

**Move accuracy MUST be included.** Without it the agent can only distinguish Surf from Hydro Pump through outcomes.

**Block 2 — active matchup.**

* Boost stages (Atk, Def, SpA, SpD, Spe, accuracy, evasion) for both actives, ÷ 6 → 14 values in [−1, 1].
* Speed order: effective speed = Spe × boost multiplier × 0.5 if paralysed × 1.5 if Choice Scarf. Encode one-hot (faster / tie / slower) and ours/(ours + theirs). Ignored modifiers (Tailwind, Protosynthesis/Quark Drive, Unburden, etc.) are listed in a code comment.
* Toxic counter for each active (÷ 15).

**Block 3 — field.** Hazards per side (Stealth Rock 0/1, Spikes ÷ 3, Toxic Spikes ÷ 2, Sticky Web 0/1), screens per side with turns remaining ÷ 8, and turn ÷ 50 (clipped to 1).

**Not in the state (deliberately):** damage-calculator features such as "probability this move KOs". They would multiply power by accuracy *for* the agent, partly removing the trade-off the experiment studies.

**Caching:** static parts (stats, types, ability, moves, normalisers) are computed once per battle and cached by `battle_tag`. Only dynamic parts are recomputed each step.

### 4.3 Action a

poke-env `SinglesEnv` action space of size 26 (Gen 9):

* 0–5: switch to team position 0–5
* 6–9: use move 1–4
* 10–25: move + mega / Z / dynamax / terastallize — always masked

With 4-Pokémon teams, positions 4 and 5 are always masked. The current active, fainted Pokémon and trapped situations are masked by `get_action_mask`. The mask is a boolean array of length 26 and MUST have at least one `True` entry on every step (assert this).

### 4.4 Reward r (corrected)

**Terminal reward:** +1 win, −1 loss, 0 tie, given on the final step only.

**Potential-based shaping.** With N = 4:

```
Φ(s) = 0.5 · (HP_ours − HP_opp) / N + 0.5 · (alive_ours − alive_opp) / N        Φ ∈ [−1, 1]
```

HP_x is the sum of HP fractions of that side's 4 Pokémon; alive_x the number not fainted. The opponent terms MUST use all 4 opponent Pokémon from the OTS rows, including ones not yet revealed (at full HP).

**Shaped reward at step t:**

```
r'_t = r_t + γ · Φ(s_{t+1}) − Φ(s_t),     with Φ(s_{t+1}) := 0 if s_{t+1} is terminal
```

**Why Φ(terminal) = 0 is required.** Over a battle of T steps the discounted shaping terms telescope:

```
Σ_{t=0}^{T−1} γ^t (γ Φ(s_{t+1}) − Φ(s_t)) = γ^T Φ(s_T) − Φ(s_0)
```

If Φ(s_T) were left at its natural value, the agent would be paid extra for winning with more HP left and (because of γ^T) for ending battles quickly, which is a different objective from "win". With Φ(s_T) = 0, and since Φ(s_0) = 0 at the start of every battle (both sides full HP, 4 alive), the discounted sum of shaping is **exactly zero**: the shaped discounted return equals the true discounted return. Shaping only moves credit earlier in time.

**MUST NOT use poke-env's `reward_computing_helper`.** It returns (current value − previous value), i.e. Φ(s′) − Φ(s) with no γ.

**Reference implementation** (inside the env subclass):

```python
GAMMA = 0.99   # MUST equal the PPO gamma; import it from config.py
N_TEAM = 4

def potential(self, battle) -> float:
    ours = self.ots.our_rows(battle)        # 4 dicts with hp_frac, fainted
    theirs = self.ots.opp_rows(battle)      # 4 dicts, unrevealed = full HP, not fainted
    hp_o = sum(p["hp_frac"] for p in ours)
    hp_t = sum(p["hp_frac"] for p in theirs)
    al_o = sum(not p["fainted"] for p in ours)
    al_t = sum(not p["fainted"] for p in theirs)
    return 0.5 * (hp_o - hp_t) / N_TEAM + 0.5 * (al_o - al_t) / N_TEAM

def calc_reward(self, battle) -> float:
    tag = battle.battle_tag
    prev = self._phi_prev.get(tag, 0.0)                 # Φ(s_0) = 0
    if battle.finished:
        r = 1.0 if battle.won else (-1.0 if battle.lost else 0.0)
        nxt = 0.0                                       # Φ(terminal) = 0
    else:
        r, nxt = 0.0, self.potential(battle)
    shaped = r + GAMMA * nxt - prev
    # bookkeeping for the shaping check (Section 8)
    k = self._step_idx.get(tag, 0)
    self._disc_shaping[tag] = self._disc_shaping.get(tag, 0.0) + (GAMMA ** k) * (GAMMA * nxt - prev)
    self._step_idx[tag] = k + 1
    self._phi_prev[tag] = nxt
    if battle.finished:
        self._finished_stats[tag] = (self._disc_shaping.pop(tag), self._step_idx.pop(tag))
        self._phi_prev.pop(tag, None)
    return shaped
```

If VERIFY item 3 shows `calc_reward` is called more than once per step (or on reset), move this logic into a Gymnasium wrapper around the single-agent env so it runs exactly once per `step()`.

**Note on the critic.** With shaping, the critic learns V(s) − Φ(s) rather than V(s). This is identical in all conditions, so comparisons of explained variance remain fair; state it when presenting explained variance.

### 4.5 Discount and horizon

γ = 0.99. Effective horizon 1/(1 − γ) = 100 steps, far above a ≈15-step battle; the terminal reward reaches the first decision with weight 0.99^15 ≈ 0.86. γ is not tuned. Combined with GAE λ = 0.95, the advantage credit window is about 1/(1 − γλ) ≈ 17 steps, i.e. about one battle.

Optional safety truncation: if a battle exceeds 300 decision steps, end the episode with `truncated=True` (not `terminated`), so SB3 bootstraps instead of treating it as a loss. Log how often this happens (expected: never).

---

## 5. PPO

### 5.1 Algorithm

Actor π_θ(a|s): masked softmax (illegal logits set to −∞ or a large negative number). Critic V_φ(s). Separate MLP heads.

Rollout of `n_steps` transitions, then for `n_epochs` passes over minibatches:

```
δ_t   = r'_t + γ (1 − done_{t+1}) V(s_{t+1}) − V(s_t)
A_t   = Σ_l (γλ)^l δ_{t+l}              (GAE, reset at episode boundaries)
R_t   = A_t + V(s_t)                    (critic target)
ρ_t   = π_θ(a_t|s_t) / π_old(a_t|s_t)   (both computed WITH the stored mask m_t)
L     = −E[min(ρ_t Â_t, clip(ρ_t, 1−ε, 1+ε) Â_t)] + c_v E[(V_φ(s_t) − R_t)²] − c_e E[H(π_θ(·|s_t))]
```

Â_t is A_t normalised per minibatch. **The mask MUST be applied at update time**, not only when sampling; otherwise π_θ spreads probability over illegal actions and ρ_t is wrong. `sb3_contrib.MaskablePPO` stores masks in its rollout buffer and does this. Use it rather than a custom PPO.

### 5.2 Configuration (identical for every run)

```python
from sb3_contrib import MaskablePPO

model = MaskablePPO(
    "MlpPolicy", env,
    learning_rate=3e-4,
    n_steps=2048,            # ≈ 120–150 battles per rollout at ≈15 steps/battle
    batch_size=128,
    n_epochs=10,
    gamma=0.99,              # MUST equal config.GAMMA used in calc_reward
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    normalize_advantage=True,
    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),  # SB3 default is 64-64
    seed=seed,
    tensorboard_log=run_dir,
    verbose=1,
)
```

**VERIFY** keyword names against the installed SB3/sb3-contrib versions.

* One environment per training process (`n_envs = 1`). Parallelism comes from running several processes at once, each with **unique Showdown account names** (e.g. `ppo_lv_s0_agent`, `ppo_lv_s0_opp`, `ppo_lv_s0_eval`, `ppo_lv_s0_evopp`). Usernames must be short enough for Showdown (≤ 18 characters, VERIFY).
* Seeds 0, 1, 2 per team version. Set `random`, `numpy` and `torch` seeds from the same value. Seeds control initialisation, exploration and opponent-team sampling, not Showdown's battle RNG.
* The step budget `TOTAL_STEPS` is a config value set from a throughput benchmark; it MUST be identical for every run.
* Wrap the env with SB3's `Monitor` so `rollout/ep_len_mean` and `rollout/ep_rew_mean` are logged.

### 5.3 Action masks

Expose the mask in the way the official poke-env RL example does (VERIFY item 5). If it is not exposed automatically, wrap the env with `sb3_contrib.common.wrappers.ActionMasker(env, mask_fn)` where `mask_fn` returns `np.array(SinglesEnv.get_action_mask(current_battle), dtype=bool)`. Before calling `predict`, always pass the mask: `model.predict(obs, action_masks=mask, deterministic=...)`.

### 5.4 Fairness constraints with the DQN arm (implemented separately)

* The encoder (`encoder.py`), reward (`calc_reward`), team rotation, opponent and evaluation code are **shared modules**, imported by both pipelines, never copied.
* Same hidden layers (256-256), same `TOTAL_STEPS`, same seeds, same evaluation protocol.
* No hyperparameter tuning of one algorithm without the same effort for the other.

---

## 6. Measuring and plotting sample efficiency during training

### 6.1 In-training evaluation callback (primary)

Implement `EvalCurveCallback(BaseCallback)` in `callbacks.py`:

* Holds a separate **evaluation env** (same encoder, same format, own account names, opponent `SimpleHeuristicsPlayer`).
* Runs an evaluation at step 0 (before any training) and then every `EVAL_FREQ = 10_000` decision steps.
* Each evaluation plays `N_EVAL = 120` battles with the **deterministic** policy (argmax over legal actions). Opponent teams are **stratified**: exactly 40 battles against each of A, B, C, in a fixed order. This removes opponent-sampling noise from the curve.
* After each evaluation it:
  1. Appends one row to `runs/<run_id>/eval.csv` (schema below).
  2. Saves a checkpoint `runs/<run_id>/ckpt/step_<steps>.zip`.
  3. Recomputes the running sample-efficiency metrics (Section 6.3).
  4. Redraws `runs/<run_id>/learning_curve.png` (Section 6.4).
  5. Logs `eval/win_rate`, `eval/wilson_low`, `eval/wilson_high`, `eval/auc_so_far` to TensorBoard.

`eval.csv` columns:

```
run_id, algo, team, seed, step, battles_trained, eval_n, wins, losses, ties,
win_rate, wilson_low, wilson_high, wr_vs_A, wr_vs_B, wr_vs_C,
mean_eval_len_steps, mean_eval_len_turns, wall_time_s
```

**Fallback if two envs in one process hang or interfere** (poke-env runs its own event loop; VERIFY): set the callback to only save checkpoints, and run `eval_watch.py` as a separate process that polls `runs/<run_id>/ckpt/`, evaluates each new checkpoint with the same protocol and writes the same `eval.csv` and plot. The outputs MUST be identical in format either way.

### 6.2 Per-episode training log

`TrainEpisodeLogger(BaseCallback)` appends to `runs/<run_id>/train_episodes.csv` at the end of every training battle:

```
step, battle_idx, opp_team, outcome(+1/-1/0), len_steps, len_turns, disc_shaping_sum, ep_return_shaped
```

This gives battles-vs-steps conversion, the training win rate (stochastic policy, not comparable across algorithms, but useful for debugging) and the shaping check.

### 6.3 Metric definitions

Let `b_team` be the heuristic baseline win rate for the same team version against the heuristic opponent, read from the existing baseline CSV (path in `config.BASELINE_CSV`; the column names must be mapped once in config).

* **Wilson 95% interval** for w wins out of n, z = 1.96:
  `centre = (p + z²/2n)/(1 + z²/n)`, `half = z·sqrt(p(1−p)/n + z²/4n²)/(1 + z²/n)`, p = w/n.
* **Threshold steps:** the first evaluation step at which the win rate is ≥ `b_team + 0.10` **for 2 consecutive evaluations**; report the first of the two. If never reached, report "not reached" (store as NaN), never as the final step.
* **AUC (normalised):** trapezoidal area under (step, win_rate) from step 0 to the current step, divided by the current step. This is the average evaluation win rate over training and needs no threshold. The headline sample-efficiency number is AUC at `TOTAL_STEPS`.
* **Within-run stability:** largest drop between consecutive evaluations, and standard deviation of the last 5 evaluations.
* **Final strength and lift:** final evaluation (Section 6.5) win rate minus `b_team` = RL lift. The **lift difference** (HighVar lift − LowVar lift, per algorithm) cancels the OTS information advantage and is the headline lift number.
* **PPO diagnostics** from SB3 logs: `train/explained_variance`, `train/approx_kl`, `train/clip_fraction`, `train/entropy_loss`, `rollout/ep_len_mean`.

### 6.4 Per-run plot (`learning_curve.png`, redrawn at every evaluation)

Use matplotlib with the `Agg` backend; write to a temporary file and rename, so a viewer never sees a half-written PNG. Two stacked panels sharing the x-axis (decision steps; a secondary top axis shows battles trained, from `train_episodes.csv`):

1. **Top:** evaluation win rate with the Wilson band shaded; dashed horizontal line at `b_team` labelled "heuristic baseline"; dotted line at `b_team + 0.10` labelled "threshold"; vertical line at the threshold step once reached; title with run id, current AUC and threshold step.
2. **Bottom:** rolling mean (window 200 battles) of training episode length in steps, and `explained_variance` if available.

### 6.5 Final evaluation

At the end of training, evaluate the last checkpoint deterministically for 1,020 battles vs the heuristic (340 per opponent team), plus 204 each vs `RandomPlayer` and `MaxBasePowerPlayer` (sanity baselines), and write `runs/<run_id>/final_eval.csv`.

### 6.6 Aggregation script (`analyze.py`)

Reads every `runs/*/eval.csv` and `final_eval.csv`, groups by (algo, team), and produces:

* `figures/curves_<algo>.png`: LowVar and HighVar learning curves, **every seed drawn as its own line** plus the mean, with the two baselines as dashed lines.
* `figures/interaction_auc.png`: 2×2 interaction plot (x = team version, colour = algorithm, y = final AUC, one dot per seed, line through the means).
* `figures/threshold_steps.png`: threshold steps per cell, one dot per seed, "not reached" shown as a marker at the top.
* `figures/ep_len.png`: episode length over training per cell.
* `summary.csv`: per cell, the mean and per-seed values of AUC, threshold steps, final win rate (Wilson), RL lift, lift difference, seed std of final win rate, largest within-run drop.

With 3 seeds, show every seed; do not present a mean without the individual points.

---

## 7. Code layout and interfaces

```
project/
  config.py           # all constants: GAMMA, N_TEAM, TOTAL_STEPS, EVAL_FREQ, N_EVAL, BATTLE_FORMAT,
                      # team paths, BASELINE_CSV, server config, PPO hyperparameters
  teamsheet.py        # parse team files, compute stats, vocabularies, normalisers, opponent identification
  encoder.py          # embed_battle(battle) -> np.ndarray, observation space definition
  env.py              # PPOSinglesEnv(SinglesEnv): embed_battle, calc_reward, potential; make_env(team, seed, role)
  teams.py            # Teambuilder subclasses: fixed team for us; random (train) / stratified (eval) for opponent
  callbacks.py        # EvalCurveCallback, TrainEpisodeLogger
  evaluate.py         # evaluate(model, env, n, stratified=True) -> dict; wilson(); used by callback and scripts
  plotting.py         # plot_run(run_dir), used by callback and analyze.py
  train_ppo.py        # CLI: --team {lowvar,highvar} --seed INT [--steps INT]
  eval_watch.py       # fallback checkpoint evaluator (same outputs)
  analyze.py          # aggregation and figures
  check_teams.py      # diff script and team-rule checks (Section 8)
  tests/              # Section 8 tests
  runs/<algo>_<team>_s<seed>/   # eval.csv, train_episodes.csv, final_eval.csv, learning_curve.png, ckpt/, tb/
```

`run_id = f"ppo_{team}_s{seed}"`. `train_ppo.py` MUST write a `run_config.json` containing every config value, the git commit hash, and library versions, and refuse to start if `runs/<run_id>/` already has checkpoints (unless `--overwrite`).

---

## 8. Verification (all MUST pass before the main runs)

1. **Team diff:** `check_teams.py` asserts LowVar and HighVar differ only in the intended moves; the 3 opponent teams have pairwise different species sets and leads; all teams have exactly 4 Pokémon; no banned abilities, items or moves (Section 2.5).
2. **Stats:** computed stats equal the stats shown in the Showdown teambuilder for all 16 Pokémon.
3. **Encoder:** over 20 battles: fixed length, no NaN/inf, all values within declared bounds, opponent rows contain moves and stats at turn 1, a Pokémon's HP never appears in another Pokémon's row after switches.
4. **Opponent identification:** the identified team equals the team actually sent in 20 battles.
5. **Mask:** every mask has at least one legal action; indices 4, 5 and 10–25 are never legal; no `-terastallize` lines in battle logs.
6. **Reward:**
   * the last step of each battle contains the ±1 (or 0 for a tie) with the correct sign;
   * per battle, the **discounted** shaping sum `disc_shaping_sum` is 0 to within 1e-6 (the undiscounted sum is not exactly zero when γ < 1, so do not test that);
   * |shaping per step| ≤ 2 always, and typically ≪ 1.
7. **Smoke run:** 20k steps vs `RandomPlayer` produces a clearly rising evaluation win rate, a correctly redrawn `learning_curve.png`, and `eval.csv` rows at steps 0, 10k, 20k.
8. **Determinism of evaluation:** evaluating the same checkpoint twice gives the same action for the same observation (not the same win rate, since battle RNG differs).

Healthy PPO ranges during training (rules of thumb, not guarantees): `approx_kl` below ~0.02–0.05, `clip_fraction` ~0.05–0.3, `explained_variance` rising above 0, entropy decreasing gradually rather than collapsing in the first few updates.

---

## 9. Known pitfalls

| Pitfall | Symptom | Fix |
| :---- | :---- | :---- |
| γ in `calc_reward` differs from PPO's γ | Shaping no longer potential-based | Both import `config.GAMMA` |
| Φ(terminal) not set to 0 | `disc_shaping_sum` ≠ 0 | Check the `battle.finished` branch |
| Shaping state not keyed per battle | Huge reward spikes at battle start | Dict keyed by `battle_tag`, cleaned at battle end |
| Opponent HP from revealed Pokémon only | Φ jumps when a new opponent is revealed | Use OTS rows (unrevealed = full HP) |
| Mask not used at update time | Training unstable, KL spikes | Use `MaskablePPO`, not `PPO` |
| Evaluation with stochastic policy | Curves not comparable with DQN | `deterministic=True` everywhere in evaluation |
| Rows keyed by list position | Features swap between Pokémon after switches | Key by species |
| Same Showdown username in two processes | Challenges fail or battles cross runs | Unique names per process and role |
| Editing team files mid-batch | Observation changes silently | Never; restart all runs |
| Threshold "not reached" stored as the last step | Fake sample-efficiency numbers | Store NaN and show it explicitly |

---

## Appendix A: if the 4v4 battles are doubles

Only if the baselines were run in a doubles format:

* Use `DoublesEnv`; the action is a pair (one per active slot), each in a 107-value space (pass, switches, 4 moves × 5 targets, gimmicks).
* `MaskablePPO` supports `MultiDiscrete` action spaces with per-dimension masks (VERIFY in sb3-contrib), but independent per-slot masks cannot express joint constraints (both slots switching to the same Pokémon, both choosing "pass"). Resolve invalid joint actions with a deterministic fallback in `action_to_order`, and log how often it fires.
* The state gains partner features and per-slot rows; Φ is unchanged (N = 4).
* VGC-Bench's doubles action encoding is the reference implementation to study.

## Appendix B: sources

* poke-env documentation, env module (`reward_computing_helper` returns value difference without γ; SinglesEnv action mapping 0–25): https://poke-env.readthedocs.io/en/stable/modules/env.html
* sb3-contrib MaskablePPO documentation.
* Cited from memory, not re-opened: Schulman et al. 2017 (PPO); Schulman et al. 2016 (GAE); Ng, Harada & Russell 1999 (potential-based shaping); Grześ 2017 (shaping in episodic tasks requires Φ(terminal) = 0); van Hasselt 2010 (Double Q-learning, overestimation); Agarwal et al. 2021 (reporting with few seeds). Stat formulas and move data are also from memory and are checked by tests 1–2.
