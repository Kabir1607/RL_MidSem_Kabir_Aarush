# DQN arm: reference and specification

This is everything needed to implement the DQN arm so that it is a **fair comparison** with the PPO arm.
Read it together with `Pokemon_RL_PPO_Spec_4v4.md` (repo root; the experiment design) and `NOTES.md`
(repo root; how poke-env actually behaves). Where this document and the spec disagree, this document
reflects the code as it is and wins.

**MUST / MUST NOT** are hard requirements: breaking one makes the PPO-vs-DQN comparison (RQ2) invalid.

---

## 0. Checklist

- [ ] Budget: **`config.TOTAL_STEPS` = 2,000,000 decision steps per run**, 6 runs: {lowvar, highvar} × seeds {0, 1, 2}.
- [ ] Environment from `env.make_env(team, seed, algo="dqn")`. Never raw poke-env.
- [ ] γ = `config.GAMMA` (0.99), hidden layers `config.NET_ARCH` ([256, 256]), one env per process.
- [ ] Action masks used in **three** places: ε-exploration, greedy action, and the max in the TD target (Section 5.3).
- [ ] Terminal vs truncated handled correctly in the TD target (Section 5.4).
- [ ] Logging and evaluation through `recorder.RunRecorder` (Section 6): eval at step 0 and every 10,000 steps,
      120 battles stratified 40/40/40, deterministic (greedy) policy; final evaluation at the end.
- [ ] Hyperparameters fixed **before** the runs and identical across all 6 runs (Section 5.5).
- [ ] Tests in Section 8 pass, including a 20k-step smoke run against `RandomPlayer`.

---

## 1. Research context (short)

One team exists in two versions that differ only in four moves: accurate moves swapped for stronger,
less accurate ones (from the team files: Moltres Flamethrower→Fire Blast, Air Slash→Hurricane;
Primarina Surf→Hydro Pump, Ice Beam→Blizzard). HighVar battles have noisier transitions.

* **RQ1:** does HighVar need more environment steps to reach the same performance, and do seeds disagree more?
* **RQ2 (needs your arm):** does **DQN degrade more than PPO** on HighVar? Hypothesis: DQN's `max` over noisy
  Q-estimates is biased upward and the bias grows with noise (van Hasselt 2010), while PPO's critic baseline,
  advantage normalisation and clipping damp gradient variance.
* **RQ3 (secondary):** does final strength differ?

Because RQ2 is about the `max` operator, use **vanilla DQN** (target = `max_a' Q_target(s', a')`), not Double
DQN, unless both partners agree to change the research question. Double DQN exists precisely to remove the
bias the hypothesis is about.

---

## 2. Experimental invariants (same as the spec, Section 2)

1. The only difference between runs is the team file (lowvar/highvar) and the seed. Code, hyperparameters,
   step budget, opponent, encoder and reward are identical.
2. Format `gen9ounotera` (Gen 9 OU singles, Tera disabled for both players), 4 Pokémon per team, everyone
   brings all 4, slot 1 always leads. `make_env` does all of this.
3. Opponent: poke-env `SimpleHeuristicsPlayer`, piloting `Teams/opp_a.txt`, `opp_b.txt` or `opp_c.txt`,
   uniformly at random per training battle (seeded).
4. Never edit team files, `config.py` or shared modules in the middle of a batch. If a bug fix is needed,
   restart all affected runs (for both algorithms if a shared module changed).
5. Our agent sees the full team sheets of all 8 Pokémon (open team sheets); the heuristic opponent does not.
   This is identical in all conditions.

---

## 3. Repository, shared modules, running

```
Teams/                      lowvar.txt, highvar.txt, opp_a/b/c.txt  (never edit)
baselines_summary.csv       heuristic baseline win rates (b_team): lowvar 0.4287, highvar 0.4373
NOTES.md                    verified poke-env behaviour + design decisions
src/
  config.py                 ALL shared constants (budget, gamma, eval protocol, paths, account names)
  teamsheet.py              parses team files, stats, vocabularies, opponent identification
  encoder.py                observation vector (763 floats)
  env.py                    make_env(): the environment
  teams.py                  opponent teambuilders (random for training, stratified for evaluation)
  evaluate.py               evaluate(), final_evaluation(), make_eval_env()
  metrics.py                wilson(), baseline_win_rate(), auc(), threshold_step(), stability()
  plotting.py               plot_run(run_dir)
  recorder.py               RunRecorder: algorithm-agnostic logging/evaluation  <- you use this
  callbacks.py              SB3 callbacks wrapping RunRecorder (PPO uses these; usable by SB3-based DQN)
  train_ppo.py              the PPO trainer (a working example of the whole pipeline)
  DQN/                      <- DQN code goes here (e.g. src/DQN/train_dqn.py)
tests/                      pytest suite (needs the Showdown server for live tests)
runs/<algo>_<team>_s<seed>/ outputs per run
```

**Shared modules MUST be imported, never copied or modified for DQN.** If something must change, change it
in `src/` for both arms and restart both batches.

Importing from `src/DQN/train_dqn.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/
import config
from env import make_env
from recorder import RunRecorder, seed_everything
```

**Running anything:**

```bash
# terminal 1: Showdown server (restart it before each batch; see Section 9)
cd pokemon-showdown && node pokemon-showdown start --no-security
# terminal 2
source venv/bin/activate          # Python 3.14, versions pinned in requirements.txt
python -m pytest                  # all tests must pass
python src/check_teams.py         # team rules
```

Installed and pinned: poke-env 0.16.1, stable-baselines3 2.9.0, sb3-contrib 2.9.0, gymnasium 1.3.0, torch.

---

## 4. The environment (what `make_env` gives you)

```python
env = make_env(team, seed, role="train", opponent="heuristic", algo="dqn", opp_per_team=None,
               encoder=None, monitor=True)
```

* `team`: `"lowvar"` or `"highvar"`. `seed`: run seed (also seeds the opponent-team rotation).
* `role="train"` for the training env. The evaluation env is built for you by `RunRecorder`.
* `algo="dqn"` makes the Showdown account names `dqn_lv0_agent_xxxx` etc.; the 4-hex suffix is random per
  process launch so crashed runs never collide (NOTES.md item 9).
* Returns a Gymnasium env (SB3 `Monitor` around `OTSWrapper`). `n_envs = 1` per process; run the 6 runs as
  6 separate processes.

### 4.1 Time step

**One step = one decision request from Showdown to our agent** (a forced switch after a faint, or the
switch after U-turn/Flip Turn, is an extra decision without a new turn). poke-env also produces dummy steps
in which our action is ignored (only the opponent acts); `OTSWrapper` skips those internally. Every
budget, x-axis and episode length is in these decision steps. Measured: about 18 steps per battle with a
random policy.

### 4.2 Observation

`float32` vector of length **763**, `env.observation_space` is a `Box` with exact bounds (all features in
[0, 1], except boosts and move priority in [−1, 1]). Use it as-is: no extra features, no different
normalisation.

| Block | Size | Content |
| :-- | :-- | :-- |
| 1 | 8 × 91 = 728 | One row per Pokémon: ours (rows 0–3, team-file order), opponent's (rows 4–7, their sheet order). Keyed by species, never by list position. |
| 2 | 20 | Boosts of both actives (atk, def, spa, spd, spe, accuracy, evasion; ÷6), speed order one-hot (faster/tie/slower), our speed ÷ (ours + theirs), toxic counters ÷15 |
| 3 | 15 | Hazards per side (Stealth Rock, Spikes ÷3, Toxic Spikes ÷2, Sticky Web), screens per side (turns left ÷8), turn ÷50 |

Row layout (offsets within a row, from `encoder.ROW_OFF`): hp 0, fainted 1, status one-hot 2–8 (none, brn,
par, psn, tox, slp, frz), active 9, stats 10–15 (÷ max over the 16 Pokémon), types 16–33 (18 multi-hot),
item 34–40 (6 items + none/removed), ability 41–54 (14), moves 55–90 = 4 × 9 per move: base power ÷150,
**accuracy**, category one-hot (physical/special/status), priority ÷5, type effectiveness vs the opposing
active ÷4, STAB, PP fraction. Move slot *i* of the active Pokémon is action *6 + i*.

Unrevealed opponent Pokémon appear at full HP with their sheet item. There are deliberately no
damage-calculator features (e.g. KO probability).

### 4.3 Actions and masks

`Discrete(26)`: 0–5 switch to team position 0–5, 6–9 use move 1–4, 10–25 gimmicks (mega/Z/dynamax/tera).
With 4-Pokémon teams and Tera banned, **indices 4, 5 and 10–25 are never legal**. Typically 2–7 actions are
legal.

```python
action_masks = env.get_wrapper_attr("action_masks")   # gymnasium 1.x wrappers don't forward attributes
mask = action_masks()                                 # np.ndarray, bool, shape (26,), >= 1 True
```

Call it **after** `reset()`/`step()`; it describes the legal actions in the observation just returned.
Actions may be a Python `int` or a numpy integer.
The env raises an error if the mask is empty or contains a never-legal index. Taking an illegal action
raises an error (`strict=True`), so a missing mask fails loudly.

### 4.4 Reward

Terminal reward +1 win / −1 loss / 0 tie on the last step, plus potential-based shaping:

```
Φ(s) = 0.5·(HP_ours − HP_opp)/4 + 0.5·(alive_ours − alive_opp)/4          (unrevealed opponents count as full HP)
r'_t = r_t + γ·Φ(s_{t+1}) − Φ(s_t),   with Φ(terminal) := 0 and Φ(s_0) = 0
```

With Φ(terminal) = 0 the discounted shaping sum over a battle is exactly zero, so the optimal policy is
unchanged (Ng et al. 1999; Grześ 2017). **The γ here is `config.GAMMA`, so your DQN's γ MUST be
`config.GAMMA`**; otherwise the shaping is no longer potential-based. Typical shaped rewards per step are
small (|r'| ≪ 1); the last step carries roughly ±1.

### 4.5 Episode end

* `terminated = True` when the battle is finished (win, loss **or tie**).
* `truncated = True` only at the safety cap of `config.MAX_EPISODE_STEPS` = 300 decision steps (expected
  never). The next `reset()` forfeits the unfinished battle.
* On the last step `info["battle_end"]` is a dict: `battle_tag, opp_team ('A'/'B'/'C'), outcome (+1/−1/0),
  truncated, len_steps, len_turns, wait_skips, disc_shaping_sum, ep_return_shaped`. Pass it to
  `RunRecorder.episode_end` (Section 6).

---

## 5. DQN specification

### 5.1 Fixed by the comparison (MUST)

| Item | Value | Where |
| :-- | :-- | :-- |
| Step budget | 2,000,000 decision steps per run, stop exactly there | `config.TOTAL_STEPS` |
| Runs | lowvar/highvar × seeds 0, 1, 2 | `config.TEAMS`, `config.SEEDS` |
| Discount | 0.99 | `config.GAMMA` |
| Q-network | MLP 763 → 256 → 256 → 26, ReLU (same hidden sizes as PPO's actor and critic) | `config.NET_ARCH` |
| Envs | 1 per process | |
| Seeding | `random`, `numpy`, `torch` from the run seed | `recorder.seed_everything(seed)` |
| Device | CPU, 1 torch thread per process (6 processes share 12 cores) | `config.DEVICE`, `config.TORCH_THREADS` |
| Evaluation | step 0 and every 10,000 steps, 120 battles (40 per opponent team), greedy over legal actions | `RunRecorder.evaluate` |
| Final evaluation | 1,020 vs heuristic (340 per team) + 204 vs Random + 204 vs MaxBasePower | `RunRecorder.final` |
| run_id | `dqn_<team>_s<seed>` | `config.run_id("dqn", team, seed)` |

The step budget counts **environment steps returned by this env**: not battles, not gradient updates, not
wall-clock time. Evaluation battles do not count toward the budget.

### 5.2 Algorithm

Standard DQN (Mnih et al. 2015): experience replay, a target network, ε-greedy exploration, Huber loss,
Adam, gradient clipping. Vanilla max target (see Section 1).

### 5.3 Action masking (MUST, in three places)

SB3's `DQN` does **not** support action masks. Without masking, the greedy action can be illegal and,
worse, the TD target takes a max over actions that can never be taken, which inflates targets in a way
that has nothing to do with outcome variance and would confound RQ2.

1. **Exploration:** with probability ε, sample uniformly among **legal** actions.
2. **Greedy action:** `argmax` of Q over legal actions only.
3. **TD target:** max over **legal next actions** only. So the replay buffer MUST store the next state's mask.

```python
# acting
q = q_net(obs_t)                                    # (26,)
q[~mask] = -inf
a = int(q.argmax()) if rng.random() >= eps else int(rng.choice(np.flatnonzero(mask)))

# learning, batch from replay: s, a, r, s2, next_mask (bool, B×26), terminated (B,)
with torch.no_grad():
    q_next = q_target(s2).masked_fill(~next_mask, float("-inf")).max(dim=1).values
    q_next = torch.where(terminated, torch.zeros_like(q_next), q_next)   # terminal: no bootstrap
    target = r + gamma * q_next
loss = F.smooth_l1_loss(q_net(s).gather(1, a[:, None]).squeeze(1), target)
```

The next-state mask of a **terminal** transition can be all-False; it is never used because the
terminated branch zeroes `q_next` (guard against `-inf * 0 = nan`: use `torch.where` as above, not
multiplication).

### 5.4 Terminated vs truncated (MUST)

Bootstrap through `truncated` (time-limit) transitions, never through `terminated` ones:
`done_for_target = terminated` (not `terminated or truncated`). Truncation should never happen
(Section 4.5), but get it right anyway.

### 5.5 Hyperparameters

The spec fixes none for DQN. PPO uses the spec's configuration, which is close to library defaults and
**untuned**; to keep tuning effort equal, start from **SB3's DQN defaults** and change only what the
problem forces. Agree on the final table **before** the main runs, write it into `run_config.json`, and
never change it between the 6 runs.

| Hyperparameter | SB3 default | Suggested | Why |
| :-- | :-- | :-- | :-- |
| learning_rate | 1e-4 | 1e-4 | default |
| buffer_size | 1,000,000 | **100,000** | 1M × 763 floats × 2 (obs, next obs) ≈ 6.1 GB per run, 36 GB for 6 runs; the dev machine has 14 GB. 100k ≈ 0.6 GB (≈ 5,500 battles). |
| learning_starts | 100 | 10,000 | fill the buffer with a few hundred battles first |
| batch_size | 32 | 32 | default |
| train_freq / gradient_steps | 4 / 1 | 4 / 1 | default |
| target_update_interval | 10,000 (hard, tau = 1) | 10,000 | default |
| exploration | 1.0 → 0.05 over 10% of training | 1.0 → 0.05 over the first 200,000 steps | default (10% of 2M) |
| max_grad_norm | 10 | 10 | default |
| loss | Huber | Huber | default |
| gamma | 0.99 | `config.GAMMA` | fixed |

If anything is tuned, PPO gets the same tuning effort (spec Section 5.4).

---

## 6. Logging and evaluation: `recorder.RunRecorder`

All runs of both algorithms must produce the same files with the same columns, from the same code.
`RunRecorder` is algorithm-agnostic: it takes a **policy function** `policy(obs, mask) -> int` that must be
**deterministic** (greedy over legal actions, no ε).

```python
seed_everything(seed)
env = make_env(team, seed, algo="dqn")
action_masks = env.get_wrapper_attr("action_masks")
rec = RunRecorder("dqn", team, seed)                     # creates runs/dqn_<team>_s<seed>/, eval env, CSVs
rec.write_run_config(hyperparameters={...}, env=env)     # every DQN hyperparameter; env -> account names

def greedy(obs, mask):
    with torch.no_grad():
        q = q_net(torch.as_tensor(obs)[None]).squeeze(0)
    q[~torch.as_tensor(mask)] = -float("inf")
    return int(q.argmax())

def save(path):                                          # checkpoint callback; path has no extension
    torch.save({"q_net": q_net.state_dict(), "step": step}, path.with_suffix(".pt"))

rec.evaluate(0, greedy, save)                            # evaluation at step 0, before any training
obs, _ = env.reset(seed=seed)
for step in range(1, config.TOTAL_STEPS + 1):
    mask = action_masks()
    a = act(obs, mask, eps(step))                        # masked epsilon-greedy
    next_obs, r, terminated, truncated, info = env.step(a)
    next_mask = action_masks()
    buffer.add(obs, a, r, next_obs, next_mask, terminated)
    ... train every train_freq steps after learning_starts; update target every target_update_interval ...
    if terminated or truncated:
        rec.episode_end(step, info["battle_end"])
        next_obs, _ = env.reset()
    obs = next_obs
    if step % config.EVAL_FREQ == 0:
        rec.evaluate(step, greedy, save)                 # also at step == TOTAL_STEPS
rec.final(config.TOTAL_STEPS, greedy)                    # final evaluation of the last policy
rec.close()
```

(`rec.log_scalars(step, {...})` optionally records DQN diagnostics, e.g. `train/loss`, `train/q_mean`,
`train/epsilon`, into `runs/<run_id>/diagnostics.csv`; they are not used for the comparison.)

What `RunRecorder` writes in `runs/<run_id>/`:

| File | Content |
| :-- | :-- |
| `run_config.json` | all `config` values, your hyperparameters, git commit, library versions, account names |
| `train_episodes.csv` | one row per training battle: `step, battle_idx, opp_team, outcome, len_steps, len_turns, disc_shaping_sum, ep_return_shaped` |
| `eval.csv` | one row per evaluation: `run_id, algo, team, seed, step, battles_trained, eval_n, wins, losses, ties, win_rate, wilson_low, wilson_high, wr_vs_A, wr_vs_B, wr_vs_C, mean_eval_len_steps, mean_eval_len_turns, wall_time_s` |
| `ckpt/step_<step>.*` | your checkpoints, via the `save` callback |
| `learning_curve.png` | redrawn after every evaluation |
| `final_eval.csv` | one row per opponent (heuristic, random, maxbp) with win rate, Wilson interval, per-team win rates, RL lift |

Full signature: `RunRecorder(algo, team, seed, suffix="", opponent="heuristic", overwrite=False)`.
`suffix` makes the run id `dqn_<team>_s<seed>_<suffix>` (use it for smoke runs); `opponent` is the scripted
opponent of the in-training evaluations (`"heuristic"` for the main runs). `RunRecorder` refuses to start if
`runs/<run_id>/` already has results, unless `overwrite=True` (which deletes the old run directory).
The evaluation env lives in the same process as the training env; this is tested and works (the spec's
`eval_watch.py` fallback is not needed).

---

## 7. Metrics (computed from the files above, same for both algorithms)

* **Wilson 95% interval** for each evaluation win rate.
* **Threshold steps:** first evaluation step where win rate ≥ b_team + 0.10 for 2 consecutive evaluations
  (b_team = heuristic baseline for that team version). "Not reached" is NaN, never the last step.
* **AUC:** trapezoidal area under (step, win rate) from 0 to the current step, ÷ current step = average
  evaluation win rate over training. **Headline sample-efficiency number: AUC at 2,000,000.**
* **Stability:** largest drop between consecutive evaluations; std of the last 5 evaluations.
* **RL lift:** final win rate − b_team. Headline: HighVar lift − LowVar lift, per algorithm.
* **RQ2 test:** the LowVar→HighVar change in AUC (and threshold steps) for DQN vs for PPO, every seed shown.

---

## 8. Verification before the main runs (MUST pass)

1. `python -m pytest` passes (shared environment tests).
2. Your own tests for the DQN code, at least:
   * the greedy action is always legal (property test on random Q-values and masks);
   * the TD target ignores illegal next actions (set an illegal action's Q to +1e6: target unchanged);
   * terminal transitions do not bootstrap; truncated ones do; no NaN with all-False terminal masks;
   * ε-greedy never samples an illegal action.
3. **Smoke run:** 20,000 steps against `MaxBasePowerPlayer` (`make_env(..., opponent="maxbp")` and
   `RunRecorder(..., suffix="smoke", opponent="maxbp")`) gives a clearly rising evaluation win rate,
   `eval.csv` rows at 0, 10k and 20k, and a correctly redrawn `learning_curve.png`. Use MaxBasePower, not
   Random: an untrained greedy policy already beats `RandomPlayer` ~87% of the time, so there is no room to
   rise. PPO reference (`python src/train_ppo.py --team lowvar --seed 0 --steps 20000 --opponent maxbp
   --suffix smoke --skip-final-eval`): 0.458 → 0.617 → 0.908. DQN learns more slowly early on
   (learning_starts, ε-decay), so for the smoke run shorten them (e.g. learning_starts 1,000, ε over 10,000
   steps); the main runs use the agreed table.
4. Determinism of evaluation: the same checkpoint gives the same action for the same observation.
5. Healthy signs during the main runs: Q-values stay bounded (|Q| should stay near the [−1.5, 1.5] range
   the rewards allow), loss does not explode, `train_episodes.csv` shows `disc_shaping_sum` ≈ 0.

---

## 9. Operations and pitfalls

| Pitfall | Symptom | Fix |
| :-- | :-- | :-- |
| DQN γ ≠ `config.GAMMA` | shaping no longer potential-based | use `config.GAMMA` |
| No mask in TD target | overestimated Q, slow/unstable learning unrelated to variance | store next masks, masked max |
| Bootstrapping on terminal / not on truncated | wrong targets | `done_for_target = terminated` |
| `-inf * 0` in target | NaN loss | `torch.where`, not multiplication |
| Raw poke-env env or a copied encoder | steps or observations differ from PPO | `make_env`, shared modules |
| Replay buffer of 1M | out of memory with 6 runs | ~100k (Section 5.5) |
| Stochastic evaluation | curves not comparable with PPO | greedy, `RunRecorder.evaluate` |
| Running past 2M steps | unequal budget | stop exactly at `TOTAL_STEPS` |
| Editing shared files mid-batch | silent observation/reward changes | never; restart all runs |
| Crashed process restarted with the same Showdown names | env attaches to a half-played battle | handled by the per-launch name suffix; `reset` raises if Φ(s₀) ≠ 0 |
| Orphaned battles from crashes | server slows over time | restart Showdown before each batch |

**Time:** at about 124 steps/s per process with 6 processes on one 12-core machine, 2M steps take about
4.5 h plus about 1 h of evaluation: **≈ 5.5–6 h for the 6 DQN runs in parallel**, on a machine with its
own Showdown server. Running the DQN and PPO batches simultaneously on the same server makes the server
the bottleneck (≈ 10+ h for both). DQN's per-step compute (1 gradient step of batch 32 every 4 steps) is
comparable to PPO's, so the environment dominates either way.

---

## 10. References

* V. Mnih et al., "Human-level control through deep reinforcement learning", *Nature* 518, 2015 (DQN).
* H. van Hasselt, "Double Q-learning", NeurIPS 2010; van Hasselt, Guez & Silver, "Deep Reinforcement
  Learning with Double Q-learning", AAAI 2016 (overestimation from the max operator; the RQ2 mechanism).
* S. Thrun & A. Schwartz, "Issues in using function approximation for reinforcement learning", 1993
  (noise in Q-estimates biases the max upward).
* J. Schulman et al., "Proximal Policy Optimization Algorithms", 2017; "High-Dimensional Continuous Control
  Using Generalized Advantage Estimation", 2016 (the PPO arm).
* A. Ng, D. Harada & S. Russell, "Policy invariance under reward transformations", ICML 1999; M. Grześ,
  "Reward shaping in episodic reinforcement learning", AAMAS 2017 (Φ(terminal) = 0).
* S. Huang & S. Ontañón, "A Closer Look at Invalid Action Masking in Policy Gradient Algorithms", 2020
  (masking at sampling *and* update time).
* R. Agarwal et al., "Deep Reinforcement Learning at the Edge of the Statistical Precipice", NeurIPS 2021
  (reporting with few seeds: show every seed).
* Stable-Baselines3 DQN docs: https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
  (defaults in Section 5.5 checked against the installed 2.9.0).
* sb3-contrib MaskablePPO docs: https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html
* poke-env docs: https://poke-env.readthedocs.io/ (installed 0.16.1; behaviour verified in `NOTES.md`).
