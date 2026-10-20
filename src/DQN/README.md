# DQN implementation guide

**Status: implemented; all 104 tests and the fixed smoke run pass.**
The ten main research runs have not been launched. This document
describes the implementation and operating commands. Design decisions and equations
are recorded in [docs/dqn.tex](../../docs/dqn.tex).

## 1. Requirements and resolved decisions

Read [builder_notes.txt](../../docs/builder_notes.txt), the
[DQN reference](DQN_Reference/reference.md), the shared code, and the newer
[PPO specification](../../Pokemon_RL_PPO_Spec_4v4.md). Use the proposal PDF for
research context rather than replacing the current experiment with its older setup.

| Decision | Fixed choice |
|---|---|
| Implementation boundary | All new executable code and tests under `src/DQN/`; import shared modules. |
| Documentation | Design in `docs/dqn.tex`; implementation handoff/status in this README. |
| Teams and game | Existing LowVar/HighVar and opponent A/B/C files; 4v4 singles, fixed leads, `gen9ounotera`. |
| Information | Existing full-team-sheet encoder; 763 float32 features, separate 26-element legality mask. |
| Algorithm | Vanilla one-step masked DQN; no Double DQN, dueling, prioritized replay, or recurrence. |
| Fairness | Shared reward, gamma, network hidden sizes, decision budget, opponents, seeds, and evaluation. |
| Seeds/runs | Read `config.SEEDS`: currently 0–4, ten DQN main runs. Reference mentions of six runs are stale. |
| Batch integration | DQN-local adapter to shared scheduler; do not edit `src/run_batch.py`. |
| Existing work | Preserve shared/PPO code, team files, reference documents, setup changes, and existing results. |

The folder-only instruction applies to implementation, not generated `runs/`
outputs or the explicitly requested design document. The reference's suggested
DQN settings are the implemented defaults; long main research batches remain a
separate execution stage.

## 2. Components and interfaces

Use an `__init__.py` package marker. Executable entrypoints prepend the absolute
`src/` directory to `sys.path`, as shown in the reference, and use `DQN.*` imports
for learner components. This supports `python src/DQN/train_dqn.py` from the repo
root without copying shared modules or introducing ambiguous local `config.py` files.

| File | Responsibility and interface |
|---|---|
| `model.py` | `QNetwork(obs_size, action_size, hidden_sizes)`; float32 batch input `(B, obs_size)` to raw Q-values `(B, action_size)`. |
| `replay.py` | `ReplayBuffer(capacity, obs_size, action_size, rng)`; `add(obs, action, reward, next_obs, next_mask, terminated)` and `sample(batch_size)`. |
| `agent.py` | Immutable `DQNSettings` for profiles; epsilon schedule, legal epsilon-greedy selection, masked targets, and `DQNAgent.update(batch)` / target synchronization. |
| `policy.py` | Atomic inference checkpoint saving and required `load_policy(checkpoint_path: Path) -> Callable[[np.ndarray, np.ndarray], int]`. |
| `train_dqn.py` | CLI, exact-budget decision loop, shared recorder integration, diagnostics, and cleanup. |
| `run_batch.py` | DQN-local batch/status CLI delegating to the existing scheduler. |
| `tests/` | DQN unit and integration tests, including a server-independent fake environment/recorder. |

Shared imports are `config`, `env.make_env`, `recorder.RunRecorder`,
`recorder.seed_everything`, and the existing analysis/evaluation helpers. No new
third-party dependencies are needed. Input and output dimensions come from the
constructed environment, hidden sizes from `config.NET_ARCH`; validate the current
763/26 contract rather than silently accepting an incompatible encoder.

### Model, replay, and randomness

Use `Linear -> ReLU -> Linear -> ReLU -> Linear`, currently
`763 -> 256 -> 256 -> 26`. No output softmax, output clipping, dropout, input
normalization layer, or extra features. Initialize the target as a deep copy of the
online network, disable its gradients, and use it only for detached targets.

Replay is a preallocated NumPy ring buffer. Store observations and next observations
as float32, actions as int64, rewards as float32, next masks and termination as bool.
Insertion copies values, including masks, so an environment reset or later array
mutation cannot change past transitions. Sample uniformly without replacement once
the populated buffer contains a full minibatch. Return an explicit batch object
with fields `obs, actions, rewards, next_obs, next_masks, terminated`; convert to
CPU tensors with matching dtypes in the learner. Capacity is 100,000. The two
observation arrays require about 610 MB in decimal units, plus other arrays and
process overhead. Do not halve precision or enlarge the buffer for this arm.

Call `seed_everything(seed)` before constructing networks. Spawn two independent
NumPy generators from `np.random.SeedSequence(seed)`: exploration and replay. Greedy
evaluation uses neither. This prevents the evaluation count from advancing the
learner's random streams; it does not claim full determinism of Showdown RNG.

### Settings and profiles

| Parameter | Main | Smoke |
|---|---:|---:|
| Decision steps | `config.TOTAL_STEPS` = 200,000 | 20,000 |
| Opponent | heuristic | maxbp |
| Run suffix | none | smoke |
| Learning starts | 5,000 | 1,000 |
| Epsilon decay decisions | 20,000 | 10,000 |
| Final evaluation | yes, unless explicitly skipped | omitted |

Both profiles use Adam learning rate `1e-4`, betas `(0.9, 0.999)`, epsilon `1e-8`,
zero weight decay; replay 100,000; batch 32; one gradient update per four decisions;
hard target copy every 2,000 decisions; mean Huber loss with delta 1; gradient norm
clip 10. Gamma, device, threads, and hidden sizes come from shared config (currently
0.99, CPU, one thread, `[256, 256]`). Log all effective settings through
`write_run_config`; do not include team/seed/account names in the hyperparameters
dictionary because that dictionary must match across main runs.

CLI: required `--team {lowvar,highvar}` and `--seed` from `config.SEEDS`;
`--profile {main,smoke}` defaults to main; optional `--overwrite` and
`--skip-final-eval`. No arbitrary hyperparameter overrides or resume in v1.
Smoke always uses its fixed opponent, budget, suffix, and warmup. A main run with
final evaluation skipped remains incomplete. Existing results are refused by
default; never enable overwrite automatically in a batch.

### Selection and TD update

With `t` decisions already collected, epsilon is
`1.0 - 0.95 * min(t / decay_steps, 1.0)`. First action is selected at `t=0`.
Exploration samples uniformly from `np.flatnonzero(mask)`. Greedy selection sets
illegal Q-values to negative infinity before argmax; ties choose the first legal
index. Reject an empty acting mask and mismatched observation/mask shapes.

For each sampled transition, calculate:

```text
bootstrap = 0                                      if terminated
            max(target(next_obs)[next_mask])       otherwise
target = reward + config.GAMMA * bootstrap
loss = mean(Huber(online(obs)[action], target))
```

Compute target values under `torch.no_grad()`. Prefer calculating maxima only
for nonterminal rows, initialized with zero bootstrap for terminals. This avoids
terminal `-inf * 0` and supports mixed minibatches. Reject an empty next mask on
nonterminal rows. A time-limit truncation retains `terminated=False`, so its target
bootstraps from the last observation before reset. Record the returned reward
unchanged; do not compute or add another reward.

Update only when `step > learning_starts` and `step % 4 == 0` and enough samples
exist: first update at 5,004 in main, 1,004 in smoke. If update and synchronization
coincide, update online first, then copy it to target. Synchronization uses collected
decision count, not gradient-update count. Monitor finite losses/Q-values; do not
treat the reference's rough expected Q range as a hard clamp.

## 3. Training, outputs, and integration

### Exact lifecycle

1. Validate arguments and profile without creating a run directory or live environment.
2. Seed, construct the training environment with `make_env(..., algo="dqn")`,
   obtain dimensions, initialize learner and replay, then construct the recorder
   with matching team/seed/opponent and profile suffix.
3. Write effective config and evaluate at step zero with deterministic greedy
   inference and the checkpoint callback. Evaluation uses the recorder's separate env.
4. Reset the training env. For steps `1..budget`, get current mask, choose using
   epsilon at `step-1`, step once, and copy the transition into replay before reset.
5. Apply scheduled learning/target updates. On completed or truncated episodes,
   call `episode_end(step, info["battle_end"])` once. Reset only if training remains.
6. After those updates and episode accounting, evaluate/save at each shared
   `EVAL_FREQ`, currently 5,000. Require each profile budget to be divisible by it.
7. At the exact budget, save/evaluate the final weights. Main final evaluation uses
   those same weights, with no intervening training. Do not synthesize termination
   or log an unfinished episode just because the budget ends.
8. Close both environments, recorder files, and DQN diagnostic files in `finally`,
   including when construction partially succeeds. Exceptions propagate to a nonzero
   exit; interruption does not write successful final results or overwrite old output.

Use the shared recorder's field lists and methods without extension or duplication.
Expected main outputs are `runs/dqn_<team>_s<seed>/run_config.json`,
`train_episodes.csv`, `eval.csv`, `final_eval.csv`, `learning_curve.png`, and
`ckpt/step_<step>.pt`. There are 41 in-training evaluation rows/checkpoints from
0 through 200,000. Each evaluates 120 battles, stratified 40/40/40. Final rows
are heuristic 1,020, random 204, maxbp 204. Evaluation decisions do not consume
the training budget. Smoke has five evaluation rows at 0/5k/10k/15k/20k and a
header-only final CSV; its suffix keeps it out of shared research analysis.

### Checkpoint contract

Use one versioned dictionary with primitive metadata and CPU tensors:

```text
format_version: 1
algo: "dqn"
step, team, seed, profile
obs_size, action_size, net_arch, gamma
hyperparameters
q_net: online state_dict
```

Save to a sibling temporary file then replace `step_<step>.pt` atomically.
The path passed by `RunRecorder` has no suffix; the DQN saver adds `.pt`.
Dimensions, step, and seed are converted to native Python integers: Gymnasium's
`Discrete.n` is a NumPy integer, which must not enter restricted-load metadata.
No replay, optimizer, or RNG snapshot is promised: these are inference checkpoints,
not restartable training snapshots. `load_policy` uses CPU and
`torch.load(..., map_location="cpu", weights_only=True)`, checks version/algorithm
and current architecture compatibility, loads strictly, and returns a closure using
`torch.inference_mode()`. Loading must not open an environment, log in to Showdown,
or require training objects. The closure returns a Python int and rejects invalid
mask/observation shapes or an empty mask. Repeated loads and calls choose the same action.

### Diagnostics and research interpretation

Every 1,000 collected decisions, call `rec.log_scalars` with epsilon, replay size,
update count, and available recent loss/mean selected-action Q. Do not log fake loss
values during warmup. At each completed training episode, also write a DQN-owned
`q_returns.csv` with:

```text
step,battle_idx,opp_team,start_step,start_max_q,start_action_q,
discounted_return,terminated,truncated
```

Capture initial legal max Q and chosen-action Q before the first action/update;
accumulate `sum(gamma**k * returned_reward[k])`. Flush per completed episode;
do not report an unfinished budget-cut episode as complete. These training returns
come from an exploring, changing policy. Plotting them against initial max Q is a
diagnostic, not a calibrated estimate of frozen-policy maximization bias.

The standalone loader enables the reference's separately planned frozen-policy
move-usage replay. Do not modify the PPO-only watcher, add a cross-algorithm replay
study, or count requested actions as successful move execution in this first learner.
Actual miss/usage measurements require battle-log events in that later study.
Keep that limitation visible: PPO's pilot sometimes wins without using swapped moves,
so high win rate alone does not establish exposure to the manipulation.

Shared analysis already recognizes `dqn` and seeds from `config.SEEDS`. No analyzer
edit is needed. It does not by itself prove checkpoint loadability or that every
expected run exists; acceptance explicitly checks all ten jobs, every checkpoint,
and the loader in addition to `analyze.load_run`/`check_across_runs`.

### Batch command without shared edits

The DQN-local adapter imports shared `run_batch` after placing `src/` first on the
import path. Register `TRAINERS["dqn"]` to the absolute DQN trainer path in that
process only. Its own argparse exposes `--parallel` (positive int, default 3),
`--status`, and `--allow-dirty`. Dispatch to shared `launch("dqn", ...)` or
`status("dqn")`. It inherits the shared scheduler's job order, logs, and failure
reporting. Do not copy its scheduler or edit the shared registry on disk.

Production batches retain the shared clean-`src/`/`Teams/` requirement. An implementer
must not silently commit, tag, overwrite, or start a long batch. Commit the agreed
implementation separately before research runs. A failed existing run must be moved
aside intentionally before restarting; no resume or automatic deletion is provided.

## 4. Implementation sequence and tests

- [x] Implement model, selection, replay, settings, and update math; add CPU unit tests.
- [x] Implement/version checkpoints and deterministic standalone loader; test round trips.
- [x] Implement trainer, diagnostics, and local batch adapter using shared contracts.
- [x] Test the loop with fake environments and recorder/config fixtures in temporary paths.
- [x] Run shared and DQN tests explicitly together; run live checks with Showdown available.
- [x] Perform the fixed smoke profile and inspect its learning curve and diagnostics.
- [ ] After the code is committed, run/validate all ten main conditions as a separate stage.

Required unit tests:

- Network batch dimensions and raw finite outputs; target starts equal and receives no gradients.
- Greedy and epsilon-random actions always legal for seeded random Q-values/masks;
  deterministic tie handling; epsilon endpoints and independent exploration/replay streams.
- Illegal next Q set to `+1e6` cannot change target; correct mixed-batch terminal and
  truncated targets; all-false terminal mask yields finite reward-only target;
  all-false nonterminal mask raises; gradients flow only through selected online Q.
- Replay copies inputs, wraps at capacity, never samples uninitialized entries,
  preserves dtypes/shapes, and samples reproducibly from its own RNG.
- Learning starts/sync timing; online changes after an update and target changes
  only at synchronization; finite Huber loss and clipping behavior.
- Checkpoint round-trip equality and legal deterministic inference; malformed
  metadata, incompatible dimensions, and corrupt files fail clearly; no live server needed.

Required integration tests:

- A fake env gives distinct terminal/reset observations: replay stores the terminal
  observation and mask, and truncation bootstraps from its pre-reset observation.
- Fake runs stop at the exact budget; eval at zero and frequency boundaries;
  updates precede checkpoint capture; final evaluation uses identical last weights;
  evaluation never advances decision count or exploration/replay generators.
- Episode rows are emitted once; budget-cut episodes are not fabricated;
  resources close on success and exceptions; existing directories are refused.
- Temporary recorder outputs retain the exact shared column order and pass
  `analyze.load_run`; ten synthetic main-run identities exercise seeds 3/4 too.
  Verify grid completeness and all expected loadable checkpoints separately.
- Local batch adapter selects the DQN trainer, schedules the shared ten jobs, and
  propagates failures using mocked subprocesses; original shared files remain unchanged.

Live acceptance: correct `(763,)` observations, legal masked actions, real completed
battles, no Tera, finite updates, and near-zero discounted shaping sum for naturally
completed battles. Smoke must produce all five evaluation/checkpoint points and
show a visible increase in evaluation performance versus MaxBasePower. Report the
actual values and intervals; learning improvement is empirical, not guaranteed by
passing unit tests. If it fails, investigate correctness before proposing tuning.
Do not quietly change the preselected main hyperparameters.

## 5. Operating commands

Run from the repo root. Activate Node 22 in shells that run Showdown or Node-based tests.

```bash
# Terminal 1: local server, leave running
source "$HOME/.nvm/nvm.sh"
nvm use 22
cd pokemon-showdown
node pokemon-showdown start 8000 --no-security
```

```bash
# Terminal 2: shared + DQN verification
source "$HOME/.nvm/nvm.sh"
nvm use 22
source venv/bin/activate
python src/check_teams.py
python -m pytest -q tests src/DQN/tests

# Smoke: fixed 20k / maxbp / shorter warmup, separate output name
python src/DQN/train_dqn.py --team lowvar --seed 0 --profile smoke

# One main run: fixed 200k / heuristic
python src/DQN/train_dqn.py --team lowvar --seed 0

# All ten main runs: separate research execution after committing code
python src/DQN/run_batch.py --parallel 3
python src/DQN/run_batch.py --status

# Analyze without replacing the currently saved figures
python src/analyze.py --partial --out runs/dqn_analysis_preview
```

The root pytest configuration discovers `tests/` only. Always explicitly include
`src/DQN/tests` when verifying DQN; do not edit that shared configuration just to
add discovery. The historical reference's Python 3.14 note describes another
machine; the inspected local venv is Python 3.12.3. Recorder metadata is authoritative
for the actual runtime. Do not change Python versions merely to match that old note.

## 6. Implementation and validation record

Validation on 2026-10-20, Python 3.12.3, torch 2.14.1, poke-env 0.16.1,
Node 22.23.3, Showdown revision `51ad80fa5b264a28416b0d78667519a97ba2d2ee`:

- `python -m pytest -q src/DQN/tests`: **26 passed**.
- `python -m pytest -q tests src/DQN/tests` with Showdown running: **104 passed**.
- CLI help and DQN batch status work; no main run was launched by these checks.
- Fixed LowVar/seed-0 smoke command:
  `python src/DQN/train_dqn.py --team lowvar --seed 0 --profile smoke`.

| Decision step | Wins / 120 | Win rate | Wilson 95% interval |
|---:|---:|---:|---:|
| 0 | 21 | 17.5% | 11.7–25.3% |
| 5,000 | 74 | 61.7% | 52.7–69.9% |
| 10,000 | 107 | 89.2% | 82.3–93.6% |
| 15,000 | 112 | 93.3% | 87.4–96.6% |
| 20,000 | 119 | 99.2% | 95.4–99.9% |

The smoke took 278.6 seconds through its final evaluation and completed 1,179
training battles. The last completed episode ended at step 19,997; the final three
decisions belong to an unfinished episode and are correctly absent from episode
CSVs. There were 4,750 gradient updates and no time-limit truncations. The largest
absolute discounted shaping residual was `5.56e-16` (rounded upward).

All five evaluation checkpoints, shared CSV schemas, scalar diagnostics, episode
indices, exact decision budget, and the generated learning-curve image were checked.
The final loaded checkpoint also won a real MaxBasePower battle with legal actions.
The validation server was stopped after testing; use the startup command above
before launching another training run.

The first offline checkpoint check exposed NumPy integer metadata from Gymnasium.
The saver now normalizes that metadata, and the regression fixture uses NumPy
dimensions/step/seed to exercise the real case. The five task-generated development
checkpoints were reserialized with primitive metadata; tensor equality was checked
exactly, so the measured training weights and evaluation results did not change.
Original development files were backed up under `/tmp/pokemon-dqn-smoke-before-metadata-fix/`.
The public loader remains restricted to `weights_only=True`; it has no unsafe fallback.

These are smoke results against MaxBasePower with the smoke profile, not main
heuristic-opponent results or evidence for the LowVar/HighVar research question.

The implementation is not committed by this task. Recorder metadata truthfully
marks this development smoke run as dirty. Main batches retain the shared clean-code
check; commit the agreed code before starting research runs.
