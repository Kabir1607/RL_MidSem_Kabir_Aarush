# Results: PPO on LowVar vs HighVar (4v4 singles, open team sheets)

PPO only. The DQN arm (old RQ2) is not part of this report.

## Summary

1. **PPO learns to beat the heuristic opponent almost perfectly, on both teams.** All 10 runs end at
   99.7–100% win rate over 1,020 final evaluation battles, against a heuristic-piloted baseline of
   about 43%. Most of the learning happens in the first 50–75k decision steps.
2. **The team version makes no measurable difference to how fast PPO learns (RQ1).** AUC is 0.907 for LowVar
   and 0.906 for HighVar (p = 0.91). The first 50k steps, the steps until win rate holds above 0.95, and the
   spread between seeds are also the same for both teams.
3. **Final strength does not differ (RQ3).** Both teams hit the 100% ceiling, so this comparison cannot
   tell them apart.
4. **The only hint of a difference is stability, and it is not significant.** After 60k steps, HighVar runs
   had 1.4 evaluation dips below 0.95 per run, against 0.4 for LowVar (p = 0.14, 5 seeds per team).
5. **The most likely reason for the null result is that the winning line does not need the swapped moves.**
   PPO beats this deterministic heuristic with a narrow line. In the battles we watched, it was a Kingambit
   Swords Dance sweep that used no swapped moves. If the risky moves are not used, their accuracy cannot
   affect learning. This is not yet measured over many battles (see Limitations).

## Setup

| | |
|---|---|
| Format | Gen 9 OU singles, Terastal Clause (`gen9ounotera`), 4v4 with team preview, fixed lead order |
| Our teams | `Teams/lowvar.txt`, `Teams/highvar.txt`: identical except for 4 swapped moves (below) |
| Opponents | Teams A, B, C (`Teams/opp_*.txt`), piloted by poke-env's `SimpleHeuristicsPlayer` |
| Information | Open team sheets: the full sets of both sides are visible; 763-feature observation, 26 actions with legality masks |
| Reward | ±1 for a win or loss, plus potential-based shaping on HP and fainted Pokémon (γ = 0.99, Φ(terminal) = 0) |
| Algorithm | MaskablePPO (sb3-contrib 2.9.0): lr 3e-4, n_steps 2048, batch 128, 10 epochs, clip 0.2, entropy 0.01, MLP [256, 256] |
| Budget | 200,000 decision steps per run (≈ 15,300 training battles) |
| Runs | 2 teams × 5 seeds (0–4) = 10 runs, commit `60ba557` (tag `ppo-batch-2`) |
| Evaluation | Every 5,000 steps: 120 greedy battles vs the heuristic, 40 against each opponent team |
| Final evaluation | At 200k: 1,020 battles vs the heuristic (340 per team), and 204 each vs Random and MaxBasePower |
| Compute | CPU, 3 runs in parallel, ≈ 0.40 h per run, 1.55 h for the batch |

Swapped moves (the manipulation):

| Pokémon | LowVar | HighVar |
|---|---|---|
| Moltres | Flamethrower 90/100 | Fire Blast 110/85 |
| Moltres | Air Slash 75/95 | Hurricane 110/70 |
| Primarina | Surf 90/100 | Hydro Pump 110/80 |
| Primarina | Ice Beam 90/100 | Blizzard 110/70 |

Heuristic baseline (`baselines_summary.csv`, our team piloted by the same heuristic, 1,500 battles):
LowVar 0.429 [0.404, 0.454], HighVar 0.437 [0.412, 0.463]. Per opponent: A ≈ 0.05–0.08, B ≈ 0.98–1.00,
C ≈ 0.24–0.25. The heuristic misses its swap moves 2% of the time with LowVar and 23% with HighVar, yet its
win rate is the same for both teams.

## Where the graphs are

The aggregate figures are in `figures/` and were produced by `src/analyze.py` from the 10 runs.

| File | What it shows | Use for |
|---|---|---|
| `figures/curves_ppo.png` | **Main figure.** Eval win rate vs decision steps: mean of 5 seeds per team (thick), each seed (thin), heuristic baselines (dashed) | RQ1, headline |
| `figures/interaction_auc.png` | AUC per seed (dots) and mean (line), LowVar vs HighVar | RQ1 |
| `figures/ep_len.png` | Training battle length (rolling 200 battles) per run | Supporting: battles get shorter as the policy improves |
| `figures/threshold_steps.png` | Steps until win rate first reaches baseline + 0.10 | Not recommended: the metric degenerates (see below), and the DQN columns are empty |
| `figures/interaction_lift.png` | Final win rate minus baseline | Not recommended: shows only the 100% ceiling |

The aggregate tables are in the same folder:

- `figures/summary.csv`: every metric for every run, plus the mean and std for each team.
- `figures/contrasts.csv`: the LowVar vs HighVar differences.
- `figures/runs_index.csv`: run status and the commit each run used.

The per-run graphs are in `runs/ppo_<team>_s<seed>/learning_curve.png`, for example
`runs/ppo_highvar_s3/learning_curve.png`. Each shows the eval win rate with its Wilson 95% band, the baseline
and threshold, training battle length and the critic's explained variance. The raw data is in the same
folder: `eval.csv`, `final_eval.csv` and `train_episodes.csv`.

The 2M-step pilot (3 runs) is in `runs/pilot_2M/<run>/learning_curve.png`.

The DQN runs are also in `runs/`, so running `src/analyze.py` again now would add DQN to the figures. To
regenerate the PPO-only figures, point `--runs-dir` at a folder that holds only the `ppo_*` runs, and use
`--out` so the current figures are not overwritten.

## Results

### RL vs the heuristic

| | LowVar | HighVar |
|---|---|---|
| Eval win rate at step 0 (untrained), mean of seeds | 0.21 | 0.19 |
| Training-battle win rate, first 10k steps | 0.08–0.13 | 0.07–0.12 |
| Training-battle win rate, last 50k steps | 0.96–0.99 | 0.94–0.98 |
| Final win rate vs heuristic (1,020 battles), mean | 0.9998 | 0.9994 |
| RL lift (final − baseline) | +0.571 | +0.562 |

Every run is far above its baseline. Every final Wilson lower bound is ≥ 0.991. The 0.009 gap in lift comes
entirely from the two baselines (0.429 vs 0.437), not from any difference between the agents.

### RQ1: learning speed and disagreement between seeds

Mean ± std over 5 seeds. p is from an exact two-sided permutation test (all 252 ways of splitting the 10
runs into two groups of 5).

| Metric | LowVar | HighVar | p |
|---|---|---|---|
| AUC, 0–200k (primary metric) | 0.907 ± 0.022 | 0.906 ± 0.020 | 0.91 |
| AUC, 0–50k (learning phase) | 0.667 ± 0.090 | 0.670 ± 0.075 | 0.96 |
| Steps until ≥ 0.95 holds for 3 evaluations in a row | 50k ± 12k | 52k ± 20k | 0.95 |
| Steps to baseline + 0.10, 2 in a row (spec threshold) | 16k ± 8k | 13k ± 9k | 0.77 |

- **Learning speed is the same.** The AUC gap (−0.001) is about 20 times smaller than the spread between
  seeds. The seed-by-seed differences go both ways: HighVar is higher for seeds 0, 1 and 2, LowVar for
  seeds 3 and 4.
- **Seeds disagree about equally on both teams.** The std of AUC is 0.022 vs 0.020. The std of the steps
  to 0.95 is larger for HighVar (20k vs 12k), but one slow seed (s4, 75k) drives that.
- **The spec's threshold metric is not informative here.** The threshold is about 0.53, and every run
  crosses it within 30k steps. Seed 1's untrained network already scores 0.62–0.67 at step 0 on both teams,
  so one run "reaches" it at step 0.

### Learning order by opponent

Steps until each opponent team is beaten at ≥ 0.95 in 2 evaluations in a row, per seed (s0–s4):

| | A | B | C |
|---|---|---|---|
| LowVar | 10k, 10k, 10k, 10k, 25k | 30k, 25k, 25k, 5k, 25k | 30k, 55k, 55k, 30k, 10k |
| HighVar | 20k, 20k, 15k, 20k, 40k | 35k, 50k, 20k, 15k, 25k | 35k, 55k, 50k, 30k, 60k |

- **PPO learns the opponents in the same order on every run: A first, then B, then C.** C is the bottleneck.
  The plateau at 0.667 in `curves_ppo.png` is the stage where A and B are beaten every time and C never is.
- **A is the reverse of the baseline.** The heuristic pilot wins only 5–8% against A, yet RL beats A first.
- **HighVar beat A a little later** (mean 23k vs 13k steps), but p = 0.13. We looked for this after seeing
  the data, so treat it as exploratory only.

### Stability after convergence

| Metric (evaluations from 60k steps on) | LowVar | HighVar | p |
|---|---|---|---|
| Dips below 0.95 per run | 0.4 | 1.4 | 0.14 |
| Mean eval win rate | 0.993 | 0.987 | 0.14 |
| Largest drop between consecutive evaluations, whole run | 0.22 ± 0.16 | 0.25 ± 0.12 | 1.00 |

Each late dip has the same shape: one opponent collapses while the other two stay at 100%, and it recovers
by the next evaluation.

| Run | Step | vs A | vs B | vs C |
|---|---|---|---|---|
| ppo_lowvar_s1 | 125k | 1.00 | 1.00 | 0.00 |
| ppo_lowvar_s3 | 85k | 1.00 | 1.00 | 0.00 |
| ppo_highvar_s1 | 65k / 120k | 1.00 | 1.00 | 0.63 / 0.08 |
| ppo_highvar_s2 | 100k / 195k | 1.00 / 0.13 | 1.00 | 0.55 / 1.00 |
| ppo_highvar_s3 | 160k, 165k | 1.00 | 0.25, 0.20 | 1.00 |
| ppo_highvar_s4 | 70k | 1.00 | 1.00 | 0.65 |

This fits a policy that wins with one narrow line per matchup, where a small update can break that line for
one opponent. The ppo_highvar_s3 dip also shows up as a rise in battle length around 150–175k in
`ep_len.png`. HighVar dips more often, in the same direction as the 2M pilot (where HighVar s0 dipped and
the LowVar runs barely did). With 5 seeds it is a hint, not a result.

### RQ3: final strength

| Final evaluation (mean of 5 seeds) | LowVar | HighVar | p |
|---|---|---|---|
| vs heuristic (1,020 battles) | 0.9998 | 0.9994 | n/a (ceiling) |
| vs RandomPlayer (204) | 0.979 ± 0.012 | 0.965 ± 0.016 | 0.17 |
| vs MaxBasePowerPlayer (204) | 0.926 ± 0.030 | 0.918 ± 0.078 | 0.86 |

- **No difference in final strength.**
- **The policies are specialised to the heuristic.** They were trained only against it, and they do worse
  against weaker opponents they never saw. Against MaxBasePower, team A is the weak spot: as low as 0.41
  (HighVar s4) and 0.69 (HighVar s0), against ≥ 0.88 for B and C. The agent learned how to beat this
  particular opponent, not how to play the matchups well in general.

### Battle length

Training battles get shorter as the policy improves: about 18 decision steps in the first 10k steps, and
10.5–14 steps (9.5–12.4 turns) in the last 50k. The heuristic pilot's battles average 10.4 turns. No
battle hit the 300-step cap. In every battle, the discounted sum of the shaping terms is 0 to within 5e-16,
as the potential-based design requires.

### Pilot: 2M steps, 3 runs (`runs/pilot_2M/`)

All 3 runs (LowVar s0, HighVar s0, LowVar s1) reached 100% eval win rate by 30k–80k steps and stayed there
until they were stopped at about 870k steps. This is why the main budget was cut to 200k. HighVar s0 dipped
after first reaching 1.0: to 0.65 at 30k and 0.79 at 70k, both against team C.

## Interpretation

PPO beats the heuristic within about 50–75k decision steps, whichever team version it uses. With this setup
we find no effect of move variance on learning speed, disagreement between seeds or final strength. The
only direction we see is a possible effect on stability after convergence (more matchup collapses on
HighVar), and it is not significant with 5 seeds.

We should not conclude from this that "outcome variance does not affect PPO". Three features of the setup
limit what this experiment can show:

1. **Ceiling.** The opponent is deterministic and exploitable, so both teams reach 100%. Differences can
   only appear during the first ~50k steps of learning and in stability.
2. **Exposure to the manipulation is unknown.** The manipulation only matters if the agent uses the swapped
   moves. In the battles we watched (`src/watch_battle.py`), the agent sacrificed its lead and swept with
   Kingambit Swords Dance, never using a swapped move. If that is typical, both teams effectively play the
   same game.
3. **The baseline also shows no effect.** Even the heuristic, which misses swap moves 23% of the time on
   HighVar against 2% on LowVar, has the same win rate with both teams (0.437 vs 0.429).

## Limitations

- **Small samples.** 5 seeds per team; the p-values are exact permutation tests, and none is below 0.1.
  The per-opponent and stability analyses were done after seeing the data, so they are exploratory.
- **Each evaluation is noisy.** 120 battles per evaluation gives a Wilson half-width of up to ±0.09 at a
  50% win rate. Final evaluations (1,020 battles) are much tighter.
- **Move usage is anecdotal.** It comes from watched battles, not measured. Replaying the final checkpoints
  with battle logs would give the swap-move usage and miss rates directly. This has not been done yet.
- **One opponent policy.** All training and in-training evaluation is against a single deterministic
  heuristic.

## Reproducing

- Training: `src/run_batch.py --algo ppo --parallel 3` at commit `60ba557` / tag `ppo-batch-2`. Every run's
  `run_config.json` records its exact config and commit. 7 runs record `0720f93`, which added only result
  files, so the code is identical.
- Figures and tables: `src/analyze.py` (see the note on DQN runs in "Where the graphs are").
- The per-opponent learning steps, late-dip counts, AUC over 0–50k and permutation tests in this file were
  computed directly from each run's `eval.csv` and `final_eval.csv`.
- To watch a trained agent play: `src/watch_battle.py`.
