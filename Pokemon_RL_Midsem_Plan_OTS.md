# **Pokémon RL Midsem Plan** (revised: open team sheets)

## **Changelog for this revision (18 Oct, afternoon)**

This version adds an **open-team-sheet (OTS) information setting**: from turn 1 the agent knows every Pokémon on both teams with its species, item, ability, moves, nature, EVs, IVs and level, and therefore its real stats. Everything below that changed because of this is marked **[OTS]**.

| Section | What changed |
| :---- | :---- |
| Intro, 1 | OTS added to the setup; new framing of what "noise" now means |
| 2 | New subsection 2b: what OTS is, how we implement it, and why it makes the variance experiment *cleaner* |
| 3 | New team rules: distinct opponent species sets, explicit EVs/IVs/nature in the files, no Illusion/Transform |
| 4 | State space rewritten (≈150–250 → ≈1,100–1,300 features); partial-observability simplification replaced |
| 5 | Two new predictions; one existing prediction re-argued |
| 6 | RL lift now includes an information advantage; caveat added |
| 7 | New task T6a (team-sheet loader and stat calculator); T1, T4, T6, T8, T9, T12, T16, T17, T20 adjusted |
| 8 | New risks; critical path grows by about 1.5–2.5 h |
| 9 | Official VGC OTS hides EVs/IVs/nature; the final must choose between "full sheets" and "official OTS" |

**Terminology warning.** In real VGC, an open team sheet shows species, Tera type, item, ability and moves, but **not** EVs, IVs or nature (so not exact stats). What this plan uses is stronger: a **full team sheet**. We still call it OTS for short, but the slides should say "full open team sheets (including spreads)" so nobody in the audience assumes the official rule.

## **Introduction**

We want to measure how an RL algorithm learns different *playing strategies* in doubles VGC. Specifically, we compare a standard team with a high-risk, high-reward team such as Perish Trap.

That difference is hard to quantify directly. Perish Trap differs from a standard team in many ways at once: which moves matter, multi-turn timing, trapping and when to switch. Any result would be hard to attribute to a single cause.

So the midsem starts with a simpler difference that is easy to show and fully controlled: **variance**. We take one 6v6 singles team and build two versions that are identical except for a few moves. In the high-variance version, those moves are swapped for lower-accuracy, higher-base-power versions (e.g. Surf → Hydro Pump). Pokémon, items, abilities, EVs, opponents and algorithm settings all stay the same.

**[OTS]** Both sides play with **open team sheets**: the agent knows the full set of every Pokémon on both teams from the start of the battle. This removes almost all *hidden information*, so the remaining unpredictability the agent faces is mostly the game's random number generator (accuracy rolls, damage rolls, crits, secondary effects). That is exactly the thing the move swap manipulates, so OTS makes the experiment cleaner (Section 2b). It also matches VGC, where open team sheets are standard, so it carries into the final.

We then study how this one change affects *how the agent learns*, using three measures:

* **Sample efficiency:** how much experience it needs to get good.  
* **Stability:** how consistent learning is across random seeds and over training.  
* **RL lift:** how much the trained agent improves over a scripted player using the same team.

We do this for two kinds of RL, on both team versions:

* **DQN (deep Q-learning):** value-based; it learns a score for each move.  
* **PPO:** policy-based; it learns move probabilities directly.

For the midsem, the only opponent is poke-env's built-in heuristic player. The final moves to 4v4 doubles VGC with the standard vs Perish Trap comparison. It reuses this pipeline and these metrics, and adds existing bots as opponents (Section 9).

## **1\. Goal and research questions**

The midsem trains DQN and PPO agents on one 6v6 Gen 9 OU singles team in two versions that differ only in move accuracy and power, and compares how fast and how reliably each algorithm learns each version. **[OTS]** The agent observes both teams' full sets from turn 1. Presentation: Wed 21 Oct 2026\.

* **RQ1 (variance → learning):** Does the high-variance (inaccurate-move) team need more training battles to reach the same performance, and do its runs disagree more across seeds?  
* **RQ2 (algorithm × variance):** Does DQN degrade more than PPO on the high-variance team? Theory predicts yes (Section 5).  
* **RQ3 (strength):** Does the final win rate differ? This is secondary, because the swap also changes damage thresholds (Section 2).  
* **Deliverable 2 (spec):** MDP, value function and policy definitions that carry over to the 4v4 VGC final.

Honest framing for the slides: this is a pilot. Expect 2–3 seeds per condition and wide error bars. The deliverable is a working pipeline, a sound protocol and first evidence, not a definitive result.

**[OTS] Not a research question for the midsem:** "does OTS help the agent learn faster?" Answering that needs a second encoder without OTS and doubles the number of runs. Keep it as an optional ablation for the final (Section 9).

## **2\. Experimental design and verification of the move swap**

Verdict: the swap works and is a much cleaner experiment than two different teams. It is not a pure variance change, though, so the slides must report the expected-damage shift and control for it.

**Setup**

* Gen 9 singles, 6v6, preset teams, a fixed lead (slot 1\) and no team-preview decision.  
* Tera disabled for **both** players, so the action space is moves plus switches only.  
* **[OTS]** Full open team sheets for our agent: species, item, ability, moves, nature, EVs, IVs and level of all 12 Pokémon are part of the observation from turn 1 (Section 2b).  
* Our team comes in two versions, LowVar and HighVar, identical in species, items, abilities, EVs, natures and every non-swapped move.  
* 3 preset opponent teams, piloted only by poke-env's SimpleHeuristicsPlayer. Existing strong bots are for the final (Section 9).  
* Each training episode draws an opponent team uniformly at random. This is identical for all conditions.

**Why the swap is good design.** Holding everything else fixed means any difference in learning comes from the changed moves. In stats terms the moves are the single manipulated variable. The 4 untouched Pokémon act as within-team controls.

**Candidate swap pairs and their expected damage (base power × accuracy).** You pick the pairs you'll actually use in Section 3\. If you use a pair not listed here, compute its expected BP the same way.

| Low-variance move | High-variance move | Expected BP | Change | Other differences |
| :---- | :---- | :---- | :---- | :---- |
| Surf 90 / 100% | Hydro Pump 110 / 80% | 90 → 88 | −2% | PP 24 → 8 |
| Flamethrower 90 / 100% | Fire Blast 110 / 85% | 90 → 93.5 | \+4% | PP 24 → 8; burn chance 10% in both |
| Thunderbolt 90 / 100% | Thunder 110 / 70% | 90 → 77 | −14% | Paralysis 10% → 30%; PP 24 → 8 |
| Ice Beam 90 / 100% | Blizzard 110 / 70% | 90 → 77 | −14% | Freeze 10% in both; PP 24 → 8 |

PP values are maxed (PP Ups); verify in the teambuilder.

**Caveats to state on the slides**

1. **Not mean-preserving.** Thunder and Blizzard also lower expected damage by 14%. Hydro Pump and Fire Blast are close to a pure variance change.  
2. **KO thresholds.** 110 BP can turn a 2HKO into an OHKO, so in some matchups the inaccurate move is *better* in expectation. Raw win rate therefore mixes strength and variance. The fix is the RL-lift metric in Section 6: compare each RL agent to a heuristic piloting the same team version. **[OTS]** With known spreads you can now check this *exactly* instead of guessing: run each swapped move against each opponent Pokémon in a damage calculator (Showdown's calc, or poke-env's `calculate_damage`) and put a small "KO thresholds that change" table on a slide.  
3. **Weather breaks the manipulation.** Snow makes Blizzard 100% accurate. Rain makes Thunder 100% accurate, and sun drops it to 50%. Opponent teams must contain no weather setters. Slowking-Galar's Chilly Reception sets snow and it is a top-20 OU Pokémon, so exclude it.  
4. **Accuracy modifiers** also break it: Compound Eyes, No Guard, Wide Lens, Bright Powder, accuracy or evasion boosts. None of these may appear on any team.  
5. **PP.** 8 PP rarely runs out, but 6v6 battles can go 40+ turns, so log it.  
6. **Secondary effects.** Thunder's higher paralysis chance is a second source of randomness. It's acceptable because it is also variance, but name it.

**Optional cleaner sub-test (final project).** A "strict" HighVar team that swaps only Surf→Hydro Pump and Flamethrower→Fire Blast (nearly mean-preserving). The effect will be small, though, and probably not detectable with midsem seed counts.

**A subtle point about "variance".** A win/loss outcome has variance p(1−p), which depends only on the win rate. So "high variance" here does not mean the final result is more random. It means each *transition* is noisier: the same state and action lead to more spread-out next states. That is exactly what makes TD targets and policy-gradient estimates noisier, which is the mechanism we test.

### **2b. [OTS] Information setting: open team sheets**

**What the agent knows from turn 1.** For all 12 Pokémon (ours and the opponent's): species, types, item, ability, all 4 moves, nature, EVs, IVs and level, and therefore all 6 actual stats. During the battle it still has to *observe* everything that changes: HP, status, boosts, which Pokémon is active, PP used, items consumed or knocked off, field conditions.

**Why this makes the variance experiment cleaner.** From the agent's point of view, a transition from state s with action a can land in different next states for three reasons:

1. **Hidden information** (without OTS): the opponent's unrevealed Pokémon, moves, items and spreads. The same observation can hide different true situations.  
2. **The opponent's choice:** what the opponent does this turn. SimpleHeuristicsPlayer is close to deterministic given the battle state (verify in its source; there may be random tie-breaks), so this source is small for the midsem.  
3. **Game RNG:** accuracy rolls, damage rolls (85–100%), crits, secondary effects, speed ties.

OTS removes almost all of source 1. With a near-deterministic opponent, what remains is mostly source 3, which is precisely what the LowVar/HighVar swap changes. So a difference between team versions is now easier to attribute to RNG rather than to "the agent also didn't know what it was facing". This is worth one slide: it is the main *scientific* reason for OTS, not just convenience.

In RL vocabulary: without OTS the problem is a POMDP (partially observable MDP). With OTS and a few small counters in the state (Section 4) it becomes close enough to an MDP that a memoryless policy is a reasonable approximation, rather than an unexamined simplification.

**How we implement it (two routes).**

* **Route A — lookup from our own team files (recommended for the midsem).** We wrote the opponent teams, so we already have their full sets. At team preview, Showdown tells both players the opponent's 6 species; poke-env stores them on the battle object (check the exact attribute in your installed version, something like `teampreview_opponent_team`). Match that species set against the 3 opponent team files, then load the full sets for that team. Our own sets come from our team file.  
  * Works regardless of whether the format has an OTS rule.  
  * Gives everything, including EVs, IVs and nature.  
  * Requires that the 3 opponent teams have different species sets (Section 3 rule). For safety, also give them 3 different leads, so the lead species alone identifies the team if team preview isn't available in your format.  
* **Route B — Showdown's own open-team-sheet mechanism.** poke-env's Player has an `accept_open_team_sheet` property, so the library knows about Showdown's OTS handshake (seen in the poke-env API index; not tested). But this follows the **official** rule, which does not reveal EVs, IVs or nature, and it may only be enabled in VGC formats. Not useful for the midsem; relevant for the final (Section 9).

**Who gets the information (an asymmetry to state honestly).** With Route A, only our RL agent gets the full sheets. The heuristic opponent, and the heuristic pilot used for the RL-lift baseline, play with normal revealed-only information. Consequences:

* The asymmetry is **identical in all 4 cells** (both algorithms, both team versions), so the LowVar vs HighVar and DQN vs PPO comparisons are unaffected.  
* **Absolute** RL lift is inflated, because part of it is an information advantage rather than learning (Section 6).  
* How much this matters depends on what SimpleHeuristicsPlayer actually uses. My understanding is that it reasons from types, base stats and boosts, not EV spreads, so it might barely change with full information, but I have not verified this; read its source.

## **3\. Teams (you fill these in)**

You choose the teams and exact sets. Do it in this order:

1. Pick the swap pairs.  
2. Build our team around Pokémon that legally learn *both* moves of a pair.  
3. Build the 3 opponent teams.

Check every set with the Showdown teambuilder validator for the format you choose in task T5; illegal moves are flagged there. Current usage data to draw from: [PokeTools Gen 9 OU, September 2026](https://www.poketools.com/meta/gen-9-ou).

**[OTS]** The team files are now also **the encoder's source of truth** for the static part of the state. A typo in an EV spread is no longer just a weaker set; it is a wrong observation. Hence the extra rules below and the stat check in T4.

### **Swap pairs (pick 3–4)**

Expected BP \= base power × accuracy. Pairs with similar expected BP isolate variance best (Section 2). Candidates are in the Section 2 table.

| \# | LowVar move (BP / acc) | HighVar move (BP / acc) | Expected BP, low → high | Used by |
| :---- | :---- | :---- | :---- | :---- |
| 1 |  |  |  |  |
| 2 |  |  |  |  |
| 3 |  |  |  |  |
| 4 |  |  |  |  |

### **Our team**

Both versions are identical except the swapped move. For control Pokémon (no swap), list all 4 moves under Shared moves and leave the last two columns as —. Slot 1 is the lead in every battle. **[OTS]** Fill in IVs as well (write "31 all" if so) and the resulting stats, as shown in the teambuilder.

| Slot | Pokémon | Item | Ability | Nature / EVs / IVs | Stats (HP/Atk/Def/SpA/SpD/Spe) | Shared moves | LowVar move | HighVar move |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| 1 (lead) |  |  |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |  |  |
| 4 |  |  |  |  |  |  |  |  |
| 5 |  |  |  |  |  |  |  |  |
| 6 |  |  |  |  |  |  |  |  |

**Rules for our team**

* At least 3 of the 6 Pokémon carry a swap. The rest act as controls.  
* No weather setters (Drizzle, Drought, Snow Warning, Chilly Reception, Rain Dance, Sunny Day, Snowscape): they change Thunder, Blizzard and Hurricane accuracy.  
* No accuracy or evasion modifiers: Compound Eyes, No Guard, Hustle, Victory Star, Wide Lens, Zoom Lens, Bright Powder, evasion- or accuracy-boosting moves.  
* Avoid swap moves with big self-drawbacks (Draco Meteor, Leaf Storm, High Jump Kick crash damage). They change strategy, not just variance.  
* **[OTS]** No Illusion (Zoroark, Zoroark-Hisui) and no Transform/Imposter (Ditto, Mew with Transform): they make the displayed Pokémon differ from the sheet, which breaks both the lookup and the "known identity" assumption.  
* **[OTS]** Avoid Trick and Switcheroo on all teams. Knock Off is fine and too common to ban, but the encoder must then track the *current* item (Section 4), not just the sheet's item.

### **Opponent teams (3)**

Aim for 3 different styles (e.g. balance, offense, bulky). Same weather, accuracy and **[OTS]** Illusion/Transform rules as above, and no Pokémon that is also on our team.

**[OTS] Identification rule:** the 3 opponent teams must have **different species sets** and **different leads**, so team preview (or the lead alone) tells the encoder which sheet to load.

**Opponent A — style:**

| Pokémon | Item | Ability | Moves | Nature / EVs / IVs | Stats |
| :---- | :---- | :---- | :---- | :---- | :---- |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |

**Opponent B — style:**

| Pokémon | Item | Ability | Moves | Nature / EVs / IVs | Stats |
| :---- | :---- | :---- | :---- | :---- | :---- |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |

**Opponent C — style:**

| Pokémon | Item | Ability | Moves | Nature / EVs / IVs | Stats |
| :---- | :---- | :---- | :---- | :---- | :---- |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |

### **Opponent policy (midsem)**

Training uses only poke-env's SimpleHeuristicsPlayer (check the exact name in your installed version). Evaluation also includes random and max-base-power players as simple sanity baselines. These are not "existing bots"; those come in the final.

### **Diff check**

A 10-line script parses both team files and asserts that only the chosen swap moves differ. Put its output on a slide: it is the proof that everything else was held constant. **[OTS]** Extend it to also assert that the 3 opponent teams have pairwise different species sets and leads.

**If you want a default to start from:** Kingambit and Great Tusk as controls; Primarina (Surf ↔ Hydro Pump), Kyurem (Ice Beam ↔ Blizzard), Raging Bolt (Thunderbolt ↔ Thunder) and Moltres (Flamethrower ↔ Fire Blast) as treated. These are based on September 2026 OU usage, but the learnsets are from memory and unverified.

## **4\. MDP specification: value function and policy**

This is the formal part of the midsem deliverable. The same definitions extend to doubles in the final; only the action space and some features change.

### **State s [OTS: rewritten]**

The observation is a fixed-length vector of roughly **1,100–1,300 numbers**, all scaled to about \[0, 1\] (or \[−1, 1\] where noted). It is built from three blocks.

**Indexing rule (important).** Our 6 Pokémon always occupy the same 6 rows in team-file order, and the opponent's 6 Pokémon occupy rows in *their* sheet's order, keyed by species. Do **not** rely on the order in which poke-env stores `battle.team` or on Showdown's request order, which moves the active Pokémon around (I have not verified poke-env's ordering; keying by species avoids the question). With OTS all 6 opponent rows are filled from turn 1 instead of being zero until revealed.

**Block 1 — per-Pokémon rows (12 rows: 6 ours, 6 theirs).** Each row has a *static* part (from the team sheet) and a *dynamic* part (from the battle).

| Feature | Size | Source | Scaling |
| :---- | :---- | :---- | :---- |
| HP fraction | 1 | battle | already \[0, 1\] |
| Fainted flag | 1 | battle | 0/1 |
| Status one-hot (none / brn / par / psn / tox / slp / frz) | 7 | battle | 0/1 |
| Active flag | 1 | battle | 0/1 |
| **Actual stats** HP, Atk, Def, SpA, SpD, Spe | 6 | sheet (computed from base stats, nature, EVs, IVs, level) | divide each by the max of that stat over all 24 Pokémon in the 4 team files |
| Types (multi-hot over 18) | 18 | sheet | 0/1 |
| Current item (one-hot over the item vocabulary of the 4 team files, plus "none/removed") | I + 1 | sheet, updated by battle (Knock Off, consumed berries, Booster Energy) | 0/1 |
| Ability (one-hot over the ability vocabulary of the 4 team files) | B | sheet | 0/1 |
| For each of its 4 moves: base power/150, accuracy (never-miss → 1.0), category one-hot (physical / special / status), priority (/5, in \[−1, 1\]), type effectiveness vs the **opposing active** Pokémon (/4), STAB flag, PP fraction | 4 × 9 = 36 | sheet (moves, BP, accuracy) + battle (PP, opposing active) | as listed |

Row size = 70 + I + B. With an item vocabulary I ≈ 15 and ability vocabulary B ≈ 15 that is about 100, so 12 rows ≈ 1,200. Compute the exact number from your files; don't trust this estimate.

Two things changed compared with the previous plan: the opponent's rows now contain their **moves** (so the agent can see, for example, that the opposing bench holds a super-effective move against our active) and **real stats** instead of nothing.

**Block 2 — active matchup (both sides).**

* Stat-boost stages for each active Pokémon: 7 each (Atk, Def, SpA, SpD, Spe, accuracy, evasion), divided by 6, so in \[−1, 1\].  
* **Speed order [OTS, new]:** each active Pokémon's effective speed = Spe stat × boost multiplier × 0.5 if paralysed × 1.5 if holding Choice Scarf, and more if you have time. Encode a one-hot (we are faster / speed tie / we are slower) and the ratio ours/(ours + theirs). Without OTS this was guesswork; with exact stats it is mostly exact. State which modifiers you ignored (e.g. Booster Energy / Protosynthesis speed boosts, Unburden, Tailwind).  
* Hazard-relevant counters that make the state closer to Markov: toxic counter for each active (turns poisoned / 15), and a sleep-turn counter if cheap to read.

**Block 3 — field.** Hazards on each side (Stealth Rock, Spikes layers /3, Toxic Spikes layers /2, Sticky Web), screens on each side and their turns remaining (/8), and turn/100.

**What is still hidden after OTS.** The opponent's choice this turn (simultaneous moves), the RNG (accuracy and damage rolls, crits), and some small counters we may not encode (e.g. turns of sleep remaining is hidden even in-game). This is why the problem is "nearly" an MDP, not exactly one.

**What is deliberately not in the state (midsem).** Damage-calculator features such as "expected damage of each move as % of the target's HP" or "probability this move KOs". poke-env has a Gen 9 damage calculator (`poke_env.calc.damage_calc_gen9.calculate_damage`, seen in the API index, untested), and with OTS these would be computable. Two reasons to leave them out for now:

1. Time: it adds integration and testing work on the critical path.  
2. Experimental validity: a KO-probability feature multiplies power by accuracy *for* the agent, handing it the very trade-off we are studying. It would not remove the noise in outcomes (so the TD-target mechanism still exists), but it would shrink the representation-learning part of the difference between LowVar and HighVar. That makes it a good **ablation for the final** ("does giving the agent a damage model reduce the cost of variance?"), not a default.

Including **accuracy** as a feature still matters for this experiment. Without it the agent cannot tell Surf from Hydro Pump except through their outcomes.

**An honest note on how much the static features actually add in the midsem.** Our 6 rows always describe the same 6 Pokémon, so their static part (stats, types, item, ability, moves) is *identical in every state of every battle*. To a neural network a constant input is just an extra bias: it carries no information. The opponent's static part takes only 3 different values, one per opponent team, so it is equivalent to telling the agent "this is team A, B or C". The real value of OTS in the midsem therefore comes from:

* knowing the opponent's **unrevealed** Pokémon and moves from turn 1 (that is real information the agent would otherwise lack), and  
* **derived dynamic features** that combine static and dynamic information: effectiveness of every move against the *current* opposing active, speed order with current boosts and paralysis.

The full static description is still worth encoding, because in the final (more teams, team preview choices) it is what lets the agent generalise across sets instead of memorising team IDs. Say this on the slide rather than claiming that "the agent knows the EVs" is what makes it good.

**Action a.** The discrete action space of poke-env's SinglesEnv: 4 moves plus 5 switches are the legal subset, and gimmick actions (Tera, Mega, Z, Dynamax) are always masked. An action mask removes illegal actions at every step. **[OTS]** Unchanged: OTS changes what the agent sees, not what it can do.

**Reward r.** A terminal reward of \+1 for a win and −1 for a loss, plus potential-based shaping:

r'\_t \= r\_t \+ \\gamma \\Phi(s\_{t+1}) \- \\Phi(s\_t), \\quad \\Phi(s) \= 0.5\\,\\frac{\\text{HP}\_{ours} \- \\text{HP}\_{opp}}{6} \+ 0.5\\,\\frac{\\text{alive}\_{ours} \- \\text{alive}\_{opp}}{6}  
Shaping of this form provably leaves the optimal policy unchanged (Ng, Harada and Russell, 1999, from memory). It just gives feedback before the battle ends. Always multiply the next potential by γ; forgetting that is a common bug.

**[OTS]** The reward is unchanged on purpose. With known stats you *could* weight HP by each Pokémon's actual HP stat, but keeping the potential identical to the previous plan avoids changing two things at once. Note one small improvement OTS gives for free: the opponent's 6 Pokémon are known from turn 1, so alive\_opp and HP\_opp are well defined from the start (previously unrevealed opponents had to be assumed at full HP).

**Discount.** γ \= 0.99. The effective horizon is about 1/(1−γ) \= 100 decisions, which covers a 30–60 turn 6v6 battle.

**Value functions (what both algorithms estimate)**

V^{\\pi}(s) \= \\mathbb{E}\_{\\pi}\\Big\[\\sum\_{k=0}^{\\infty} \\gamma^k r\_{t+k+1} \\,\\Big|\\, s\_t \= s\\Big\] \\qquad Q^{\\pi}(s,a) \= \\mathbb{E}\_{\\pi}\\Big\[\\sum\_{k=0}^{\\infty} \\gamma^k r\_{t+k+1} \\,\\Big|\\, s\_t \= s, a\_t \= a\\Big\]  
In words, V(s) means "how good is this board for me if I keep playing my way". Q(s, a) means "how good is it if I play a now, then my way".

**[OTS]** These definitions are now better justified. They assume the state s summarises everything relevant about the future (the Markov property). Without OTS that assumption is clearly false, because unrevealed opponent sets matter. With OTS it is approximately true, so V and Q are well-defined functions of the observation rather than averages over hidden situations.

**Policy 1: DQN (value-based).** Learn Q with a neural network using the Bellman optimality target. The policy is derived from Q: ε-greedy over legal actions, with ε decaying from 1.0 to 0.05 over the first 30% of training, and greedy at evaluation.

y\_t \= r\_t \+ \\gamma \\max\_{a' \\in \\mathcal{A}\_{legal}(s\_{t+1})} Q\_{\\bar\\theta}(s\_{t+1}, a') \\qquad \\mathcal{L}(\\theta) \= \\big(y\_t \- Q\_\\theta(s\_t, a\_t)\\big)^2  
Here θ̄ is a slowly updated target network, and transitions are sampled from a replay buffer. The max must also respect the mask: illegal actions get −∞.

**Policy 2: PPO (policy-based, actor-critic).** An actor outputs π(a|s) as a masked softmax. A critic learns V(s). Updates push up the probability of actions with positive advantage A, with clipping so one update cannot change the policy too much.

L^{CLIP}(\\theta) \= \\mathbb{E}\_t\\Big\[\\min\\big(\\rho\_t A\_t,\\ \\text{clip}(\\rho\_t, 1-\\epsilon, 1+\\epsilon) A\_t\\big)\\Big\], \\quad \\rho\_t \= \\frac{\\pi\_\\theta(a\_t|s\_t)}{\\pi\_{old}(a\_t|s\_t)}  
A\_t is computed with GAE (λ \= 0.95), and ε \= 0.2.

**Fairness rules between the algorithms**

* Same encoder, same reward, same network body (e.g. 2 hidden layers of 256), same number of environment steps, same opponents, same seeds.  
* Default-ish hyperparameters for both. Do not tune one more than the other.  
* Evaluate both deterministically (DQN greedy, PPO argmax).  
* **[OTS]** Both algorithms get exactly the same OTS observation.

**[OTS] Network size.** The input grows from ~200 to ~1,200 features. A 256-256 MLP is still fine: the first layer has about 1,200 × 256 ≈ 310k weights, trivial for your CPU or the RTX 3050. Keep 256-256 for both algorithms rather than enlarging; changing the architecture now adds another variable.

**Simplifications for the midsem, stated explicitly**

* Fixed teams and lead, no Tera, one fixed opponent policy.  
* **[OTS, replaces the old "partial observability is ignored" bullet]** Near-full observability through open team sheets, and no memory or recurrence. Residual hidden information (the opponent's simultaneous choice, RNG, a few counters) is not modelled.  
* **[OTS]** Information is asymmetric: only our agent sees full sheets (Section 2b).

## **5\. Algorithms: Q-learning vs PPO and expected results**

The main prediction is an *interaction*: DQN should lose more performance than PPO when moving from LowVar to HighVar. There is a clear theoretical reason for this, so it is a good hypothesis to test.

**"Q-learning" here means DQN.** Tabular Q-learning needs a table entry per state. Pokémon has astronomically many states, so a neural network must approximate Q. That is DQN: the same Q-learning update with a function approximator, a replay buffer and a target network.

**Masking problem with DQN.** Stable-Baselines3's MaskablePPO lives in sb3-contrib ([docs](https://docsearch.algolia.com/mcp/docs/repo/stable-baselines-team/stable-baselines3-contrib)). There is no maskable DQN there. The options are:

1. **Recommended:** write a small masked DQN in PyTorch (about 150 lines). Masking goes in both action selection and the max in the target. A one-line change turns it into Double DQN. It is also the best learning exercise in the project.  
2. Fallback: SB3 DQN, where an illegal choice is replaced by a random legal move. Simple, but it adds noise and favours PPO, so label it as a confound if used.

**Expected results and why**

| Prediction | Reason | How we'll see it |
| :---- | :---- | :---- |
| DQN overestimates action values more on HighVar | Q-learning uses the max over noisy estimates, which is biased upward; van Hasselt showed this makes Q-learning perform very poorly in some stochastic environments ([Double Q-learning, NeurIPS 2010](https://papers.nips.cc/paper/3964-double-q-learning)). **[OTS]** With hidden information mostly removed, the noise in DQN's targets comes mainly from RNG, so a larger gap on HighVar is more directly attributable to the swap | Plot mean predicted max-Q at battle start vs actual discounted return; the gap should be larger on HighVar |
| PPO is more stable across seeds | Clipped, on-policy updates and a learned baseline (the critic) reduce gradient variance | Smaller spread between seed curves |
| DQN can be more sample-efficient early on LowVar | Replay reuses each transition many times, while PPO discards data after each update | DQN curve rises first on LowVar, if anything |
| Both learn slower on HighVar | Noisier TD targets and advantages, so more samples per unit of signal | Later threshold crossing, lower learning-curve area |
| PPO critic explains less variance on HighVar | Returns are less predictable from the state | SB3's logged explained\_variance is lower |
| **[OTS, new]** The critic's explained variance is higher overall than it would be without OTS, and the LowVar–HighVar gap in it is a cleaner "RNG-only" signal | The critic no longer has to average over unknown opponent sets; what it cannot explain is mostly dice | Not directly testable in the midsem (no no-OTS arm); report the absolute numbers and flag the comparison as future work |
| **[OTS, new]** Early learning is somewhat slower than a small encoder would give, then catches up | ~1,000 extra input dimensions, most of them constant or 3-valued, add some optimisation noise early on, but carry little information to fit | Can't be tested without a second encoder; mention as a caveat if curves are slow to start |
| Final win rates may barely differ | Stronger 110-BP hits partly offset misses (Section 2\) | Use RL lift, not raw win rate |

**Honest expectations for the midsem**

* With about 100k–300k steps and 3 seeds per cell, differences may sit inside the noise. Report that plainly.  
* The DQN pipeline is the biggest schedule risk. If it isn't stable by Monday morning, present PPO results plus DQN as preliminary.  
* These are theory-based predictions, not results anyone has published for Pokémon specifically. I found no Pokémon study comparing these algorithms across accuracy variants.  
* **[OTS]** The two new predictions are my reasoning, not literature. The "constant features act as a bias" argument is standard neural-network reasoning; the size of any slowdown is a guess.

**Optional third arm (only if time): Double DQN.** If the overestimation story is right, Double DQN should close most of DQN's gap on HighVar. That would be a strong, clean finding for the final.

## **6\. Metrics: what sample efficiency and stability measure**

Both metrics measure **how hard a team is to learn for a given algorithm**, not how strong the team is. Keep those two claims separate in the slides.

| Metric | Exact definition | What it tells us | What it does not tell us |
| :---- | :---- | :---- | :---- |
| Sample efficiency: threshold | Environment steps until the evaluation win rate vs the heuristic first stays above a target (e.g. heuristic baseline \+ 10 points) for 2 consecutive evals | How much experience the algorithm needs to extract a useful signal from this team | Final strength; it is sensitive to the chosen threshold |
| Sample efficiency: AUC | Area under the eval win-rate curve divided by total steps (the average win rate over training) | Same idea, with no threshold to choose; more robust with few seeds | Where learning plateaus |
| Stability: across seeds | Standard deviation (or range) of final win rate over seeds | How much the outcome depends on luck in initialisation and exploration, i.e. how reproducible the method is on this team | Anything reliable with only 3 seeds; show every seed curve, not just a mean |
| Stability: within run | Largest drop between consecutive evals, and std of the last 5 evals | Whether learning oscillates or collapses (common for DQN) | The cause of the oscillation |
| RL lift | Final RL win rate minus win rate of the heuristic piloting the same team version | How much learning **plus the OTS information advantage** adds beyond scripted play, which cancels most of the strength difference from the swap | Absolute team strength; **[OTS]** how much is learning vs information |
| **[OTS]** Lift difference | RL lift on HighVar minus RL lift on LowVar, per algorithm | The variance effect on what learning adds; the information advantage is the same in both versions, so it cancels | Why the difference exists |
| Diagnostics | DQN: predicted max-Q vs realised return. PPO: critic explained variance. Both: miss rate, battle length | The *mechanism* behind any difference | — |

**[OTS] Why "lift difference" is now the headline lift number.** Because only our agent has full sheets, raw RL lift mixes learning with an information advantage that the heuristic pilot does not have. The advantage is identical for LowVar and HighVar, so subtracting the two lifts removes it. Present raw lift with the caveat, and lead with the difference.

**Measure the x-axis in environment steps (decisions), and also report battles.** HighVar battles may be longer because misses prolong them, so steps and battles diverge.

**Statistics**

* Win rate over N battles: report a Wilson 95% confidence interval. N \= 500 gives about ±4.4 points near 50%, and N \= 1000 gives about ±3.1.  
* With 3 seeds, show every individual curve. The deep-RL evaluation literature warns that point estimates from a handful of runs are unreliable and recommends interval estimates and robust aggregates such as the interquartile mean ([Agarwal et al., NeurIPS 2021](https://arxiv.org/abs/2108.13264v4)). IQM needs more runs than you'll have, so cite it as the method for the final.  
* Headline figure: a 2×2 interaction plot (algorithm × team), with AUC on the y-axis and one dot per seed.

## **7\. Step-by-step task list (from Sunday morning)**

It is Sunday 18 October. That leaves today, Monday and Tuesday before the Wednesday presentation.

The **critical path** is T2 → T3 → **T6a** → T6 → T10/T11 → T13 → T14 → T15 → T20 → T21. Anything on it that slips moves the deadline, so the pipeline work (T6a, T6, T10, T11) must finish today so the main runs can train overnight. The biggest single risk is T11 (DQN).

**[OTS] Schedule impact.** OTS adds about 1.5–2.5 h to the critical path (T6a plus a longer T6). If you are already behind, use the **minimum OTS** version in T6a/T6 tonight and skip the optional parts; everything else in the plan works unchanged.

Tasks without a dependency on each other can run in parallel, so split them between the two of you. A natural split is one person on the training pipeline and one on teams, baselines, evaluation and slides. Fill in Owner together.

| ID | When | Task | Needs | Est. | Owner |
| :---- | :---- | :---- | :---- | :---- | :---- |
| T1 | Sun AM | Choose teams and fill Section 3 (incl. IVs and stats) | — | 1–1.5 h |  |
| T2 | Sun AM | Set up the environment (both laptops) | — | 1–2 h |  |
| T3 | Sun AM | Pin versions, read the API, run the official RL example | T2 | 1 h |  |
| T4 | Sun midday | Team files, validation, diff script, **stat check** | T1 | 1–1.5 h |  |
| T5 | Sun midday | Battle format, Tera off, team rotation, first real battle | T2, T4 | 1 h |  |
| **T6a** | Sun PM | **[OTS] Team-sheet loader, stat calculator, opponent identification** | T3, T4 | 1.5 h |  |
| T6 | Sun PM | Observation encoder (OTS version) | T3, T6a | 3.5–4 h |  |
| T7 | Sun PM | Reward function | T3 | 1 h |  |
| T8 | Sun PM | Baselines (runs in background) | T5 | 1 h \+ runtime |  |
| T9 | Sun PM | Throughput benchmark and step budget | T6 | 1 h |  |
| T10 | Sun evening | PPO pipeline | T6, T7 | 3 h |  |
| T11 | Sun evening | Masked DQN pipeline | T6, T7 | 4–5 h |  |
| T12 | Sun evening | Checkpoint evaluation script | T5, T6 | 2 h |  |
| T13 | Sun night | Smoke runs: all 4 cells, short | T9–T12 | 1 h \+ runtime |  |
| T14 | Sun night → Mon | Main training runs (12) | T13 | runtime |  |
| T15 | Mon | Checkpoint evaluations for learning curves | T12, T14 | runtime |  |
| T16 | Mon | Final-project spec | T6 | 3 h |  |
| T17 | Mon | Theory and related-work slides | — | 3 h |  |
| T18 | Mon | Doubles smoke test | T2 | 1 h |  |
| T19 | Tue AM | Final evaluations | T14 | runtime |  |
| T20 | Tue | Analysis and plots | T8, T15, T19 | 3 h |  |
| T21 | Tue | Slides, replays and rehearsal | T16, T17, T20 | 3 h |  |

### **What each task involves**

**T1 — Choose teams and fill Section 3\.** Pick the swap pairs first, then the Pokémon that legally learn both moves, then the opponents. Type each set into the Showdown teambuilder as you go so illegal moves show up immediately. This comes first because the team files, baselines and training all depend on it, and it's the one decision you need to make together.

* **[OTS]** Also follow the new rules: no Illusion/Transform, avoid Trick/Switcheroo, 3 opponent teams with different species sets and different leads. Copy each Pokémon's final stats from the teambuilder into the tables; T4 checks your code against them.  
* *Done when:* every table in Section 3 is filled and both of you agree.

**T2 — Set up the environment.** Do this on both laptops, so training can be split across two machines later.

sudo pacman \-S nodejs npm git  
git clone https\://github.com/smogon/pokemon-showdown.git  
cd pokemon-showdown && npm install && cp config/config-example.js config/config.js  
node pokemon-showdown start \--no-security   \# leave running in its own tmux pane  
Then the Python side. poke-env needs Python 3.10 or newer; 3.11 avoids PyTorch wheel gaps on Arch's newest Python.

uv venv \--python 3.11 .venv && source .venv/bin/activate  
uv pip install poke-env stable-baselines3 sb3-contrib torch tensorboard  
uv pip freeze \> requirements.txt   \# share this file so both laptops are identical

* *Done when:* the "first local battle" script from the poke-env README (two RandomPlayers) finishes on both machines.

**T3 — Pin versions, read the API, run the official example.** poke-env changes between versions, so read your installed source rather than blog posts. Write down five things in a shared notes file:

1. The exact action-space size.  
2. Which action index means which move or switch (action\_to\_order).  
3. How the action mask is exposed.  
4. The heuristic player's class name.  
5. How a Teambuilder supplies teams.

**[OTS]** Add three more notes:

6. Where the battle object stores the opponent's team-preview species (attribute name).  
7. Whether poke-env has a stats helper for computing actual stats from base stats, EVs, IVs and nature (the docs list a "Stats utilities" module; check what it offers).  
8. What SimpleHeuristicsPlayer uses about the opponent (types, base stats, boosts, spreads?) and whether it has random tie-breaks. This decides how big the information asymmetry is (Section 2b).

Then run poke-env's official reinforcement-learning example unchanged. It already trains PPO against SimpleHeuristicsPlayer with action masking, so it is the template for T10: your pipeline is that example plus your encoder, reward and teams.

* *Done when:* the example runs end to end and the eight notes are written.

**T4 — Team files, validation, diff script, stat check.** Export each team from the teambuilder into teams/lowvar.txt, teams/highvar.txt and teams/opp\_a.txt, opp\_b.txt, opp\_c.txt. Validate all five in the format chosen in T5. Write the diff script (Section 3).

* **[OTS]** Showdown's export omits IVs that are 31 and EVs that are 0. Your parser must fill those defaults (IV 31, EV 0, level 100 unless stated) rather than leaving them blank.  
* **[OTS]** Extend the diff script to assert different species sets and leads across the 3 opponent teams.  
* *Done when:* all five validate and the diff script reports only the intended swaps and passes the identification check.

**T5 — Battle format, Tera off, team rotation, first real battle.**

* **Tera must be off for both players.** Masking only our Tera actions would leave the heuristic opponent free to Terastallize. Find a format or rule setup where Tera is banned for everyone.  
* If that's not possible through poke-env, mask our Tera actions *and* subclass the opponent so it never Terastallizes. Write down which route you took.  
* Write a small Teambuilder subclass that returns one of the 3 opponent teams at random each battle.  
* Leave team order as given, so slot 1 always leads.  
* **[OTS]** Confirm the format has team preview (Gen 9 OU does). If you use a custom format without it, rely on the lead species to identify the opponent team.  
* *Done when:* 20 test battles run with slot 1 always leading, no \-terastallize lines in the logs, and all 3 opponent teams appearing.

**T6a — [OTS] Team-sheet loader, stat calculator, opponent identification.** A small module, written and tested before the encoder:

* Parse the 5 team files into a dictionary per Pokémon: species, types, item, ability, moves, nature, EVs, IVs, level.  
* Compute the 6 actual stats. Use poke-env's stats helper if T3 found one; otherwise use the standard Gen 3+ formulas (from memory, verify): HP = floor((2·Base + IV + floor(EV/4)) · Level / 100) + Level + 10; other stats = floor((floor((2·Base + IV + floor(EV/4)) · Level / 100) + 5) · nature), with nature 1.1, 1.0 or 0.9.  
* **Verification:** assert that your computed stats equal the stats you copied from the teambuilder in T1, for all 24 Pokémon. This is the single best check that the static block is right.  
* Build the item and ability vocabularies and the per-stat normalisers from the 4 files.  
* A function that takes a battle and returns which opponent team it is (species set at team preview, falling back to the lead), and raises an error if no team or more than one matches.  
* Cache the result per battle tag, so this runs once per battle, not once per step.  
* **Minimum OTS (if short on time):** opponent identification + moves + types + stats only; skip items and abilities in the vector. You lose a little information but nothing breaks.  
* *Done when:* the stat assertion passes for all 24 Pokémon and identification is correct in 20 test battles (log the identified team next to the team the Teambuilder actually sent).

**T6 — Observation encoder (OTS version).** Subclass SinglesEnv and implement embed\_battle following Section 4\. Declare the observation space with correct bounds and scale every feature to roughly \[0, 1\] (or \[−1, 1\] for boosts).

* Include **move accuracy**. Never-miss moves are reported as True, so convert them to 1.0. Without accuracy, the agent can't tell the two team versions apart except through outcomes.  
* **[OTS]** Fill all 12 rows from turn 1 using T6a, keyed by species in sheet order, and overwrite the dynamic parts (HP, status, item, PP) from the battle object each step.  
* **[OTS]** Implement the speed-order block, and state which speed modifiers you ignored.  
* **[OTS]** Test the opponent rows specifically: at turn 1, the opponent's bench rows must already contain moves and stats, not zeros.  
* *Done when:* a test over 20 random battles shows a fixed vector length, no NaN or infinite values, everything within bounds, and opponent rows filled from turn 1.

**T7 — Reward function.** Implement calc\_reward as a terminal ±1 plus the potential-based shaping from Section 4\. Store the previous potential per battle, and remember the γ in front of the next potential.

* *Done when:* hand-checking 2–3 battles shows the final step carries ±1 with the right sign and the shaping terms are small (well under 1 per step).

**T8 — Baselines.** Pilot each of our team versions with random, max-base-power and heuristic players against the heuristic opponent, 500 battles each (6 cells, 3,000 battles; let it run in the background). Record:

* Win rate with Wilson confidence interval.  
* Mean battle length.  
* Miss rate (count \-miss lines in the logs).

This is your first real result: it shows how much the swap changed strength and noise before any learning. The heuristic-with-our-team win rate is also the reference for RL lift.

* **[OTS]** These scripted baselines do **not** use the full sheets. Write that next to the table; it is why Section 6 leads with the lift *difference*.  
* *Done when:* a CSV with all 6 cells exists.

**T9 — Throughput benchmark and step budget.** Run a random agent through your env for 2 minutes each with 1, 4 and 8 parallel envs, and measure steps per second. Then set the step budget so 12 runs fit overnight on two laptops, using the formula in Section 8, and write the number there. The simulator is CPU-bound, so the GPU barely matters.

* **[OTS]** Run the benchmark *with the OTS encoder*. Building a ~1,200-number vector in Python is still cheap compared with Showdown, but only if the static parts are cached per battle (T6a). If steps/s drop noticeably compared with a trivial encoder, the cache isn't working.  
* *Done when:* the step budget is written in Section 8\.

**T10 — PPO pipeline.** Start from the T3 example and swap in your env, encoder, reward and team rotation.

* Settings: learning rate 3e-4, about 2,048 steps per update, batch 128, γ \= 0.99, GAE λ \= 0.95, clip 0.2, entropy coefficient 0.01, network 256-256.  
* Save a checkpoint every 10k steps and log each training episode's win or loss.  
* *Done when:* a 20k-step sanity run against RandomPlayer shows a clearly rising win rate.

**T11 — Masked DQN pipeline.** Write it in PyTorch, with the same 256-256 network body as PPO for fairness. You need:

* A replay buffer (about 100k transitions) and a target network (synced every 2,000 steps).  
* Learning rate 1e-4, batch 64, γ \= 0.99, learning starts after 5k steps.  
* ε decaying from 1.0 to 0.05 over the first 30% of steps.  
* The mask applied in **two** places: set illegal Q values to −∞ before choosing an action, and before taking the max in the target.  
* No bootstrapping on the final step of a battle.  
* An optional double=True flag that picks the next action with the online network but values it with the target network.  
* Log the predicted max-Q at each battle's first state alongside the actual discounted return; that is the overestimation plot.  
* **[OTS]** Memory check: 100k transitions × 2 observations × ~1,200 float32 values ≈ 1 GB. Fine on 16 GB RAM for one run, but if you run 4–6 DQN runs in parallel that is 4–6 GB. Store observations as float16 or store only s and s' once per transition if memory gets tight.  
* *Done when:* the same 20k-step sanity run shows a rising win rate.

**T12 — Checkpoint evaluation script.** Load any checkpoint (PPO or DQN), act **deterministically** (DQN greedy, PPO argmax), play N battles against a chosen opponent with the same team rotation, and append one CSV row: algorithm, team, seed, step, opponent, wins, N, Wilson low/high, mean length.

* Deterministic play matters because training win rates aren't comparable between algorithms: DQN trains with ε-random moves, PPO with sampled moves.  
* **[OTS]** It must use the same OTS encoder as training (import it, don't copy it), or checkpoints will receive a different input than they were trained on.  
* *Done when:* it works on a T10 and a T11 checkpoint.

**T13 — Smoke runs.** Run all 4 cells (2 algorithms × 2 teams), 1 seed each, about 20k steps, then evaluate their checkpoints with T12. This catches crashes, NaN losses and wiring bugs before you spend the night training.

* *Done when:* all 4 finish and evaluation produces sensible numbers.

**T14 — Main training runs.** 12 runs: 2 algorithms × 2 teams × seeds 0, 1, 2, all at the same step budget. Split the runs across both laptops only if requirements.txt and the code are identical on both. Use one tmux window per batch and log to runs/\<algo\>\_\<team\>\_s\<seed\>/.

* Rule: **never change code or settings mid-batch.** If a bug fix is needed, restart every affected run so all cells stay comparable. **[OTS]** This includes the team files: they now feed the encoder, so editing a spread mid-batch changes the observation.  
* Note: seeds control network initialisation and exploration, not Showdown's battle RNG.  
* *Done when:* all 12 runs have finished and saved checkpoints.

**T15 — Checkpoint evaluations for learning curves.** Evaluate every 20k-step checkpoint of every run, 100 battles each against the heuristic. These points are the learning curves for sample efficiency and within-run stability. Start on finished runs while others are still training.

* *Done when:* every checkpoint has a CSV row.

**T16 — Final-project spec.** Doubles 4v4 MDP:

* DoublesEnv per-slot actions with targets.  
* Extra features (perish counter, trapped flag, partner state).  
* The reward.  
* Opponents: heuristic first, then a mix with existing bots (Section 9).  
* The same metrics.  
* **[OTS]** The information setting: full sheets vs official OTS (Section 9), Tera type in the sheet, and team-preview "bring 4 of 6" as a new decision that OTS makes informed.

This is half the midsem deliverable.

**T17 — Theory and related-work slides.** MDP and Bellman equation, how DQN and PPO learn, overestimation bias and why it predicts the DQN × HighVar interaction, then VGC-Bench, Metamon and the PokéAgent Challenge. **[OTS]** Add one slide on MDP vs POMDP: the three sources of transition noise from Section 2b, and why OTS leaves mainly RNG.

**T18 — Doubles smoke test.** Two random players through DoublesEnv for a few battles. One slide's worth of proof that the final is technically feasible.

**T19 — Final evaluations.** The last checkpoint of each of the 12 runs: 1,000 battles against the heuristic (drop to 500 if short on time), plus 200 each against random and max-base-power.

**T20 — Analysis and plots.**

* Learning curves with every seed shown.  
* Threshold steps and AUC per cell.  
* Seed spread and largest within-run drop.  
* RL lift: final win rate minus the T8 heuristic baseline for the same team. **[OTS]** Plus the lift difference (HighVar − LowVar) per algorithm, which cancels the information advantage.  
* The 2×2 interaction plot.  
* DQN's predicted-Q vs actual-return plot, and PPO's explained variance.  
* **[OTS]** The KO-threshold table from Section 2 caveat 2, computed with the real spreads.  
* One summary table per cell.

**T21 — Slides, replays and rehearsal.** Suggested order:

1. Introduction and motivation.  
2. RL basics.  
3. Setup and swap verification (the expected-BP table, diff check, **[OTS]** information setting and why it isolates RNG).  
4. The MDP spec.  
5. Algorithms and predictions.  
6. Baselines.  
7. Results.  
8. Honest limitations (**[OTS]** including the information asymmetry and "full sheets ≠ official OTS").  
9. Plan for the final.

Save 2 replays: a game lost to a Hydro Pump-style miss, and a clean LowVar win. Rehearse once with a timer.

## **8\. Compute budget, cut lines and risks**

The step budget is set after T9. I don't know the real throughput on your machine, so the numbers below are placeholders to replace.

**Budget formula.** Hours per batch \= steps per run ÷ (steps per second per run × 3600). Example: 200k steps at 50 steps/s comes to about 1.1 h per run. With 4–6 runs in parallel on the Ryzen's 8 cores, 12 runs take about 2–4 h. Several runs can share one Showdown server, but start a second server on another port if it becomes the bottleneck.

**Cut lines, in order, if time runs short**

1. **[OTS]** Use "minimum OTS" (T6a): skip item and ability one-hots and the speed block refinements. Do this before cutting anything experimental.  
2. Drop to 2 seeds per cell. This is the minimum for any stability statement.  
3. Halve the step budget, keeping it equal across all cells.  
4. Drop the Double DQN arm (it is optional anyway).  
5. Present PPO on both teams plus DQN as preliminary. Never drop a team: the team comparison is the point.

| Risk | Sign | Fallback |
| :---- | :---- | :---- |
| DQN doesn't learn | Win rate vs random not rising after 50k steps | Check masking in the target, reward scale and target-network update rate; else use SB3 DQN with legal-move fallback and label it a confound |
| Low throughput | Below 20 steps/s | Fewer steps, fewer seeds, run overnight; **[OTS]** check the per-battle cache first |
| Neither algorithm beats the heuristic | Flat curves vs the heuristic | Present learning vs random plus the baselines from T8; still a valid pilot |
| poke-env API mismatch | Code from examples fails | Follow the examples folder of the installed version |
| A learnset is illegal | Validator error | Replace that Pokémon with another OU user of the same swap pair |
| **[OTS]** Wrong static features | Stat assertion in T6a fails, or opponent rows are zero at turn 1 | Fix before any training; this bug would silently corrupt every run |
| **[OTS]** Opponent team misidentified | Logged identified team ≠ team actually sent | Make leads distinct; fall back to lead-species identification |
| **[OTS]** Row order drifts during battle | A Pokémon's HP appears in another Pokémon's row after a switch | Key rows by species, never by list position |
| **[OTS]** DQN replay buffer uses too much RAM | System starts swapping with several DQN runs | float16 observations, smaller buffer (50k), or fewer parallel DQN runs |

## **9\. How this carries into the final, and sources**

The final reuses the encoder, reward, both training pipelines, the evaluation script and the metrics. It changes to DoublesEnv, 4v4 with fixed teams and per-slot actions with targets. It adds the Perish Trap vs standard comparison as a second axis: strategic complexity rather than RNG variance. VGC-Bench's doubles action encoding is the reference implementation to study ([VGC-Bench](https://arxiv.org/abs/2506.10326v3)).

**[OTS] Information setting in the final**

* **Official VGC OTS hides spreads.** Tournament open team sheets list species, Tera type, item, ability and moves, but not EVs, IVs or nature, so players see everything except exact stats. Two options:  
  * **Full sheets** (what the midsem uses): keeps the experiment controlled and comparable with the midsem. Less realistic.  
  * **Official OTS**: replace each opponent's exact stats with an estimate (e.g. a standard spread for that species) and add Tera type. More realistic, and Showdown's own OTS mechanism (Route B) could supply it directly.  
  * Recommendation: keep full sheets as the main setting so midsem and final are comparable, and run official OTS as a realism check if time allows.  
* **OTS matters more for Perish Trap.** Whether a trap works depends on whether the opponent has Ghost types, Shed Shell, pivot moves or Protect, all of which are on the sheet. So the "strategic complexity" axis of the final benefits even more from OTS than the variance axis does.  
* **Team preview becomes a real decision** ("bring 4 of 6"), and OTS is what makes it an informed one. It also makes the static features genuinely useful, since the agent then faces different subsets of sets rather than 3 fixed teams.  
* **Optional ablations:** (a) no-OTS encoder vs OTS encoder, to measure how much information helps each algorithm; (b) adding damage-calculator features (Section 4), to test whether a built-in damage model reduces the cost of variance.

**Existing bots as opponents (final only)**

* **Curriculum.** Learn the basics against the heuristic player first, then train against a mix (e.g. 80% heuristic, 20% an existing bot), sampling the opponent per battle the same way opponent teams are sampled now.  
* **Candidates.** Metamon's heuristic baselines and pretrained agents (Gen 9 OU supported) ([Metamon](https://github.com/UT-Austin-RPL/metamon)). Foul Play, a tree-search bot built on poke-engine ([poke-engine](https://github.com/pmariglia/poke-engine)), run at a fixed search time per turn so its strength stays constant.  
* **Held-out evaluation.** Evaluate final checkpoints against a strong bot the agent never trained on, to test whether the variance effect holds against good play and whether the agent overfit to the heuristic.  
* **Hypothesis to test: variance helps the underdog.** Against an opponent stronger than our agent, the HighVar team may do relatively *better*, because a weaker player needs luck to win. A flip between the heuristic and strong-bot results would refine the original "low variance is better" idea.  
* **Practical limits.** Strong bots think per turn, so they are far slower than the heuristic. Keep them for evaluation and a small training share, and always give them our preset opponent teams so only one thing changes. **[OTS]** Check what information each bot gets; a bot without OTS against our agent with OTS is another asymmetry to report.

**Sources (opened while writing this plan)**

* [poke-env documentation: env module](https://poke-env.readthedocs.io/en/stable/modules/env.html)  
* [sb3-contrib overview (MaskablePPO)](https://docsearch.algolia.com/mcp/docs/repo/stable-baselines-team/stable-baselines3-contrib)  
* [van Hasselt, Double Q-learning (NeurIPS 2010\)](https://papers.nips.cc/paper/3964-double-q-learning)  
* [Agarwal et al., Deep RL at the Edge of the Statistical Precipice (NeurIPS 2021\)](https://arxiv.org/abs/2108.13264v4)  
* [Angliss et al., VGC-Bench (AAMAS 2026\)](https://arxiv.org/abs/2506.10326v3)  
* [Grigsby et al., Metamon (RLC 2025\)](https://rpl.cs.utexas.edu/publications/2025/08/05/grigsby-rlc25-metamon/)  
* [PokeTools Gen 9 OU usage, September 2026](https://www.poketools.com/meta/gen-9-ou)  
* **[OTS]** [poke-env API index](https://poke-env.readthedocs.io/en/stable/genindex.html): lists `Player.accept_open_team_sheet`, the `poke_env.calc.damage_calc_gen9.calculate_damage` damage calculator and a stats-utilities module. Seen in the index only; behaviour not tested.  
* **[OTS]** Tournament rules describing official open team sheets as everything except stats/EVs/IVs/nature, e.g. [Dark Storm Team Series #1 (Limitless)](https://play.limitlesstcg.com/tournament/685ad3dfc4596e8122362064/details) and [Lab Gamer's Reg E OTS rules (Limitless)](https://play.limitlesstcg.com/tournament/655d794aa5f82e05ee5694da/details).

**Cited from memory, not opened:** Ng, Harada and Russell (1999) on potential-based shaping; Mnih et al. (2015) on DQN; Schulman et al. (2017) on PPO. Move base power, accuracy and PP values are also from memory, so verify them in Showdown. **[OTS]** The stat formulas in T6a and the claim about what SimpleHeuristicsPlayer uses are also from memory; T6a's stat assertion and T3's note 8 verify them.

Own Notes: 

- Sparse Reward Problem: Each game is about 30-50 turns, decisions (actions to be taken) need to be made each turn, but reward is gotten only at end of battle 
