"""Team diff and team-rule checks (spec Section 8, test 1).

* LowVar and HighVar differ only in swapped moves, each swap keeps its slot, type and category and
  trades accuracy for power.
* The 3 opponent teams have pairwise different species sets and leads.
* Every team has exactly N_TEAM Pokémon, with known species, legal abilities and known moves.
* No weather setters, accuracy/evasion modifiers, Illusion, Transform/Imposter or Trick/Switcheroo
  (spec Section 2.5).

Exits with status 1 if any check fails.
"""

import itertools
import sys

import config
from teamsheet import GEN_DATA, OPP_LEADS, OPP_SPECIES, OPP_SHEETS, OUR_SHEETS

# Hand-listed because poke-env ships no ability/item data. Moves are also scanned generically below.
WEATHER_ABILITIES = {"drizzle", "drought", "sandstream", "snowwarning", "orichalcumpulse", "desolateland",
                     "primordialsea", "deltastream", "sandspit"}
ACCURACY_ABILITIES = {"compoundeyes", "hustle", "victorystar", "noguard", "sandveil", "snowcloak", "tangledfeet",
                      "keeneye", "illuminate", "mindseye"}
LOOKUP_ABILITIES = {"illusion", "imposter"}
ACCURACY_ITEMS = {"brightpowder", "laxincense", "widelens", "zoomlens", "blunderpolicy", "micleberry"}
ACCURACY_MOVES = {"lockon", "mindreader", "foresight", "odorsleuth", "miracleeye", "gravity"}
LOOKUP_MOVES = {"transform", "trick", "switcheroo"}
ACC_EVA = {"accuracy", "evasion"}


def move_boosts(entry):
    """Every boosts dict a move can apply, to its target or to the user."""
    effects = [entry, entry.get("self") or {}]
    for sec in [entry.get("secondary")] + list(entry.get("secondaries") or []):
        if sec:
            effects += [sec, sec.get("self") or {}]
    return [e["boosts"] for e in effects if e.get("boosts")]


def banned_move_reason(move):
    entry = GEN_DATA.moves[move]
    if entry.get("weather"):
        return "sets weather"
    if move in ACCURACY_MOVES or any(ACC_EVA & set(b) for b in move_boosts(entry)):
        return "changes accuracy/evasion"
    if move in LOOKUP_MOVES:
        return "breaks the team-sheet lookup"
    return None


def check_team(name, sheets):
    problems = []
    if len(sheets) != config.N_TEAM:
        problems.append(f"{name}: {len(sheets)} Pokémon, expected {config.N_TEAM}")
    for m in sheets:
        where = f"{name}/{m.species}"
        legal = {a.lower().replace(" ", "").replace("-", "") for a in GEN_DATA.pokedex[m.species]["abilities"].values()}
        if m.ability not in legal:
            problems.append(f"{where}: ability {m.ability} not legal ({sorted(legal)})")
        if m.ability in WEATHER_ABILITIES | ACCURACY_ABILITIES | LOOKUP_ABILITIES:
            problems.append(f"{where}: banned ability {m.ability}")
        if m.item in ACCURACY_ITEMS:
            problems.append(f"{where}: banned item {m.item}")
        if len(m.moves) != 4 or len(set(m.moves)) != 4:
            problems.append(f"{where}: needs 4 distinct moves, has {list(m.moves)}")
        for move in m.moves:
            if move not in GEN_DATA.moves:
                problems.append(f"{where}: unknown move {move}")
            elif reason := banned_move_reason(move):
                problems.append(f"{where}: banned move {move} ({reason})")
    return problems


def accuracy(move):
    acc = GEN_DATA.moves[move]["accuracy"]
    return 100 if acc is True else acc


def diff_versions(low, high):
    """Return (problems, swaps); swaps are (species, slot, low_move, high_move)."""
    problems, swaps = [], []
    for slot, (lm, hm) in enumerate(zip(low, high)):
        same = ("species", "item", "ability", "nature", "level", "evs", "ivs", "stats")
        for field in same:
            if getattr(lm, field) != getattr(hm, field):
                problems.append(f"slot {slot}: {field} differs ({getattr(lm, field)} vs {getattr(hm, field)})")
        for i, (a, b) in enumerate(zip(lm.moves, hm.moves)):
            if a == b:
                continue
            swaps.append((lm.species, i, a, b))
            ea, eb = GEN_DATA.moves[a], GEN_DATA.moves[b]
            if (ea["type"], ea["category"]) != (eb["type"], eb["category"]):
                problems.append(f"{lm.species}: {a} -> {b} changes type or category")
            if not (eb["basePower"] > ea["basePower"] and accuracy(b) < accuracy(a)):
                problems.append(f"{lm.species}: {a} -> {b} is not a power-for-accuracy swap")
    if len(low) != len(high):
        problems.append(f"team sizes differ ({len(low)} vs {len(high)})")
    if not swaps:
        problems.append("lowvar and highvar are identical")
    return problems, swaps


def run_checks():
    """Return (problems, swaps)."""
    problems = []
    for team, sheets in OUR_SHEETS.items():
        problems += check_team(team, sheets)
    for key, sheets in OPP_SHEETS.items():
        problems += check_team(f"opp_{key}", sheets)
    diff_problems, swaps = diff_versions(OUR_SHEETS["lowvar"], OUR_SHEETS["highvar"])
    problems += [f"lowvar vs highvar: {p}" for p in diff_problems]
    for a, b in itertools.combinations(OPP_SHEETS, 2):
        if OPP_SPECIES[a] == OPP_SPECIES[b]:
            problems.append(f"opp_{a} and opp_{b} have the same species set")
        if OPP_LEADS[a] == OPP_LEADS[b]:
            problems.append(f"opp_{a} and opp_{b} have the same lead ({OPP_LEADS[a]})")
    return problems, swaps


def main():
    problems, swaps = run_checks()
    print("Swapped moves (lowvar -> highvar), expected BP = power x accuracy:")
    for species, slot, a, b in swaps:
        ea, eb = GEN_DATA.moves[a], GEN_DATA.moves[b]
        print(f"  {species:12} move {slot + 1}: {a} {ea['basePower']}/{accuracy(a)} -> {b} {eb['basePower']}/{accuracy(b)}"
              f"   ({ea['basePower'] * accuracy(a) / 100:g} -> {eb['basePower'] * accuracy(b) / 100:g})")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("\nall team checks passed")


if __name__ == "__main__":
    main()
