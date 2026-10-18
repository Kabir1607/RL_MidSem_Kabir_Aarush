"""T8 baselines: scripted pilots on our team vs the heuristic opponent.

Grid: team version (low/high) x pilot (random/maxbp/heuristic) x opponent team (A/B/C),
N battles per block (default 500), one CSV row per battle.

Requires a local Showdown server with the custom format `gen9ounotera`
(pokemon-showdown/config/custom-formats.ts), so Tera is off for both players.
Scripted pilots use poke-env's normal revealed-only information, not full team sheets.
"""

import argparse
import asyncio
import csv
import random
import time
from collections import Counter
from pathlib import Path

from poke_env import AccountConfiguration
from poke_env.data.normalize import to_id_str
from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer

ROOT = Path(__file__).resolve().parent.parent
TEAMS_DIR = ROOT / "Teams"
DEFAULT_OUT = ROOT / "Baselines_Log - Battles.csv"

FORMAT = "gen9ounotera"
N_PER_BLOCK = 500
CONCURRENCY = 10

OUR_FILES = {"low": TEAMS_DIR / "lowvar.txt", "high": TEAMS_DIR / "highvar.txt"}
OPP_FILES = {"A": TEAMS_DIR / "opp_a.txt", "B": TEAMS_DIR / "opp_b.txt", "C": TEAMS_DIR / "opp_c.txt"}

FIELDS = [
    "battle_tag", "team_version", "pilot", "opponent_team", "won", "turns",
    "our_remaining", "opp_remaining", "swap_moves_used", "swap_moves_missed",
    "all_moves_used", "all_moves_missed",
]


# ---------- team files ----------

def parse_team(path):
    """Return [(species_id, [move_id, ...]), ...] in slot order from a Showdown export."""
    team = []
    for block in path.read_text().strip().split("\n\n"):
        lines = [l.strip() for l in block.strip().splitlines() if l.strip()]
        head = lines[0].split(" @ ")[0].replace("(M)", "").replace("(F)", "").strip()
        if head.endswith(")") and "(" in head:  # "Nickname (Species)"
            head = head[head.rindex("(") + 1:-1]
        moves = [to_id_str(l[2:]) for l in lines if l.startswith("- ")]
        team.append((to_id_str(head), moves))
    return team


def swap_moves(low, high):
    """Diff the two versions; assert only moves differ. Returns ({low moves}, {high moves})."""
    assert [s for s, _ in low] == [s for s, _ in high], "species/order differ between versions"
    low_only, high_only = set(), set()
    for (_, lm), (_, hm) in zip(low, high):
        low_only |= set(lm) - set(hm)
        high_only |= set(hm) - set(lm)
    return low_only, high_only


OUR_TEAMS = {v: parse_team(p) for v, p in OUR_FILES.items()}
OPP_TEAMS = {k: parse_team(p) for k, p in OPP_FILES.items()}
_low_swaps, _high_swaps = swap_moves(OUR_TEAMS["low"], OUR_TEAMS["high"])
SWAP_MOVES = {"low": _low_swaps, "high": _high_swaps}
OPP_SPECIES = {k: frozenset(s for s, _ in t) for k, t in OPP_TEAMS.items()}
assert len(set(OPP_SPECIES.values())) == 3, "opponent teams must have distinct species sets"
N_TEAM = len(OUR_TEAMS["low"])


# ---------- players ----------

class FixedLeadMixin:
    """Bring the team in file order, so slot 1 always leads."""

    def teampreview(self, battle):
        for mon in battle.team.values():
            mon._selected_in_teampreview = True
        return "/team " + "".join(str(i) for i in range(1, len(battle.team) + 1))


class LeadRandom(FixedLeadMixin, RandomPlayer):
    pass


class LeadMaxBP(FixedLeadMixin, MaxBasePowerPlayer):
    pass


class LeadHeuristic(FixedLeadMixin, SimpleHeuristicsPlayer):
    pass


PILOTS = {"random": LeadRandom, "maxbp": LeadMaxBP, "heuristic": LeadHeuristic}


# ---------- per-battle extraction ----------

def identify_opponent(battle):
    species = frozenset(to_id_str(m.species) for m in battle.teampreview_opponent_team)
    matches = [k for k, s in OPP_SPECIES.items() if s == species]
    if len(matches) != 1:
        raise ValueError(f"{battle.battle_tag}: cannot identify opponent from {sorted(species)}")
    return matches[0]


def battle_to_row(battle, version, pilot, opp):
    if not battle.finished:
        raise RuntimeError(f"{battle.battle_tag} did not finish")
    me = battle.player_role
    swaps = SWAP_MOVES[version]
    leads = {me: OUR_TEAMS[version][0][0], "p2" if me == "p1" else "p1": OPP_TEAMS[opp][0][0]}

    first_switch = {}
    last_move = {}
    swap_used = swap_missed = all_used = all_missed = 0
    for ev in battle._replay_data:
        if len(ev) < 2:
            continue
        kind = ev[1]
        if kind == "-terastallize":
            raise RuntimeError(f"{battle.battle_tag}: Terastallization happened")
        if kind in ("switch", "drag"):
            side = ev[2][:2]
            first_switch.setdefault(side, to_id_str(ev[3].split(",")[0]))
        elif kind == "move":
            src = ev[2]
            move = to_id_str(ev[3])
            last_move[src] = move
            all_used += 1
            if src.startswith(me) and move in swaps:
                swap_used += 1
        elif kind == "-miss":
            src = ev[2]
            all_missed += 1
            if src.startswith(me) and last_move.get(src) in swaps:
                swap_missed += 1

    for side, lead in leads.items():
        if first_switch.get(side) != lead:
            raise RuntimeError(f"{battle.battle_tag}: {side} led {first_switch.get(side)}, expected {lead}")

    found = identify_opponent(battle)
    if found != opp:
        raise RuntimeError(f"{battle.battle_tag}: identified opponent {found}, block sent {opp}")

    return {
        "battle_tag": battle.battle_tag,
        "team_version": version,
        "pilot": pilot,
        "opponent_team": found,
        "won": int(battle.won is True),
        "turns": battle.turn,
        "our_remaining": sum(not m.fainted for m in battle.team.values()),
        "opp_remaining": N_TEAM - sum(m.fainted for m in battle.opponent_team.values()),
        "swap_moves_used": swap_used,
        "swap_moves_missed": swap_missed,
        "all_moves_used": all_used,
        "all_moves_missed": all_missed,
    }


# ---------- CSV ----------

def load_done(out):
    """Count existing rows per block; drop the placeholder row if present."""
    if not out.exists() or out.stat().st_size == 0:
        with out.open("w", newline="") as f:
            csv.DictWriter(f, FIELDS).writeheader()
        return Counter()
    with out.open(newline="") as f:
        rows = list(csv.DictReader(f))
    real = [r for r in rows if r["battle_tag"] != "battle-gen9ou-1234"]
    if len(real) != len(rows):
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, FIELDS)
            w.writeheader()
            w.writerows(real)
    return Counter((r["team_version"], r["pilot"], r["opponent_team"]) for r in real)


def append_rows(out, rows):
    with out.open("a", newline="") as f:
        csv.DictWriter(f, FIELDS).writerows(rows)


# ---------- main loop ----------

async def run_block(version, pilot, opp, n, concurrency):
    tag = f"{random.randint(0, 9999):04d}"
    pilot_player = PILOTS[pilot](
        account_configuration=AccountConfiguration(f"{pilot[:5]}{version}{opp}{tag}", None),
        battle_format=FORMAT,
        team=OUR_FILES[version].read_text(),
        max_concurrent_battles=concurrency,
    )
    opponent = LeadHeuristic(
        account_configuration=AccountConfiguration(f"opp{version}{pilot[:4]}{opp}{tag}", None),
        battle_format=FORMAT,
        team=OPP_FILES[opp].read_text(),
        max_concurrent_battles=concurrency,
    )
    await pilot_player.battle_against(opponent, n_battles=n)
    battles = list(pilot_player.battles.values())
    if len(battles) != n:
        raise RuntimeError(f"expected {n} battles, got {len(battles)}")
    rows = [battle_to_row(b, version, pilot, opp) for b in battles]
    await pilot_player.ps_client.stop_listening()
    await opponent.ps_client.stop_listening()
    return rows, sum(b.won is None for b in battles)


def matches(only, version, pilot, opp):
    return all(x in (version, pilot, opp) for x in only)


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=N_PER_BLOCK, help="battles per block")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--only", nargs="*", default=[], help="filter, e.g. --only high heuristic A")
    ap.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = ap.parse_args()

    print(f"swap moves: low={sorted(SWAP_MOVES['low'])} high={sorted(SWAP_MOVES['high'])}")
    done = load_done(args.out)
    blocks = [(v, p, o) for v in OUR_FILES for p in PILOTS for o in OPP_FILES if matches(args.only, v, p, o)]
    t_all = time.perf_counter()
    for i, (v, p, o) in enumerate(blocks, 1):
        have = done[(v, p, o)]
        if have >= args.n:
            print(f"[{i}/{len(blocks)}] {v:4} {p:9} vs {o}: {have} rows already, skipping")
            continue
        if have:
            raise RuntimeError(f"block {(v, p, o)} is partial ({have} rows); delete those rows and rerun")
        t0 = time.perf_counter()
        rows, ties = await run_block(v, p, o, args.n, args.concurrency)
        append_rows(args.out, rows)
        dt = time.perf_counter() - t0
        wins = sum(r["won"] for r in rows)
        print(f"[{i}/{len(blocks)}] {v:4} {p:9} vs {o}: {wins}/{len(rows)} won, {ties} ties, "
              f"{dt:.0f}s ({len(rows) / dt:.1f} battles/s)", flush=True)
    print(f"done in {time.perf_counter() - t_all:.0f}s -> {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
