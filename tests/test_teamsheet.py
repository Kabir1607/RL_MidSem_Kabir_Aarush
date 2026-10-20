"""Spec Section 8 tests 1 (team diff and rules) and 2 (stats), plus opponent identification from species."""

import dataclasses
import json
import shutil
import subprocess
from pathlib import Path

import pytest

import config
import teamsheet
from check_teams import banned_move_reason, diff_versions, run_checks

ROOT = Path(__file__).resolve().parent.parent
ALL_FILES = {**{f"our_{k}": p for k, p in config.OUR_TEAM_FILES.items()},
             **{f"opp_{k}": p for k, p in config.OPP_TEAM_FILES.items()}}
SHEETS = {**{f"our_{k}": s for k, s in teamsheet.OUR_SHEETS.items()},
          **{f"opp_{k}": s for k, s in teamsheet.OPP_SHEETS.items()}}


# ---------- test 1: team diff and rules ----------

def test_team_checks_pass():
    problems, swaps = run_checks()
    assert problems == []
    assert {(s, a, b) for s, _, a, b in swaps} == {
        ("moltres", "flamethrower", "fireblast"), ("moltres", "airslash", "hurricane"),
        ("primarina", "surf", "hydropump"), ("primarina", "icebeam", "blizzard"),
    }


@pytest.mark.parametrize("move", ["raindance", "sunnyday", "sandattack", "doubleteam", "mudslap", "minimize",
                                  "gravity", "trick", "switcheroo", "transform"])
def test_banned_moves_detected(move):
    assert banned_move_reason(move) is not None


@pytest.mark.parametrize("move", ["surf", "hydropump", "knockoff", "stealthrock", "swordsdance", "uturn"])
def test_allowed_moves_pass(move):
    assert banned_move_reason(move) is None


def test_diff_catches_non_move_change():
    low = teamsheet.OUR_SHEETS["lowvar"]
    high = list(teamsheet.OUR_SHEETS["highvar"])
    high[0] = dataclasses.replace(high[0], item="choicespecs")
    problems, _ = diff_versions(low, high)
    assert any("item differs" in p for p in problems)


def test_diff_catches_wrong_direction_swap():
    low, high = list(teamsheet.OUR_SHEETS["lowvar"]), list(teamsheet.OUR_SHEETS["highvar"])
    low[2], high[2] = high[2], low[2]  # Primarina's swaps reversed
    problems, _ = diff_versions(low, high)
    assert any("not a power-for-accuracy swap" in p for p in problems)


# ---------- test 2: stats equal Showdown's ----------

@pytest.fixture(scope="module")
def showdown_stats():
    if shutil.which("node") is None:
        pytest.skip("node not installed")
    files = [str(p) for p in ALL_FILES.values()]
    out = subprocess.run(["node", str(ROOT / "tests" / "showdown_stats.js"), *files],
                         capture_output=True, text=True, check=True, cwd=ROOT)
    by_path = json.loads(out.stdout)
    return {name: by_path[str(p)] for name, p in ALL_FILES.items()}


@pytest.mark.parametrize("name", list(ALL_FILES))
def test_stats_match_showdown(name, showdown_stats):
    ours = [(m.species, list(m.stats)) for m in SHEETS[name]]
    theirs = [(m["species"], m["stats"]) for m in showdown_stats[name]]
    assert ours == theirs


def test_normalisers_cover_all_16():
    assert len(teamsheet.ALL_MONS) == 16
    for m in teamsheet.ALL_MONS.values():
        assert all(0 < s / mx <= 1 for s, mx in zip(m.stats, teamsheet.STAT_MAX))


# ---------- opponent identification (from species; the live-battle version is Phase 2 test 4) ----------

@pytest.mark.parametrize("key", list(config.OPP_TEAM_FILES))
def test_match_opponent_by_species_and_lead(key):
    sheets = teamsheet.OPP_SHEETS[key]
    assert teamsheet.match_opponent(frozenset(m.species for m in sheets)) == key
    assert teamsheet.match_opponent(lead=sheets[0].species) == key


def test_match_opponent_rejects_unknown():
    with pytest.raises(ValueError):
        teamsheet.match_opponent(frozenset(m.species for m in teamsheet.OUR_SHEETS["lowvar"]))
    with pytest.raises(ValueError):
        teamsheet.match_opponent(lead="pikachu")
    with pytest.raises(ValueError):
        teamsheet.match_opponent()
