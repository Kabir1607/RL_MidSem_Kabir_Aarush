"""Open team sheets (spec Section 3.2): parse the 5 team files, compute actual stats, build the
vocabularies and normalisers the encoder uses, and identify which opponent file a battle is using.

Everything static is computed once at import from the files named in config.py.
"""

from dataclasses import dataclass

from poke_env.battle import PokemonType
from poke_env.data import GenData
from poke_env.data.normalize import to_id_str
from poke_env.stats import compute_raw_stats
from poke_env.teambuilder import Teambuilder

import config

GEN_DATA = GenData.from_gen(9)
STAT_NAMES = ("hp", "atk", "def", "spa", "spd", "spe")

# Showdown exports omit these values
DEFAULT_LEVEL = 100
DEFAULT_IV = 31
DEFAULT_EV = 0
DEFAULT_NATURE = "serious"


@dataclass(frozen=True)
class MonSheet:
    """One Pokémon's full set. Every name is a Showdown id (lowercase, no punctuation)."""
    species: str
    item: str
    ability: str
    nature: str
    level: int
    evs: tuple
    ivs: tuple
    moves: tuple  # move ids in file order
    stats: tuple  # actual stats (hp, atk, def, spa, spd, spe)
    types: tuple  # PokemonType, 1 or 2 entries


def parse_mon(tb_mon):
    species = to_id_str(tb_mon.species or tb_mon.nickname)
    if species not in GEN_DATA.pokedex:
        raise ValueError(f"unknown species {species!r}")
    level = tb_mon.level or DEFAULT_LEVEL
    evs = tuple(tb_mon.evs or [DEFAULT_EV] * 6)
    ivs = tuple(tb_mon.ivs or [DEFAULT_IV] * 6)
    nature = to_id_str(tb_mon.nature or DEFAULT_NATURE)
    return MonSheet(
        species=species,
        item=to_id_str(tb_mon.item or ""),
        ability=to_id_str(tb_mon.ability or ""),
        nature=nature,
        level=level,
        evs=evs,
        ivs=ivs,
        moves=tuple(to_id_str(m) for m in tb_mon.moves),
        stats=tuple(compute_raw_stats(species, list(evs), list(ivs), level, nature, GEN_DATA)),
        types=tuple(PokemonType.from_name(t) for t in GEN_DATA.pokedex[species]["types"]),
    )


def parse_team_file(path):
    """Return [MonSheet, ...] in slot order from a Showdown export."""
    return [parse_mon(m) for m in Teambuilder.parse_showdown_team(path.read_text().strip())]


OUR_SHEETS = {team: parse_team_file(p) for team, p in config.OUR_TEAM_FILES.items()}
OPP_SHEETS = {key: parse_team_file(p) for key, p in config.OPP_TEAM_FILES.items()}

# The 16 distinct Pokémon (lowvar and highvar share species and stats; check_teams.py asserts this).
ALL_MONS = {m.species: m for sheets in (*OUR_SHEETS.values(), *OPP_SHEETS.values()) for m in sheets}

# ---------- vocabularies and normalisers (spec Section 4.2) ----------

STAT_MAX = tuple(max(m.stats[i] for m in ALL_MONS.values()) for i in range(6))
ITEM_VOCAB = tuple(sorted({m.item for m in ALL_MONS.values()} - {""}))  # encoder adds one "none/removed" slot
ABILITY_VOCAB = tuple(sorted({m.ability for m in ALL_MONS.values()}))
TYPE_VOCAB = tuple(t for t in PokemonType if t not in (PokemonType.THREE_QUESTION_MARKS, PokemonType.STELLAR))
assert len(TYPE_VOCAB) == 18

# ---------- opponent identification ----------

OPP_SPECIES = {key: frozenset(m.species for m in sheets) for key, sheets in OPP_SHEETS.items()}
OPP_LEADS = {key: sheets[0].species for key, sheets in OPP_SHEETS.items()}


def match_opponent(species=frozenset(), lead=None):
    """Return the opponent key whose species set equals `species`; if `species` is empty, the key
    whose lead is `lead`. Raises ValueError unless exactly one file matches."""
    if species:
        matches = [k for k, s in OPP_SPECIES.items() if s == species]
    else:
        matches = [k for k, s in OPP_LEADS.items() if s == lead]
    if len(matches) != 1:
        raise ValueError(f"cannot identify opponent from species={sorted(species)} lead={lead}: {matches}")
    return matches[0]


_identified = {}


def identify_opponent(battle):
    """Opponent key ('A'/'B'/'C') for a battle, from team preview with the lead as fallback.

    Cached per (player, battle_tag): both env players see the same battle_tag (NOTES.md item 3).
    """
    cache_key = (battle.player_username, battle.battle_tag)
    if cache_key not in _identified:
        species = frozenset(to_id_str(m.species) for m in battle.teampreview_opponent_team)
        active = battle.opponent_active_pokemon
        lead = to_id_str(active.species) if active is not None else None
        _identified[cache_key] = match_opponent(species, lead)
    return _identified[cache_key]


def forget_battle(battle):
    """Drop cached state for a finished battle."""
    _identified.pop((battle.player_username, battle.battle_tag), None)


# ---------- battle Pokémon keyed by species (spec Section 4.2 indexing rule) ----------

def our_mons(battle, team):
    """Our 4 battle Pokémon in team-file order (rows 0-3)."""
    by_species = {to_id_str(m.species): m for m in battle.team.values()}
    return [by_species[s.species] for s in OUR_SHEETS[team]]


def opp_mons(battle):
    """The opponent's 4 battle Pokémon in their sheet order (rows 4-7); None if not revealed yet.

    Team-preview Pokémon objects have max_hp 0 (and so HP fraction 0), so they count as unrevealed.
    """
    revealed = {to_id_str(m.species): m for m in battle.opponent_team.values() if m.max_hp}
    return [revealed.get(s.species) for s in OPP_SHEETS[identify_opponent(battle)]]
