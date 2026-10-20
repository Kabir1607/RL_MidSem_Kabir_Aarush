"""Observation encoder (spec Section 4.2). Shared by the PPO and DQN pipelines.

Layout of the float32 vector:
  Block 1: 8 rows of ROW_SIZE, ours in team-file order then the opponent's in their sheet order,
           keyed by species (teamsheet.our_mons / opp_mons).
  Block 2: active matchup (boosts, speed order, toxic counters).
  Block 3: field (hazards, screens, turn).
Everything is in [0, 1] except boosts and move priority, which are in [-1, 1].

Static parts (stats, types, ability, move power/accuracy/category/priority/STAB) come from the team
sheets and are precomputed per row once, at construction. Deliberately NOT included: damage-calculator
features such as KO probability (spec Section 4.2).
"""

import numpy as np
from gymnasium.spaces import Box
from poke_env.battle import Move, MoveCategory, SideCondition, Status
from poke_env.data import GenData

import config
import teamsheet
from teamsheet import ABILITY_VOCAB, GEN_DATA, ITEM_VOCAB, STAT_MAX, TYPE_VOCAB

N_MOVES = 4
MAX_BASE_POWER = 150
MAX_PRIORITY = 5
MAX_EFFECTIVENESS = 4
MAX_BOOST = 6
MAX_TOXIC_COUNTER = 15
MAX_TURN = 50
SCREEN_TURNS = 5  # no Light Clay on any team
SCREEN_NORM = 8

STATUSES = (None, Status.BRN, Status.PAR, Status.PSN, Status.TOX, Status.SLP, Status.FRZ)
CATEGORIES = (MoveCategory.PHYSICAL, MoveCategory.SPECIAL, MoveCategory.STATUS)
BOOSTS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")
HAZARDS = ((SideCondition.STEALTH_ROCK, 1), (SideCondition.SPIKES, 3), (SideCondition.TOXIC_SPIKES, 2),
           (SideCondition.STICKY_WEB, 1))
SCREENS = (SideCondition.REFLECT, SideCondition.LIGHT_SCREEN, SideCondition.AURORA_VEIL)
ITEM_INDEX = {item: i for i, item in enumerate(ITEM_VOCAB)}  # index len(ITEM_VOCAB) = none/removed
ABILITY_INDEX = {a: i for i, a in enumerate(ABILITY_VOCAB)}
TYPE_INDEX = {t: i for i, t in enumerate(TYPE_VOCAB)}

# Per-move features: base power, accuracy, category one-hot (3), priority, effectiveness vs the opposing
# active, STAB, PP fraction.
MOVE_FIELDS = (("bp", 1), ("acc", 1), ("cat", 3), ("prio", 1), ("eff", 1), ("stab", 1), ("pp", 1))
MOVE_SIZE = sum(n for _, n in MOVE_FIELDS)
MOVE_OFF = {}
_o = 0
for _name, _n in MOVE_FIELDS:
    MOVE_OFF[_name] = _o
    _o += _n

ROW_FIELDS = (("hp", 1), ("fainted", 1), ("status", len(STATUSES)), ("active", 1), ("stats", 6),
              ("types", len(TYPE_VOCAB)), ("item", len(ITEM_VOCAB) + 1), ("ability", len(ABILITY_VOCAB)),
              ("moves", N_MOVES * MOVE_SIZE))
ROW_SIZE = sum(n for _, n in ROW_FIELDS)
ROW_OFF = {}
_o = 0
for _name, _n in ROW_FIELDS:
    ROW_OFF[_name] = _o
    _o += _n

N_ROWS = 2 * config.N_TEAM
BLOCK1_SIZE = N_ROWS * ROW_SIZE
BLOCK2_SIZE = 2 * len(BOOSTS) + 3 + 1 + 2  # boosts, speed one-hot, speed ratio, toxic counters
BLOCK3_SIZE = 2 * len(HAZARDS) + 2 * len(SCREENS) + 1
OBS_SIZE = BLOCK1_SIZE + BLOCK2_SIZE + BLOCK3_SIZE


def move_offset(slot, field):
    return ROW_OFF["moves"] + slot * MOVE_SIZE + MOVE_OFF[field]


def boost_multiplier(stage):
    return (2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)


def effectiveness(move, defender_types):
    """Type multiplier of a damaging move against the defender's sheet types, 0 for status moves.
    Ignored: ability immunities (Levitate etc.) and move-specific exceptions (e.g. Freeze-Dry on Water)."""
    if move.category == MoveCategory.STATUS or not defender_types:
        return 0.0
    return move.type.damage_multiplier(*defender_types, type_chart=GEN_DATA.type_chart)


def current_item(sheet, mon):
    """Item id now held, or None if removed/consumed. Unrevealed opponent items fall back to the sheet."""
    if mon is None or mon.item == GenData.UNKNOWN_ITEM:
        return sheet.item or None
    return mon.item or None


def effective_speed(sheet, mon, item):
    """Spe x boost x 0.5 if paralysed x 1.5 if Choice Scarf.
    Ignored: Tailwind, Protosynthesis/Quark Drive, Unburden, weather/terrain speed abilities, Trick Room."""
    speed = sheet.stats[5] * boost_multiplier(mon.boosts.get("spe", 0))
    if mon.status == Status.PAR:
        speed *= 0.5
    if item == "choicescarf":
        speed *= 1.5
    return speed


def side_conditions(conditions, turn):
    """poke-env stores layers for stackable conditions and the start turn for the others."""
    out = [min(conditions.get(cond, 0), cap) / cap if cap > 1 else float(cond in conditions) for cond, cap in HAZARDS]
    for cond in SCREENS:
        remaining = SCREEN_TURNS - (turn - conditions[cond]) if cond in conditions else 0
        out.append(max(remaining, 0) / SCREEN_NORM)
    return out


def static_row(sheet):
    """The parts of a row that never change during a battle."""
    row = np.zeros(ROW_SIZE, dtype=np.float32)
    o = ROW_OFF["stats"]
    row[o:o + 6] = np.array(sheet.stats) / np.array(STAT_MAX)
    for t in sheet.types:
        row[ROW_OFF["types"] + TYPE_INDEX[t]] = 1.0
    row[ROW_OFF["ability"] + ABILITY_INDEX[sheet.ability]] = 1.0
    for slot, move_id in enumerate(sheet.moves):
        move = Move(move_id, gen=9)
        row[move_offset(slot, "bp")] = min(move.base_power / MAX_BASE_POWER, 1.0)
        row[move_offset(slot, "acc")] = move.accuracy  # never-miss moves are 1.0
        row[move_offset(slot, "cat") + CATEGORIES.index(move.category)] = 1.0
        row[move_offset(slot, "prio")] = np.clip(move.priority / MAX_PRIORITY, -1.0, 1.0)
        row[move_offset(slot, "stab")] = float(move.category != MoveCategory.STATUS and move.type in sheet.types)
    return row


class Encoder:
    """embed(battle) -> float32 vector for our side of a battle; observation_space declares its bounds."""

    def __init__(self, team):
        self.team = team
        self.our_sheets = teamsheet.OUR_SHEETS[team]
        self.our_static = [static_row(s) for s in self.our_sheets]
        self.opp_static = {key: [static_row(s) for s in sheets] for key, sheets in teamsheet.OPP_SHEETS.items()}
        self.moves = {s.species: [Move(m, gen=9) for m in s.moves]
                      for s in (*self.our_sheets, *(s for sheets in teamsheet.OPP_SHEETS.values() for s in sheets))}
        low = np.zeros(OBS_SIZE, dtype=np.float32)
        for r in range(N_ROWS):
            for slot in range(N_MOVES):
                low[r * ROW_SIZE + move_offset(slot, "prio")] = -1.0
        low[BLOCK1_SIZE:BLOCK1_SIZE + 2 * len(BOOSTS)] = -1.0
        self.observation_space = Box(low, np.ones(OBS_SIZE, dtype=np.float32), dtype=np.float32)

    def _row(self, static, sheet, mon, is_active, defender):
        """Fill the dynamic parts of one row. mon is None for an unrevealed opponent (full HP, sheet item)."""
        row = static.copy()
        fainted = mon is not None and mon.fainted
        row[ROW_OFF["hp"]] = 1.0 if mon is None else mon.current_hp_fraction
        row[ROW_OFF["fainted"]] = float(fainted)
        status = None if mon is None or fainted else mon.status
        row[ROW_OFF["status"] + STATUSES.index(status)] = 1.0
        row[ROW_OFF["active"]] = float(is_active)
        item = current_item(sheet, mon)
        row[ROW_OFF["item"] + (ITEM_INDEX[item] if item in ITEM_INDEX else len(ITEM_VOCAB))] = 1.0
        defender_types = defender.types if defender is not None else ()
        for slot, move in enumerate(self.moves[sheet.species]):
            row[move_offset(slot, "eff")] = effectiveness(move, defender_types) / MAX_EFFECTIVENESS
            seen = mon.moves.get(move.id) if mon is not None else None
            row[move_offset(slot, "pp")] = seen.current_pp / seen.max_pp if seen is not None else 1.0
        return row

    def embed(self, battle):
        opp_key = teamsheet.identify_opponent(battle)
        opp_sheets = teamsheet.OPP_SHEETS[opp_key]
        ours = teamsheet.our_mons(battle, self.team)
        theirs = teamsheet.opp_mons(battle)
        our_act = next((i for i, m in enumerate(ours) if m.active and not m.fainted), None)
        opp_act = next((i for i, m in enumerate(theirs) if m is not None and m.active and not m.fainted), None)
        our_def = self.our_sheets[our_act] if our_act is not None else None  # defender for opponent moves
        opp_def = opp_sheets[opp_act] if opp_act is not None else None  # defender for our moves

        rows = [self._row(self.our_static[i], self.our_sheets[i], ours[i], i == our_act, opp_def)
                for i in range(config.N_TEAM)]
        rows += [self._row(self.opp_static[opp_key][i], opp_sheets[i], theirs[i], i == opp_act, our_def)
                 for i in range(config.N_TEAM)]

        # Block 2: active matchup
        block2 = np.zeros(BLOCK2_SIZE, dtype=np.float32)
        actives = [(ours[our_act], self.our_sheets[our_act]) if our_act is not None else None,
                   (theirs[opp_act], opp_sheets[opp_act]) if opp_act is not None else None]
        speeds = []
        for side, active in enumerate(actives):
            if active is None:
                continue
            mon, sheet = active
            block2[side * len(BOOSTS):(side + 1) * len(BOOSTS)] = [mon.boosts.get(b, 0) / MAX_BOOST for b in BOOSTS]
            if mon.status == Status.TOX:
                block2[2 * len(BOOSTS) + 4 + side] = min(mon.status_counter / MAX_TOXIC_COUNTER, 1.0)
            speeds.append(effective_speed(sheet, mon, current_item(sheet, mon)))
        o = 2 * len(BOOSTS)
        if len(speeds) == 2:
            ours_spe, theirs_spe = speeds
            block2[o + (0 if ours_spe > theirs_spe else 1 if ours_spe == theirs_spe else 2)] = 1.0
            block2[o + 3] = ours_spe / (ours_spe + theirs_spe)
        else:
            block2[o + 3] = 0.5

        # Block 3: field
        block3 = np.array(side_conditions(battle.side_conditions, battle.turn)
                          + side_conditions(battle.opponent_side_conditions, battle.turn)
                          + [min(battle.turn / MAX_TURN, 1.0)], dtype=np.float32)
        return np.concatenate(rows + [block2, block3])
