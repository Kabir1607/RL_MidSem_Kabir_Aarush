"""Spec Section 8 test 3 (encoder), plus offline checks of the static features the experiment relies on."""

import numpy as np
import pytest
from poke_env.battle import Move
from poke_env.data.normalize import to_id_str

import config
import encoder as E
import teamsheet
from env import make_env

N_BATTLES = 20


# ---------- offline ----------

def test_sizes():
    assert E.ROW_SIZE == 70 + len(teamsheet.ITEM_VOCAB) + 1 + len(teamsheet.ABILITY_VOCAB)
    assert E.OBS_SIZE == 8 * E.ROW_SIZE + 20 + 15
    assert E.Encoder("lowvar").observation_space.shape == (E.OBS_SIZE,)


def test_lowvar_highvar_static_rows_differ_only_in_swapped_moves():
    low, high = E.Encoder("lowvar"), E.Encoder("highvar")
    expected = {(1, 0), (1, 1), (2, 0), (2, 2)}  # (row, move slot): Moltres 1-2, Primarina 1 and 3
    changed = set()
    for r, (a, b) in enumerate(zip(low.our_static, high.our_static)):
        for i in np.flatnonzero(a != b):
            slot, field_off = divmod(int(i) - E.ROW_OFF["moves"], E.MOVE_SIZE)
            assert i >= E.ROW_OFF["moves"] and field_off in (E.MOVE_OFF["bp"], E.MOVE_OFF["acc"]), (r, i)
            changed.add((r, slot))
    assert changed == expected
    for r, slot in expected:
        assert high.our_static[r][E.move_offset(slot, "acc")] < low.our_static[r][E.move_offset(slot, "acc")]
        assert high.our_static[r][E.move_offset(slot, "bp")] > low.our_static[r][E.move_offset(slot, "bp")]
    assert high.our_static[2][E.move_offset(0, "acc")] == pytest.approx(0.8)  # Hydro Pump


def test_effectiveness():
    gholdengo = teamsheet.ALL_MONS["gholdengo"]
    corviknight = teamsheet.ALL_MONS["corviknight"]
    assert E.effectiveness(Move("headlongrush", gen=9), gholdengo.types) == 2  # Ground vs Steel/Ghost
    assert E.effectiveness(Move("headlongrush", gen=9), corviknight.types) == 0  # Ground vs Flying
    assert E.effectiveness(Move("swordsdance", gen=9), gholdengo.types) == 0  # status move
    assert E.effectiveness(Move("surf", gen=9), ()) == 0  # no defender


def test_boost_multiplier():
    assert [E.boost_multiplier(s) for s in (-6, -1, 0, 1, 6)] == [0.25, 2 / 3, 1, 1.5, 4]


# ---------- live: 20 battles with random legal actions ----------

def row(obs, r):
    return obs[r * E.ROW_SIZE:(r + 1) * E.ROW_SIZE]


@pytest.fixture(scope="module")
def checked(showdown):
    """Play battles and check every observation against the battle it came from; return the failures."""
    team = "highvar"
    env = make_env(team, seed=1, algo="tst", monitor=False)
    enc = env.env.env.encoder
    rng = np.random.default_rng(1)
    failures, n_obs = [], 0
    try:
        for _ in range(N_BATTLES):
            obs, _ = env.reset()
            first = True
            done = False
            while not done:
                failures += check_obs(obs, env.battle, team, enc.observation_space, first)
                n_obs += 1
                first = False
                obs, _, terminated, truncated, _ = env.step(rng.choice(np.flatnonzero(env.action_masks())))
                done = terminated or truncated
            failures += check_obs(obs, env.battle, team, enc.observation_space, False)
    finally:
        env.close()
    return failures, n_obs


def check_obs(obs, battle, team, space, first):
    tag = battle.battle_tag
    out = []
    if obs.shape != (E.OBS_SIZE,) or obs.dtype != np.float32:
        return [f"{tag}: bad shape/dtype {obs.shape} {obs.dtype}"]
    if not np.all(np.isfinite(obs)):
        out.append(f"{tag}: non-finite values")
    if not space.contains(obs):
        bad = np.flatnonzero((obs < space.low) | (obs > space.high))
        out.append(f"{tag}: out of bounds at {bad.tolist()[:10]}")
    mons = teamsheet.our_mons(battle, team) + teamsheet.opp_mons(battle)
    sheets = teamsheet.OUR_SHEETS[team] + teamsheet.OPP_SHEETS[teamsheet.identify_opponent(battle)]
    for r, (mon, sheet) in enumerate(zip(mons, sheets)):
        rw = row(obs, r)
        hp = 1.0 if mon is None else mon.current_hp_fraction
        if rw[E.ROW_OFF["hp"]] != pytest.approx(hp, abs=1e-6):  # HP stays in its own species' row
            out.append(f"{tag} turn {battle.turn}: row {r} ({sheet.species}) hp {rw[E.ROW_OFF['hp']]} != {hp}")
        if mon is not None and mon.species != sheet.species:
            out.append(f"{tag}: row {r} holds {mon.species}, expected {sheet.species}")
        if first and (not np.all(rw[E.ROW_OFF["stats"]:E.ROW_OFF["stats"] + 6] > 0)
                      or not all(rw[E.move_offset(s, "acc")] > 0 for s in range(E.N_MOVES))):
            out.append(f"{tag}: row {r} missing stats or moves at turn 1")
    for side in (0, 1):
        n_active = sum(row(obs, r)[E.ROW_OFF["active"]] for r in range(side * 4, side * 4 + 4))
        if n_active > 1:
            out.append(f"{tag}: {n_active} active rows on side {side}")
    active = battle.active_pokemon  # still the fainted mon while we choose its replacement
    if active is not None and not active.fainted and not battle.finished:
        r = [s.species for s in teamsheet.OUR_SHEETS[team]].index(to_id_str(active.species))
        if row(obs, r)[E.ROW_OFF["active"]] != 1:
            out.append(f"{tag}: our active {active.species} not flagged")
        for slot, move in enumerate(list(active.moves.values())[:4]):  # move slot i = action 6 + i
            if row(obs, r)[E.move_offset(slot, "bp")] != pytest.approx(min(move.base_power / 150, 1)):
                out.append(f"{tag}: move slot {slot} of {active.species} is not {move.id}")
    return out


def test_encoder_live(checked):
    failures, n_obs = checked
    assert n_obs > N_BATTLES * 5
    assert failures == []
