"""Spec Section 8 tests 4 (opponent identification), 5 (mask) and 6 (reward), plus the action-index
alignment and fixed-lead checks from NOTES.md. Plays real battles with random legal actions against the
heuristic opponent; needs the local Showdown server (skipped if it is not running).
"""

import numpy as np
import pytest
from poke_env.data.normalize import to_id_str

import config
import teamsheet
from env import NEVER_LEGAL, make_env

N_BATTLES = {"lowvar": 20, "highvar": 10}


def play(team, n_battles, seed=0):
    """Play n battles with random legal actions; return one record per battle."""
    env = make_env(team, seed, algo="tst", monitor=False)
    rng = np.random.default_rng(seed)
    records = []
    try:
        for _ in range(n_battles):
            env.reset()
            battle = env.battle
            rec = dict(masks=[env.action_masks()], rewards=[], phis=[0.0], stale=0, misaligned=[])
            done = False
            while not done:
                active = battle.active_pokemon
                if active is not None:
                    sheet = next(s for s in teamsheet.OUR_SHEETS[team] if s.species == to_id_str(active.species))
                    if list(active.moves) != list(sheet.moves):
                        rec["misaligned"].append((list(active.moves), sheet.moves))
                _, reward, terminated, truncated, info = env.step(rng.choice(np.flatnonzero(env.action_masks())))
                done = terminated or truncated
                rec["rewards"].append(reward)
                rec["phis"].append(env._phi_prev)
                if not done:
                    rec["masks"].append(env.action_masks())
                    rec["stale"] += not env.env.env.agent1_to_move
            rec.update(info["battle_end"], battle=battle, team_order=[to_id_str(m.species) for m in battle.team.values()])
            records.append(rec)
        sent = list(env.opp_teams.sent)
    finally:
        env.close()
    return records, sent


@pytest.fixture(scope="module", params=list(N_BATTLES))
def played(request, showdown):
    team = request.param
    records, sent = play(team, N_BATTLES[team])
    return team, records, sent


def first_switches(battle):
    out = {}
    for ev in battle._replay_data:
        if len(ev) > 3 and ev[1] in ("switch", "drag"):
            out.setdefault(ev[2][:2], to_id_str(ev[3].split(",")[0]))
    return out


# ---------- test 4 ----------

def test_opponent_identification(played):
    _, records, sent = played
    assert len(sent) == len(records), "opponent teambuilder not called exactly once per battle"
    assert [r["opp_team"] for r in records] == sent


# ---------- test 5 ----------

def test_masks(played):
    _, records, _ = played
    for r in records:
        for m in r["masks"]:
            assert m.dtype == bool and m.shape == (config.ACTION_SPACE_SIZE,)
            assert m.any()
            assert not m[NEVER_LEGAL].any()


def test_no_terastallization(played):
    _, records, _ = played
    for r in records:
        assert not any(len(ev) > 1 and ev[1] == "-terastallize" for ev in r["battle"]._replay_data)


# ---------- test 6 ----------

def test_terminal_reward_sign(played):
    _, records, _ = played
    for r in records:
        b = r["battle"]
        assert not r["truncated"]
        assert r["outcome"] == (1 if b.won else (-1 if b.lost else 0))
        # last reward = outcome + gamma * 0 - Phi(s_{T-1})
        assert r["rewards"][-1] == pytest.approx(r["outcome"] - r["phis"][-2], abs=1e-12)


def test_discounted_shaping_sums_to_zero(played):
    _, records, _ = played
    for r in records:
        assert abs(r["disc_shaping_sum"]) < 1e-6
        disc = sum(config.GAMMA ** k * (config.GAMMA * r["phis"][k + 1] - r["phis"][k]) for k in range(len(r["rewards"])))
        assert disc == pytest.approx(r["disc_shaping_sum"], abs=1e-12)


def test_shaping_per_step_bounded(played):
    _, records, _ = played
    for r in records:
        shaping = np.array(r["rewards"]) - np.r_[np.zeros(len(r["rewards"]) - 1), r["outcome"]]
        assert np.all(np.abs(shaping) <= 2)


# ---------- MDP bookkeeping (NOTES.md items 3-5) ----------

def test_no_stale_steps_reach_agent(played):
    _, records, _ = played
    assert all(r["stale"] == 0 for r in records)
    assert all(r["len_steps"] == len(r["rewards"]) for r in records)


def test_action_indices_follow_team_file(played):
    team, records, _ = played
    file_order = [s.species for s in teamsheet.OUR_SHEETS[team]]
    for r in records:
        assert r["team_order"] == file_order
        assert r["misaligned"] == []


def test_slot_one_leads(played):
    team, records, _ = played
    for r in records:
        b = r["battle"]
        opp_role = "p2" if b.player_role == "p1" else "p1"
        assert first_switches(b) == {b.player_role: teamsheet.OUR_SHEETS[team][0].species,
                                     opp_role: teamsheet.OPP_SHEETS[r["opp_team"]][0].species}
