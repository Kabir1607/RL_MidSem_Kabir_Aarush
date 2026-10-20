"""Opponent teambuilders: seeded random rotation and stratified evaluation order."""

from collections import Counter

from teams import OPP_PACKED, RandomOpponentTeambuilder, StratifiedOpponentTeambuilder


def test_random_rotation_is_seeded_and_uniformish():
    a, b = RandomOpponentTeambuilder(0), RandomOpponentTeambuilder(0)
    seq = [a.next_key() for _ in range(3000)]
    assert seq == [b.next_key() for _ in range(3000)]
    assert all(900 < c < 1100 for c in Counter(seq).values())


def test_stratified_order_and_restart():
    tb = StratifiedOpponentTeambuilder(40)
    first = [tb.next_key() for _ in range(120)]
    assert first == ["A"] * 40 + ["B"] * 40 + ["C"] * 40
    tb.next_key()
    tb.restart()
    assert [tb.next_key() for _ in range(120)] == first


def test_yield_team_records_what_was_sent():
    tb = StratifiedOpponentTeambuilder(1)
    teams = [tb.yield_team() for _ in range(3)]
    assert tb.sent == ["A", "B", "C"]
    assert teams == [OPP_PACKED[k] for k in "ABC"]
