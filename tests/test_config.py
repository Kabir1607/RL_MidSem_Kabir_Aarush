"""Showdown account names: short enough to survive Showdown's name filter, and unique."""

import itertools

import config


def test_account_names_fit_and_are_unique():
    combos = list(itertools.product(("ppo", "dqn"), config.TEAMS, range(10), config.ROLES))
    names = [config.account_name(*c) for c in combos]
    assert max(map(len, names)) <= config.MAX_USERNAME_LEN
    assert len({n.lower().replace("_", "") for n in names}) == len(combos)


def test_launch_tag_changes_names():
    assert config.account_name("ppo", "lowvar", 0, "agent", tag="0000") != config.account_name("ppo", "lowvar", 0, "agent", tag="0001")
