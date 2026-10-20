"""Teambuilders (spec Section 7): a fixed team for us; random (training) or stratified (evaluation)
opponent teams, one per battle."""

import random

from poke_env.teambuilder import Teambuilder

import config


def packed(path):
    return Teambuilder.join_team(Teambuilder.parse_showdown_team(path.read_text().strip()))


OPP_PACKED = {key: packed(p) for key, p in config.OPP_TEAM_FILES.items()}


class OpponentTeambuilder(Teambuilder):
    """Yields one opponent team per battle and records which key it sent, in order."""

    def __init__(self):
        self.sent = []

    def next_key(self):
        raise NotImplementedError

    def yield_team(self):
        key = self.next_key()
        self.sent.append(key)
        return OPP_PACKED[key]


class RandomOpponentTeambuilder(OpponentTeambuilder):
    """Uniformly random opponent team per training battle, from a seeded RNG."""

    def __init__(self, seed):
        super().__init__()
        self.rng = random.Random(seed)

    def next_key(self):
        return self.rng.choice(config.EVAL_OPP_ORDER)


class StratifiedOpponentTeambuilder(OpponentTeambuilder):
    """n_per_team battles against each team in a fixed order (A..., B..., C...), then repeats.
    Call restart() before each evaluation."""

    def __init__(self, n_per_team, order=config.EVAL_OPP_ORDER):
        super().__init__()
        self.schedule = [key for key in order for _ in range(n_per_team)]
        self.restart()

    def restart(self):
        self.i = 0

    def next_key(self):
        key = self.schedule[self.i % len(self.schedule)]
        self.i += 1
        return key
