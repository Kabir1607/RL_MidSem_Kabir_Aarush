"""The battle environment (spec Sections 3-4): poke-env SinglesEnv + SingleAgentWrapper, wrapped so that
one step is one decision request, the reward is the potential-shaped reward with Phi(terminal) = 0,
and MaskablePPO can read the action mask. See NOTES.md for why each piece is needed.
"""

import numpy as np
from gymnasium import Wrapper
from poke_env import AccountConfiguration
from poke_env.environment import SingleAgentWrapper, SinglesEnv
from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer
from stable_baselines3.common.monitor import Monitor

import config
import teamsheet
from teams import RandomOpponentTeambuilder, StratifiedOpponentTeambuilder

OPPONENTS = {"heuristic": SimpleHeuristicsPlayer, "random": RandomPlayer, "maxbp": MaxBasePowerPlayer}
ROLE_ACCOUNTS = {"train": ("agent", "opp"), "eval": ("eval", "evopp")}
NEVER_LEGAL = np.r_[config.N_TEAM:6, 10:config.ACTION_SPACE_SIZE]  # empty team slots; mega/z/dynamax/tera


def fixed_teampreview(battle):
    """Bring the team in file order so slot 1 always leads (as FixedLeadMixin in src/baselines.py)."""
    for mon in battle.team.values():
        mon._selected_in_teampreview = True
    return "/team " + "".join(str(i) for i in range(1, config.N_TEAM + 1))


def side_totals(mons):
    """(sum of HP fractions, number alive); None = unrevealed opponent = full HP, alive."""
    hp = sum(1.0 if m is None else m.current_hp_fraction for m in mons)
    alive = sum(m is None or not m.fainted for m in mons)
    return hp, alive


def potential(battle, team):
    """Phi(s) in [-1, 1] (spec Section 4.4)."""
    hp_o, al_o = side_totals(teamsheet.our_mons(battle, team))
    hp_t, al_t = side_totals(teamsheet.opp_mons(battle))
    return 0.5 * (hp_o - hp_t) / config.N_TEAM + 0.5 * (al_o - al_t) / config.N_TEAM


class PPOSinglesEnv(SinglesEnv):
    def __init__(self, encoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.observation_spaces = {agent: encoder.observation_space for agent in self.possible_agents}

    def embed_battle(self, battle):
        if battle.player_username != self.agent1.username:
            return np.zeros(self.encoder.observation_space.shape, dtype=np.float32)  # opponent side, never used
        return self.encoder.embed(battle)

    def calc_reward(self, battle):
        return 0.0  # computed once per step in OTSWrapper instead (NOTES.md item 3)


class OTSWrapper(Wrapper):
    """Around SingleAgentWrapper:

    * returns the flat observation and exposes action_masks() for MaskablePPO;
    * steps through requests where our action would be ignored ("wait"), so one step = one decision;
    * computes r' = r + gamma * Phi(s') - Phi(s) with Phi(terminal) = 0, once per step, for our side only;
    * terminated = battle finished (ties included); truncated only at the MAX_EPISODE_STEPS cap;
    * at the end of an episode, puts a summary in info["battle_end"].
    """

    def __init__(self, env, team, gamma=config.GAMMA, max_steps=config.MAX_EPISODE_STEPS):
        super().__init__(env)
        self.observation_space = env.observation_space["observation"]
        self.team = team
        self.gamma = gamma
        self.max_steps = max_steps
        self._mask = None

    @property
    def battle(self):
        return self.env.env.battle1

    def action_masks(self):
        return self._mask.astype(bool)

    def _set_mask(self, obs, terminal):
        self._mask = obs["action_mask"]
        if terminal:
            return
        if not self._mask.any():
            raise RuntimeError(f"{self.battle.battle_tag}: empty action mask")
        if self._mask[NEVER_LEGAL].any():
            raise RuntimeError(f"{self.battle.battle_tag}: illegal index in mask {np.flatnonzero(self._mask)}")

    def reset(self, *, seed=None, options=None):
        if self.battle is not None:
            teamsheet.forget_battle(self.battle)
        obs, info = self.env.reset(seed=seed, options=options)
        self._set_mask(obs, terminal=False)
        phi0 = potential(self.battle, self.team)
        if abs(phi0) > 1e-9:
            raise RuntimeError(f"{self.battle.battle_tag}: Phi(s_0) = {phi0}, expected 0")
        self._phi_prev = 0.0
        self._k = 0
        self._disc_shaping = 0.0
        self._shaped_return = 0.0
        self._waits = 0
        return obs["observation"], info

    def step(self, action):
        obs, _, _, _, info = self.env.step(action)
        while not self.battle.finished and not self.env.env.agent1_to_move:
            obs, _, _, _, info = self.env.step(np.int64(0))  # ignored by poke-env; only the opponent acts
            self._waits += 1
        battle = self.battle
        terminated = battle.finished
        if terminated:
            r = 1.0 if battle.won else (-1.0 if battle.lost else 0.0)
            phi = 0.0  # Phi(terminal) = 0
        else:
            r, phi = 0.0, potential(battle, self.team)
        shaping = self.gamma * phi - self._phi_prev
        self._disc_shaping += self.gamma ** self._k * shaping
        self._k += 1
        self._phi_prev = phi
        reward = r + shaping
        self._shaped_return += reward
        truncated = not terminated and self._k >= self.max_steps
        self._set_mask(obs, terminal=terminated)
        if terminated or truncated:
            info = dict(info, battle_end=dict(
                battle_tag=battle.battle_tag,
                opp_team=teamsheet.identify_opponent(battle),
                outcome=int(r),
                truncated=truncated,
                len_steps=self._k,
                len_turns=battle.turn,
                wait_skips=self._waits,
                disc_shaping_sum=self._disc_shaping,
                ep_return_shaped=self._shaped_return,
            ))
        return obs["observation"], reward, terminated, truncated, info


def make_env(team, seed, role="train", opponent="heuristic", algo="ppo", opp_per_team=None, encoder=None,
             monitor=True):
    """Build one wrapped env.

    role: "train" or "eval" (picks the Showdown account names).
    opponent: key of OPPONENTS; plays the opponent team files.
    opp_per_team: None = random opponent team per battle (seeded); int = stratified, that many per team.
    """
    agent_role, opp_role = ROLE_ACCOUNTS[role]
    if encoder is None:
        from encoder import Encoder
        encoder = Encoder(team)
    env = PPOSinglesEnv(
        encoder,
        account_configuration1=AccountConfiguration(config.account_name(algo, team, seed, agent_role), None),
        account_configuration2=AccountConfiguration(config.account_name(algo, team, seed, opp_role), None),
        battle_format=config.BATTLE_FORMAT,
        server_configuration=config.SERVER_CONFIGURATION,
        team=config.OUR_TEAM_FILES[team].read_text(),
        choose_on_teampreview=False,
        log_level=40,
        strict=True,
    )
    opp_teams = RandomOpponentTeambuilder(seed) if opp_per_team is None else StratifiedOpponentTeambuilder(opp_per_team)
    env.agent2.update_team(opp_teams)
    env.agent1.teampreview = fixed_teampreview
    env.agent2.teampreview = fixed_teampreview
    opponent_player = OPPONENTS[opponent](battle_format=config.BATTLE_FORMAT, start_listening=False)
    wrapped = OTSWrapper(SingleAgentWrapper(env, opponent_player), team)
    wrapped.opp_teams = opp_teams
    return Monitor(wrapped) if monitor else wrapped
