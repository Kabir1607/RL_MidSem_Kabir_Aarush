"""All constants for the RL runs (spec Section 7). Shared by the PPO and DQN pipelines.

Never edit this file in the middle of a batch of runs (spec Section 2.6).
"""

import secrets
from pathlib import Path

from poke_env.ps_client import LocalhostServerConfiguration

ROOT = Path(__file__).resolve().parent.parent
TEAMS_DIR = ROOT / "Teams"
RUNS_DIR = ROOT / "runs"
FIGURES_DIR = ROOT / "figures"

# ---------- battle setup (must match src/baselines.py) ----------

BATTLE_FORMAT = "gen9ounotera"  # custom format in pokemon-showdown/config/custom-formats.ts, Tera off
SERVER_CONFIGURATION = LocalhostServerConfiguration
N_TEAM = 4

TEAMS = ("lowvar", "highvar")
OUR_TEAM_FILES = {"lowvar": TEAMS_DIR / "lowvar.txt", "highvar": TEAMS_DIR / "highvar.txt"}
OPP_TEAM_FILES = {"A": TEAMS_DIR / "opp_a.txt", "B": TEAMS_DIR / "opp_b.txt", "C": TEAMS_DIR / "opp_c.txt"}
TEAM_ABBREV = {"lowvar": "lv", "highvar": "hv"}

# ---------- heuristic baseline (spec Section 6.3) ----------

BASELINE_CSV = ROOT / "baselines_summary.csv"
BASELINE_TEAM_KEY = {"lowvar": "low", "highvar": "high"}  # our team name -> team_version value in the CSV
BASELINE_COLUMNS = {"team": "team_version", "pilot": "pilot", "opponent": "opponent_team", "win_rate": "win_rate"}
BASELINE_PILOT = "heuristic"
BASELINE_OPPONENT = "all"  # pooled over opponents A/B/C

# ---------- MDP (spec Section 4) ----------

GAMMA = 0.99  # MUST be the gamma passed to the learner; the shaping imports it from here
MAX_EPISODE_STEPS = 300  # safety truncation in decision steps (truncated=True, not terminated)
ACTION_SPACE_SIZE = 26  # SinglesEnv, gen 9

# ---------- learners ----------

NET_ARCH = [256, 256]  # shared with the DQN arm (spec Section 5.4)
DEVICE = "cpu"  # small MLPs; 6 processes share the machine
TORCH_THREADS = 1  # per process (benchmarked in NOTES.md)
SEEDS = (0, 1, 2)
TOTAL_STEPS = 2_000_000  # decision steps per run (as counted by env.make_env), identical for every run of both algorithms

PPO_KWARGS = dict(
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=128,
    n_epochs=10,
    gamma=GAMMA,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    normalize_advantage=True,
    policy_kwargs=dict(net_arch=dict(pi=NET_ARCH, vf=NET_ARCH)),
)

# ---------- evaluation (spec Section 6) ----------

EVAL_FREQ = 10_000  # decision steps between in-training evaluations
N_EVAL = 120  # battles per evaluation, stratified 40 per opponent team
EVAL_OPP_ORDER = ("A", "B", "C")
FINAL_EVAL_N_HEURISTIC = 1020  # 340 per opponent team
FINAL_EVAL_N_SANITY = 204  # each vs RandomPlayer and MaxBasePowerPlayer, 68 per opponent team

WILSON_Z = 1.959964  # same as src/summarize_baselines.py
THRESHOLD_MARGIN = 0.10  # threshold = b_team + margin
THRESHOLD_CONSECUTIVE = 2
STABILITY_LAST_K = 5
EP_LEN_ROLLING_WINDOW = 200  # battles

# ---------- Showdown accounts ----------

ROLES = ("agent", "opp", "eval", "evopp")
MAX_USERNAME_LEN = 18  # Showdown truncates longer names (server/chat.ts namefilter), which breaks poke-env login
# Random per process launch. Showdown keeps a battle open when its process dies, and re-attaches it to
# the next login with the same name, so a restarted run must never reuse names (NOTES.md item 9).
LAUNCH_TAG = secrets.token_hex(2)


def run_id(algo, team, seed):
    return f"{algo}_{team}_s{seed}"


def account_name(algo, team, seed, role, tag=LAUNCH_TAG):
    """Unique per run, role and launch, e.g. ppo_lv0_agent_3f9a (spec Section 5.2)."""
    assert role in ROLES, role
    name = f"{algo}_{TEAM_ABBREV[team]}{seed}_{role}_{tag}"
    assert len(name) <= MAX_USERNAME_LEN, name
    return name
