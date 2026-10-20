"""Watch one battle live: a trained checkpoint vs the heuristic opponent, slowed down to human speed.

In the browser: open http://localhost:8000 (it redirects to Showdown's own client, connected to your local
server), pick any guest name, then type  /join <battle tag>  in the chat box. The tag is printed below.
The terminal shows the same battle: each decision with the agent's probabilities over its legal actions
and its value estimate, then a play-by-play. Swapped moves (the LowVar/HighVar manipulation) are marked *.
An HTML replay is saved to runs/replays/ when the battle ends.

  python src/watch_battle.py                                   # newest checkpoint of ppo_lowvar_s0
  python src/watch_battle.py --run ppo_highvar_s0 --opp C --delay 4
  python src/watch_battle.py --ckpt runs/ppo_lowvar_s0/ckpt/step_100000.zip --no-wait --delay 0

Needs the Showdown server. Safe to run while a training batch is going (own accounts, one battle).
"""

import argparse
import re
import time
from pathlib import Path

import numpy as np
import torch
from poke_env.battle import Move
from sb3_contrib import MaskablePPO

import config
import teamsheet
from check_teams import diff_versions
from env import OPPONENTS, make_env
from teams import StratifiedOpponentTeambuilder

_, _swaps = diff_versions(teamsheet.OUR_SHEETS["lowvar"], teamsheet.OUR_SHEETS["highvar"])
SWAP_MOVES = {m for _, _, low, high in _swaps for m in (low, high)}
RUN_RE = re.compile(r"^(?P<algo>[a-z]+)_(?P<team>lowvar|highvar)_s(?P<seed>\d+)")
REPLAY_DIR = config.RUNS_DIR / "replays"


def step_of(path):
    return int(re.search(r"step_(\d+)", path.name).group(1))


def load_model(run, ckpt):
    """The given checkpoint, or the newest loadable one of the run (the newest may still be being written)."""
    paths = [Path(ckpt)] if ckpt else sorted((config.RUNS_DIR / run / "ckpt").glob("step_*.zip"), key=step_of)[::-1][:2]
    if not paths:
        raise SystemExit(f"no checkpoints in runs/{run}/ckpt/")
    for i, path in enumerate(paths):
        try:
            return MaskablePPO.load(path, device="cpu"), path
        except Exception:
            if i == len(paths) - 1:
                raise


def move_name(move_id):
    move = Move(move_id, gen=9)
    star = "*" if move_id in SWAP_MOVES else ""
    acc = "never misses" if move.entry["accuracy"] is True else f"{move.entry['accuracy']}%"
    return f"{move.entry['name']}{star} ({move.base_power} BP, {acc})" if move.base_power else f"{move.entry['name']}{star}"


def action_label(battle, action):
    if action < 6:
        return f"switch to {list(battle.team.values())[action].species}"
    return move_name(list(battle.active_pokemon.moves.values())[action - 6].id)


def hp(mon):
    return "fainted" if mon is None or mon.fainted else f"{mon.current_hp_fraction:.0%}"


class Narrator:
    """Prints the battle log lines poke-env has received since the last call."""

    def __init__(self, battle):
        self.battle = battle
        self.seen = 0

    def who(self, ident):
        side = "You" if ident.startswith(self.battle.player_role) else "Opp"
        return f"{side} {ident.split(': ', 1)[-1]}"

    def flush(self):
        lines = self.battle._replay_data[self.seen:]
        self.seen += len(lines)
        for ev in lines:
            kind = ev[1] if len(ev) > 1 else ""
            arg = ev[2] if len(ev) > 2 else ""
            text = None
            if kind == "turn":
                text = f"\n──────── Turn {arg} ────────"
            elif kind in ("switch", "drag"):
                text = f"  {self.who(arg).split()[0]} sent out {ev[3].split(',')[0]} ({ev[4].split()[0]} HP)"
            elif kind == "move":
                ours = arg.startswith(self.battle.player_role)
                star = " *" if ours and re.sub(r"[^a-z0-9]", "", ev[3].lower()) in SWAP_MOVES else ""
                text = f"  {self.who(arg)} used {ev[3]}{star}"
            elif kind == "-miss":
                text = "    ...but it MISSED"
            elif kind == "-crit":
                text = "    critical hit!"
            elif kind == "-supereffective":
                text = "    super effective"
            elif kind == "-resisted":
                text = "    not very effective"
            elif kind == "-immune":
                text = f"    {self.who(arg)} is immune"
            elif kind in ("-damage", "-heal"):
                source = f"  [{ev[4].replace('[from] ', '')}]" if len(ev) > 4 and ev[4].startswith("[from]") else ""
                text = f"    {self.who(arg)} → {ev[3].split()[0].replace('0 fnt', '0')} HP{source}"
            elif kind == "faint":
                text = f"  ✖ {self.who(arg)} fainted"
            elif kind == "-status":
                text = f"    {self.who(arg)} is now {ev[3]}"
            elif kind in ("-boost", "-unboost"):
                sign = "+" if kind == "-boost" else "-"
                text = f"    {self.who(arg)} {ev[3]} {sign}{ev[4]}"
            elif kind == "-sidestart":
                text = f"    {ev[3].replace('move: ', '')} on {'your' if arg.startswith(self.battle.player_role) else 'opponent'} side"
            elif kind == "-enditem":
                text = f"    {self.who(arg)} lost its {ev[3]}"
            elif kind == "win":
                text = f"\n🏆 {arg} wins"
            if text:
                print(text, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="ppo_lowvar_s0", help="run whose newest checkpoint to use")
    ap.add_argument("--ckpt", help="a specific checkpoint .zip (its run directory gives the team)")
    ap.add_argument("--opp", choices=config.EVAL_OPP_ORDER, help="opponent team (default: random)")
    ap.add_argument("--opponent", choices=list(OPPONENTS), default="heuristic")
    ap.add_argument("--delay", type=float, default=3.0, help="seconds to pause before each of our decisions")
    ap.add_argument("--no-wait", action="store_true", help="start immediately instead of waiting for Enter")
    ap.add_argument("--stochastic", action="store_true", help="sample actions instead of the greedy policy")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run = Path(args.ckpt).resolve().parent.parent.name if args.ckpt else args.run
    m = RUN_RE.match(run)
    if not m:
        raise SystemExit(f"cannot tell the team from run name {run!r}")
    team = m["team"]
    model, ckpt = load_model(args.run, args.ckpt)
    print(f"checkpoint: {ckpt.relative_to(config.ROOT) if ckpt.is_absolute() else ckpt} ({team}, step {step_of(ckpt):,})")

    env = make_env(team, args.seed, role="eval", opponent=args.opponent, algo="wch", monitor=False)
    poke = env.env.env
    if args.opp:
        env.opp_teams = StratifiedOpponentTeambuilder(1, order=(args.opp,))
        poke.agent2.update_team(env.opp_teams)
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    poke.agent1._save_replays = str(REPLAY_DIR)

    obs, _ = env.reset()
    battle = env.battle
    opp_key = teamsheet.identify_opponent(battle)
    print(f"\nbattle: {battle.battle_tag}   (you: {team}, opponent: {args.opponent} with team {opp_key})")
    print("  browser: open http://localhost:8000 , choose a name, then type in chat:")
    print(f"      /join {battle.battle_tag}")
    if not args.no_wait:
        input("\nPress Enter to start the battle... ")

    narrator = Narrator(battle)
    narrator.flush()
    done, rng = False, np.random.default_rng(args.seed)
    while not done:
        mask = env.action_masks()
        obs_t, _ = model.policy.obs_to_tensor(obs)
        with torch.no_grad():
            probs = model.policy.get_distribution(obs_t, action_masks=mask[None]).distribution.probs[0].numpy()
            value = model.policy.predict_values(obs_t)[0, 0].item()
        action = int(rng.choice(len(probs), p=probs)) if args.stochastic else int(np.argmax(probs))
        ours, theirs = battle.active_pokemon, battle.opponent_active_pokemon
        print(f"\n  [decision] You: {ours.species if ours else '-'} {hp(ours)}  vs  "
              f"Opp: {theirs.species if theirs else '-'} {hp(theirs)}   value estimate {value:+.2f}")
        options = sorted(np.flatnonzero(mask), key=lambda a: -probs[a])
        print("    options: " + " | ".join(f"{action_label(battle, a)} {probs[a]:.0%}" for a in options))
        print(f"    → {action_label(battle, action)}", flush=True)
        time.sleep(args.delay)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        narrator.flush()

    end = info["battle_end"]
    result = {1: "WON", -1: "LOST", 0: "TIE"}[end["outcome"]]
    print(f"\n{result} in {end['len_turns']} turns ({end['len_steps']} decisions)")
    time.sleep(1)  # let poke-env write the replay
    replay = REPLAY_DIR / f"{poke.agent1.username} - {battle.battle_tag}.html"
    if replay.exists():
        print(f"replay: {replay.relative_to(config.ROOT)}  (open it in a browser)")
    env.close()


if __name__ == "__main__":
    main()
