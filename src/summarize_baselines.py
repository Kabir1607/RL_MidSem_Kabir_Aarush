"""Summarise the T8 baseline battles into per-cell tables.

Reads Baselines_Log - Battles.csv (one row per battle) and writes baselines_summary.csv
with one row per (team_version, pilot) cell pooled over opponents (opponent_team = "all")
and one row per (team_version, pilot, opponent_team) block.

Note: the scripted pilots play with revealed-only information, not full team sheets,
so RL lift against these numbers includes an information advantage (plan Section 6).
"""

import argparse
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
Z = 1.959964  # 95%


def wilson(wins, n, z=Z):
    if n == 0:
        return float("nan"), float("nan")
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def summarise(g):
    n, wins = len(g), int(g.won.sum())
    lo, hi = wilson(wins, n)
    return pd.Series({
        "n": n,
        "wins": wins,
        "win_rate": wins / n,
        "wilson_low": lo,
        "wilson_high": hi,
        "mean_turns": g.turns.mean(),
        "std_turns": g.turns.std(),
        "mean_our_remaining": g.our_remaining.mean(),
        "mean_opp_remaining": g.opp_remaining.mean(),
        "swap_moves_used": int(g.swap_moves_used.sum()),
        "swap_miss_rate": g.swap_moves_missed.sum() / max(g.swap_moves_used.sum(), 1),
        "all_moves_used": int(g.all_moves_used.sum()),
        "all_miss_rate": g.all_moves_missed.sum() / max(g.all_moves_used.sum(), 1),
    })


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inp", type=Path, default=ROOT / "Baselines_Log - Battles.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "baselines_summary.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    order = {"team_version": ["low", "high"], "pilot": ["random", "maxbp", "heuristic"]}
    for col, cats in order.items():
        df[col] = pd.Categorical(df[col], cats, ordered=True)

    pooled = df.groupby(["team_version", "pilot"], observed=True).apply(summarise, include_groups=False).reset_index()
    pooled.insert(2, "opponent_team", "all")
    per_opp = (df.groupby(["team_version", "pilot", "opponent_team"], observed=True)
               .apply(summarise, include_groups=False).reset_index())
    for t in (pooled, per_opp):
        t[["n", "wins", "swap_moves_used", "all_moves_used"]] = t[["n", "wins", "swap_moves_used", "all_moves_used"]].astype(int)
    summary = pd.concat([pooled, per_opp], ignore_index=True)
    summary.to_csv(args.out, index=False, float_format="%.4f")

    fmt = {c: "{:.3f}".format for c in ["win_rate", "wilson_low", "wilson_high", "swap_miss_rate", "all_miss_rate"]}
    fmt.update({c: "{:.2f}".format for c in ["mean_turns", "std_turns", "mean_our_remaining", "mean_opp_remaining"]})
    cols = ["team_version", "pilot", "opponent_team", "n", "win_rate", "wilson_low", "wilson_high",
            "mean_turns", "mean_our_remaining", "mean_opp_remaining", "swap_miss_rate", "all_miss_rate"]
    print("Pooled over opponents (one cell = version x pilot):")
    print(pooled[cols].to_string(index=False, formatters=fmt))
    print("\nPer opponent block:")
    print(per_opp[cols].to_string(index=False, formatters=fmt))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
