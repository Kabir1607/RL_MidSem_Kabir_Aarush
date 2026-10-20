"""DQN-only entrypoint to the existing batch scheduler, without shared file edits.

    python src/DQN/run_batch.py --parallel 3
    python src/DQN/run_batch.py --status

The shared scheduler keeps the clean-code check, ten-job grid, logs and exit codes.
The trainer is registered in memory only; no file in the shared pipeline is changed.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_batch as shared_batch


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("parallel must be a positive integer")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parallel", type=positive_int, default=3)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true", help="development only: bypass shared clean-code check")
    args = parser.parse_args()
    shared_batch.TRAINERS["dqn"] = Path(__file__).with_name("train_dqn.py").resolve()
    if args.status:
        shared_batch.status("dqn")
        return 0
    return shared_batch.launch("dqn", args.parallel, args.allow_dirty)


if __name__ == "__main__":
    sys.exit(main())
