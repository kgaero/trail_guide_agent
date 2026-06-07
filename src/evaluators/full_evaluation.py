"""
Run the full Trail Guide evaluation flow for one experiment.

This script first generates fresh agent responses for the experiment, then
builds the evaluation dataset and runs the cloud evaluator.
"""

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_BATCH_TESTS = REPO_ROOT / "src" / "tests" / "run_batch_tests.py"
BUILD_AND_EVALUATE = REPO_ROOT / "src" / "evaluators" / "build_and_evaluate_dataset.py"


def run_command(args: list[str]) -> None:
    print("\n" + "=" * 80)
    print("Running: " + " ".join(args))
    print("=" * 80)
    subprocess.run(args, cwd=str(REPO_ROOT), check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate fresh agent responses and run cloud evaluation."
    )
    parser.add_argument(
        "experiment",
        help="Experiment name used for experiments/<name>/agent-responses.json.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    run_command([sys.executable, str(RUN_BATCH_TESTS), args.experiment])
    run_command([sys.executable, str(BUILD_AND_EVALUATE), "--experiment", args.experiment])

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except subprocess.CalledProcessError as error:
        print(f"\nERROR: command failed with exit code {error.returncode}", file=sys.stderr)
        sys.exit(error.returncode)
