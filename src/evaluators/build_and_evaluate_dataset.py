"""
Build a fresh evaluation dataset from batch-test responses and run evaluation.

This script joins:
  - experiments/{experiment}/agent-responses.json
  - data/ground_truth_evaluation_dataset.jsonl

It writes data/trail_guide_evaluation_dataset.jsonl with the original query and
ground_truth plus the fresh agent response, then invokes evaluate_agent.py with
a unique DATASET_VERSION so Foundry does not reuse a stale uploaded dataset.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
GROUND_TRUTH_FILE = REPO_ROOT / "data" / "ground_truth_evaluation_dataset.jsonl"
EVALUATION_DATASET_FILE = REPO_ROOT / "data" / "trail_guide_evaluation_dataset.jsonl"
EVALUATOR_SCRIPT = REPO_ROOT / "src" / "evaluators" / "evaluate_agent.py"


def fail(message: str) -> None:
    raise ValueError(message)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as error:
                fail(f"{path} line {line_number} is not valid JSON: {error}")
            if not isinstance(row, dict):
                fail(f"{path} line {line_number} must be a JSON object.")
            rows.append(row)
    return rows


def require_text(row: dict, field: str, source: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        fail(f"{source} is missing a non-empty '{field}' field.")
    return value


def resolve_experiment_file(experiment: str) -> Path:
    experiments_root = EXPERIMENTS_DIR.resolve()
    experiment_dir = (experiments_root / experiment).resolve()

    try:
        experiment_dir.relative_to(experiments_root)
    except ValueError:
        fail(f"Experiment path escapes {experiments_root}: {experiment}")

    responses_file = experiment_dir / "agent-responses.json"
    if not responses_file.exists():
        fail(f"agent-responses.json not found at {responses_file}")

    return responses_file


def load_ground_truth(path: Path) -> list[dict]:
    if not path.exists():
        fail(f"Ground-truth dataset not found at {path}")

    rows = read_jsonl(path)
    if not rows:
        fail(f"Ground-truth dataset is empty: {path}")

    seen_queries: set[str] = set()
    for index, row in enumerate(rows, start=1):
        query = require_text(row, "query", f"{path} row {index}")
        require_text(row, "ground_truth", f"{path} row {index}")
        if query in seen_queries:
            fail(f"Duplicate query in ground-truth dataset: {query}")
        seen_queries.add(query)

    return rows


def load_batch_responses(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as file:
        try:
            payload = json.load(file)
        except json.JSONDecodeError as error:
            fail(f"{path} is not valid JSON: {error}")

    test_results = payload.get("test_results") if isinstance(payload, dict) else None
    if not isinstance(test_results, list):
        fail(f"{path} must contain a 'test_results' list.")

    responses_by_prompt: dict[str, str] = {}
    for index, result in enumerate(test_results, start=1):
        if not isinstance(result, dict):
            fail(f"{path} test_results[{index}] must be a JSON object.")
        prompt = require_text(result, "prompt", f"{path} test_results[{index}]")
        response = require_text(result, "response", f"{path} test_results[{index}]")
        if prompt in responses_by_prompt:
            fail(f"Duplicate prompt in batch responses: {prompt}")
        responses_by_prompt[prompt] = response

    if not responses_by_prompt:
        fail(f"{path} contains no batch responses.")

    return responses_by_prompt


def build_evaluation_rows(ground_truth_rows: list[dict], responses_by_prompt: dict[str, str]) -> list[dict]:
    ground_truth_queries = {row["query"] for row in ground_truth_rows}
    response_queries = set(responses_by_prompt)

    missing_responses = sorted(ground_truth_queries - response_queries)
    if missing_responses:
        fail(
            "Fresh responses are missing for ground-truth queries:\n"
            + "\n".join(f"  - {query}" for query in missing_responses)
        )

    extra_responses = sorted(response_queries - ground_truth_queries)
    if extra_responses:
        fail(
            "Fresh responses do not map to ground-truth queries:\n"
            + "\n".join(f"  - {query}" for query in extra_responses)
        )

    return [
        {
            "query": row["query"],
            "response": responses_by_prompt[row["query"]],
            "ground_truth": row["ground_truth"],
        }
        for row in ground_truth_rows
    ]


def validate_output_rows(rows: list[dict], expected_count: int) -> None:
    if len(rows) != expected_count:
        fail(f"Generated {len(rows)} rows, expected {expected_count}.")

    for index, row in enumerate(rows, start=1):
        require_text(row, "query", f"generated row {index}")
        require_text(row, "response", f"generated row {index}")
        require_text(row, "ground_truth", f"generated row {index}")


def write_jsonl_atomically(path: Path, rows: list[dict]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def dataset_version_for(experiment: str) -> str:
    safe_experiment = re.sub(r"[^A-Za-z0-9_.-]+", "-", experiment).strip("-")
    if not safe_experiment:
        safe_experiment = "experiment"
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{safe_experiment}-{timestamp}"


def run_evaluator(dataset_version: str) -> int:
    env = os.environ.copy()
    env["DATASET_VERSION"] = dataset_version
    completed = subprocess.run(
        [sys.executable, str(EVALUATOR_SCRIPT)],
        cwd=str(REPO_ROOT),
        env=env,
        check=False,
    )
    return completed.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a fresh Trail Guide evaluation dataset and run cloud evaluation."
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment folder name under experiments/ containing agent-responses.json.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    responses_file = resolve_experiment_file(args.experiment)
    ground_truth_rows = load_ground_truth(GROUND_TRUTH_FILE)
    responses_by_prompt = load_batch_responses(responses_file)
    evaluation_rows = build_evaluation_rows(ground_truth_rows, responses_by_prompt)

    validate_output_rows(evaluation_rows, expected_count=len(ground_truth_rows))
    write_jsonl_atomically(EVALUATION_DATASET_FILE, evaluation_rows)

    version = dataset_version_for(args.experiment)
    print(f"Generated {EVALUATION_DATASET_FILE} with {len(evaluation_rows)} rows.")
    print(f"Running evaluator with DATASET_VERSION={version}")

    return run_evaluator(version)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
