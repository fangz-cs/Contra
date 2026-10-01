#!/usr/bin/env python3
"""Evaluate contextual generations with the official LiveCodeBench executor."""
from __future__ import annotations

import argparse
import base64
import json
import pickle
import subprocess
import sys
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from contextual_pass1_common import (
    METHOD_ORDER,
    ExperimentError,
    atomic_json,
    atomic_jsonl,
    model_slug,
    read_json,
    read_jsonl,
    sha256_file,
    stable_hash,
    task_sort_key,
    utc_now,
)


PER_TASK_SCHEMA = "contextual-livecodebench-pass1-task-v1"
METHOD_SCHEMA = "contextual-livecodebench-pass1-method-v1"
MODEL_SCHEMA = "contextual-livecodebench-pass1-model-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--livecodebench-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=6)
    return parser.parse_args()


def lcb_commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def sanitize_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {"error_code": None, "error_message": "unparseable metadata"}
    if not isinstance(value, Mapping):
        return {"error_code": None, "error_message": None}
    return {
        "error_code": value.get("error_code"),
        "error_message": value.get("error_message"),
        "error_type": (
            str(value.get("error", "")).split("(", 1)[0][:120] or None
        ),
    }


def passed_all(test_results: Any) -> bool:
    if not isinstance(test_results, list) or not test_results:
        return False
    return all(value is True for value in test_results)


def failure_category(
    *,
    passed: bool,
    code: str,
    metadata: Mapping[str, Any],
) -> str:
    if passed:
        return "passed"
    if not code.strip():
        return "empty_generation"
    error_code = metadata.get("error_code")
    if error_code == -2:
        return "wrong_answer"
    if error_code == -3:
        return "time_limit_exceeded"
    if error_code == -4:
        return "runtime_or_compile_error"
    if error_code == -5:
        return "evaluator_error"
    return "failed_tests"


def evaluation_sample(source: Mapping[str, Any]) -> dict[str, str]:
    """Mirror CodeGenerationProblem.get_evaluation_sample without HF imports."""
    public_tests = json.loads(str(source["public_test_cases"]))
    raw_private = str(source["private_test_cases"])
    try:
        private_tests = json.loads(raw_private)
    except json.JSONDecodeError:
        private_tests = json.loads(
            pickle.loads(
                zlib.decompress(base64.b64decode(raw_private.encode("utf-8")))
            )
        )
    raw_metadata = source["metadata"]
    metadata = (
        json.loads(raw_metadata)
        if isinstance(raw_metadata, str)
        else raw_metadata
    )
    if (
        not isinstance(public_tests, list)
        or not isinstance(private_tests, list)
        or not isinstance(metadata, Mapping)
    ):
        raise ExperimentError("invalid LiveCodeBench source encoding")
    tests = public_tests + private_tests
    if any(
        not isinstance(test, Mapping)
        or not isinstance(test.get("input"), str)
        or not isinstance(test.get("output"), str)
        for test in tests
    ):
        raise ExperimentError("malformed LiveCodeBench test case")
    return {
        "input_output": json.dumps(
            {
                "inputs": [test["input"] for test in tests],
                "outputs": [test["output"] for test in tests],
                "fn_name": metadata.get("func_name"),
            }
        )
    }


def main() -> int:
    args = parse_args()
    if args.workers < 1 or args.timeout < 1:
        raise ExperimentError("evaluation workers and timeout must be positive")
    lcb_path = str(args.livecodebench_root.resolve())
    if lcb_path not in sys.path:
        sys.path.insert(0, lcb_path)
    from lcb_runner.evaluation.compute_code_generation_metrics import (  # noqa: PLC0415
        codegen_metrics,
    )

    generation_root = args.generation_dir / model_slug(args.model)
    generation_summary_path = generation_root / "SUMMARY.json"
    generation_summary = read_json(generation_summary_path)
    if (
        not isinstance(generation_summary, dict)
        or generation_summary.get("schema_version")
        != "contextual-pass1-generation-summary-v1"
        or generation_summary.get("model") != args.model
    ):
        raise ExperimentError("invalid generation summary")
    results_path = Path(str(generation_summary["results"]["path"]))
    if sha256_file(results_path) != generation_summary["results"]["sha256"]:
        raise ExperimentError("generation result hash changed")
    generations = read_jsonl(results_path)
    if len(generations) != 1312:
        raise ExperimentError("expected 1312 generation rows")

    source_rows = read_jsonl(args.sources)
    source_by_qid = {
        row["question_id"]: row
        for row in source_rows
        if isinstance(row.get("question_id"), str)
    }
    if len(source_by_qid) != len(source_rows):
        raise ExperimentError("LiveCodeBench source IDs are missing or duplicate")
    problems = {
        question_id: evaluation_sample(source)
        for question_id, source in source_by_qid.items()
    }

    model_root = args.out_dir / model_slug(args.model)
    model_root.mkdir(parents=True, exist_ok=True)
    method_summaries: dict[str, dict[str, Any]] = {}
    for method in METHOD_ORDER:
        rows = sorted(
            [row for row in generations if row.get("method") == method],
            key=lambda row: task_sort_key(str(row["task_id"])),
        )
        if len(rows) != 328 or len({row["task_id"] for row in rows}) != 328:
            raise ExperimentError(f"{args.model}/{method}: generation census drift")
        missing_qids = sorted(
            {
                str(row["question_id"])
                for row in rows
                if str(row["question_id"]) not in problems
            }
        )
        if missing_qids:
            raise ExperimentError(
                f"{args.model}/{method}: tests missing for {missing_qids[:5]}"
            )

        method_root = model_root / method
        summary_path = method_root / "SUMMARY.json"
        input_identity = stable_hash(
            {
                "model": args.model,
                "method": method,
                "generation_results_sha256": sha256_file(results_path),
                "source_sha256": sha256_file(args.sources),
                "lcb_commit": lcb_commit(args.livecodebench_root),
                "timeout": args.timeout,
            }
        )
        if summary_path.exists():
            prior = read_json(summary_path)
            if (
                isinstance(prior, dict)
                and prior.get("schema_version") == METHOD_SCHEMA
                and prior.get("input_identity") == input_identity
            ):
                method_summaries[method] = prior
                print(f"{args.model}/{method}: reuse completed evaluation")
                continue
            raise ExperimentError(f"incompatible evaluation state: {summary_path}")

        samples = [
            problems[str(row["question_id"])]
            for row in rows
        ]
        code_lists = [[str(row.get("code") or "")] for row in rows]
        metrics, raw_results, raw_metadata = codegen_metrics(
            samples,
            code_lists,
            k_list=[1],
            num_process_evaluate=args.workers,
            timeout=args.timeout,
            debug=False,
        )
        per_task: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            generation_results = raw_results[index]
            if (
                not isinstance(generation_results, list)
                or len(generation_results) != 1
            ):
                raise ExperimentError("LiveCodeBench result shape changed")
            tests = generation_results[0]
            normalized_tests = [
                bool(value) if type(value).__module__ == "numpy" else value
                for value in tests
            ]
            passed = passed_all(normalized_tests)
            metadata_items = raw_metadata[index]
            metadata = sanitize_metadata(
                metadata_items[0]
                if isinstance(metadata_items, list) and metadata_items
                else None
            )
            category = failure_category(
                passed=passed,
                code=str(row.get("code") or ""),
                metadata=metadata,
            )
            per_task.append(
                {
                    "schema_version": PER_TASK_SCHEMA,
                    "model": args.model,
                    "method": method,
                    "task_id": row["task_id"],
                    "question_id": row["question_id"],
                    "question_count": row["question_count"],
                    "generation_request_sha256": row["request_sha256"],
                    "parse_status": row["parse_status"],
                    "passed": passed,
                    "outcome": category,
                    "tests_executed": len(normalized_tests),
                    "error_code": metadata.get("error_code"),
                    "error_message": metadata.get("error_message"),
                    "error_type": metadata.get("error_type"),
                }
            )

        solved = sum(row["passed"] for row in per_task)
        outcome_counts = dict(Counter(row["outcome"] for row in per_task))
        per_task_path = method_root / "per_task.jsonl"
        atomic_jsonl(per_task_path, per_task)
        official_pass_at_1 = metrics.get("pass@1")
        if official_pass_at_1 is None:
            official_pass_at_1 = metrics.get("pass@1", solved / len(rows))
        if abs(float(official_pass_at_1) - solved / len(rows)) > 1e-12:
            raise ExperimentError(
                f"{args.model}/{method}: official pass@1 disagrees with census"
            )
        summary = {
            "schema_version": METHOD_SCHEMA,
            "completed_at": utc_now(),
            "input_identity": input_identity,
            "model": args.model,
            "method": method,
            "tasks": len(rows),
            "solved": solved,
            "pass_at_1": solved / len(rows),
            "official_metrics": metrics,
            "outcomes": outcome_counts,
            "execution": {
                "includes_public_tests": True,
                "includes_private_tests": True,
                "timeout_seconds_per_test": args.timeout,
                "workers": args.workers,
                "livecodebench_root": lcb_path,
                "livecodebench_commit": lcb_commit(args.livecodebench_root),
            },
            "inputs": {
                "generation_results": {
                    "path": str(results_path.resolve()),
                    "sha256": sha256_file(results_path),
                },
                "sources": {
                    "path": str(args.sources.resolve()),
                    "sha256": sha256_file(args.sources),
                },
            },
            "per_task": {
                "path": str(per_task_path.resolve()),
                "sha256": sha256_file(per_task_path),
            },
        }
        summary["summary_identity"] = stable_hash(summary)
        atomic_json(summary_path, summary)
        method_summaries[method] = summary
        print(
            f"{args.model}/{method}: {solved}/328 "
            f"pass@1={100 * solved / 328:.2f}%",
            flush=True,
        )

    model_summary = {
        "schema_version": MODEL_SCHEMA,
        "completed_at": utc_now(),
        "model": args.model,
        "methods": {
            method: {
                "solved": method_summaries[method]["solved"],
                "tasks": method_summaries[method]["tasks"],
                "pass_at_1": method_summaries[method]["pass_at_1"],
                "outcomes": method_summaries[method]["outcomes"],
                "summary_identity": method_summaries[method][
                    "summary_identity"
                ],
            }
            for method in METHOD_ORDER
        },
        "generation_summary": {
            "path": str(generation_summary_path.resolve()),
            "sha256": sha256_file(generation_summary_path),
        },
    }
    model_summary["summary_identity"] = stable_hash(model_summary)
    atomic_json(model_root / "SUMMARY.json", model_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
