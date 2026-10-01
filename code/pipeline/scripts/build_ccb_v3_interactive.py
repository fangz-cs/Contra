#!/usr/bin/env python3
"""Project ClarifyCodeBench-v3 into the official interactive-harness schema.

Unlike the recall-only actionable projection, this dataset retains all 419
variants. Tasks with no open-and-observable Gold question remain as K=0 cases,
so the official TKQR and ORA metrics can penalize unnecessary questions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "ClarifyCodeBench-v3-interactive"
EXPECTED_TASKS = 419
EXPECTED_GOLD = 463
EXPECTED_K0_TASKS = 91
EXPECTED_SOURCE_PROBLEMS = 80


class ProjectionError(RuntimeError):
    pass


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.endswith("\n") or not line.strip():
                raise ProjectionError(f"{path}:{line_number}: invalid JSONL line")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ProjectionError(
                    f"{path}:{line_number}: invalid JSON"
                ) from exc
            if not isinstance(row, dict):
                raise ProjectionError(
                    f"{path}:{line_number}: expected object"
                )
            rows.append(row)
    return rows


def actionable(gold: dict[str, Any]) -> bool:
    return (
        gold.get("status") == "open"
        and gold.get("behaviour") == "observable"
    )


def project(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    task_ids: set[str] = set()
    for record in records:
        task_id = record.get("task_id")
        question_id = record.get("question_id")
        statement = record.get("modified_content")
        gold_values = record.get("qa_pairs_v3")
        if (
            not isinstance(task_id, str)
            or not task_id
            or task_id in task_ids
            or not isinstance(question_id, str)
            or not question_id
            or not isinstance(statement, str)
            or not statement
            or not isinstance(gold_values, list)
        ):
            raise ProjectionError(f"invalid V3 record: {task_id!r}")
        task_ids.add(task_id)

        qa_pairs: list[dict[str, Any]] = []
        seen_questions: set[str] = set()
        for gold in gold_values:
            if not isinstance(gold, dict) or not actionable(gold):
                continue
            question = gold.get("question")
            answer = gold.get("answer_v3") or gold.get("answer")
            if (
                not isinstance(question, str)
                or not question.strip()
                or question in seen_questions
                or not isinstance(answer, str)
                or not answer.strip()
            ):
                raise ProjectionError(f"{task_id}: invalid actionable Gold")
            seen_questions.add(question)
            qa_pairs.append(
                {
                    "question": question,
                    "answer": answer,
                    "provenance": gold.get("provenance"),
                }
            )

        output.append(
            {
                "schema_version": SCHEMA_VERSION,
                "task_id": task_id,
                "question_id": question_id,
                "keywords": record.get("keywords"),
                "modified_content": statement,
                "qa_pairs": qa_pairs,
                "k": len(qa_pairs),
            }
        )
    return output


def jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    ).encode("utf-8")


def atomic_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.new-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build(source: Path, output: Path, manifest_path: Path) -> dict[str, Any]:
    source_bytes = source.read_bytes()
    rows = project(read_jsonl(source))
    distribution = Counter(int(row["k"]) for row in rows)
    gold_count = sum(int(row["k"]) for row in rows)
    source_problems = {str(row["question_id"]) for row in rows}
    if (
        len(rows) != EXPECTED_TASKS
        or gold_count != EXPECTED_GOLD
        or distribution[0] != EXPECTED_K0_TASKS
        or len(source_problems) != EXPECTED_SOURCE_PROBLEMS
    ):
        raise ProjectionError(
            "V3 interactive census differs from the frozen expectation"
        )

    output_bytes = jsonl_bytes(rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "built_from": str(source),
        "source_sha256": sha256_bytes(source_bytes),
        "output_sha256": sha256_bytes(output_bytes),
        "tasks": len(rows),
        "tasks_with_actionable_gold": len(rows) - distribution[0],
        "tasks_with_zero_actionable_gold": distribution[0],
        "source_problems": len(source_problems),
        "actionable_gold": gold_count,
        "k_distribution": {
            str(key): distribution[key] for key in sorted(distribution)
        },
        "evaluation_policy": {
            "gold_filter": "status == open and behaviour == observable",
            "retain_zero_gold_tasks": True,
            "intended_harness": "ClarifyCodeBench official interactive protocol",
        },
    }
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    atomic_write(output, output_bytes)
    atomic_write(manifest_path, manifest_bytes)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data_v3/ClarifyCodeBench-v3.jsonl"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data_v3/ClarifyCodeBench-v3-interactive.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data_v3/v3_interactive_manifest.json"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = build(args.source, args.out, args.manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
