"""Clarification-only adapter for the ClarifyCodeBench-v3 harness.

RITE supplies a frozen, Gold-blind ordering of clarification questions. This
module asks those questions through ClarifyCodeBench's official matcher and
records a logical terminal turn, but never invokes a code-generation model.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


PLAN_SCHEMA = "ccb-v3-pregenerated-question-plan-v1"
RUN_SCHEMA = "ccb-v3-clarification-only-run-v1"
FORBIDDEN_PLAN_KEYS = {
    "answer",
    "answers",
    "gold",
    "gold_answer",
    "gold_answers",
    "qa_pairs",
}


class HarnessError(RuntimeError):
    pass


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.endswith("\n") or not line.strip():
                raise HarnessError(f"{path}:{line_number}: invalid JSONL line")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise HarnessError(
                    f"{path}:{line_number}: invalid JSON"
                ) from exc
            if not isinstance(row, dict):
                raise HarnessError(
                    f"{path}:{line_number}: expected object"
                )
            rows.append(row)
    return rows


def index_tasks(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = row.get("task_id")
        if (
            not isinstance(task_id, str)
            or not task_id
            or task_id in output
        ):
            raise HarnessError(f"invalid or duplicate task_id: {task_id!r}")
        output[task_id] = dict(row)
    return output


def _scan_forbidden_keys(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        overlap = FORBIDDEN_PLAN_KEYS & set(value)
        if overlap:
            raise HarnessError(
                f"Gold-bearing plan keys at {path}: {sorted(overlap)}"
            )
        for key, child in value.items():
            _scan_forbidden_keys(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _scan_forbidden_keys(child, f"{path}[{index}]")


def load_plan(
    path: Path,
    *,
    dataset_path: Path,
    tasks: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnessError(f"invalid question plan: {path}") from exc
    if not isinstance(plan, dict) or plan.get("schema_version") != PLAN_SCHEMA:
        raise HarnessError(f"{path}: unexpected question-plan schema")
    _scan_forbidden_keys(plan)

    expected_dataset_hash = sha256_bytes(dataset_path.read_bytes())
    if plan.get("benchmark_sha256") != expected_dataset_hash:
        raise HarnessError(f"{path}: benchmark hash mismatch")
    entries = plan.get("tasks")
    if not isinstance(entries, dict) or set(entries) != set(tasks):
        missing = sorted(set(tasks) - set(entries or {}))
        extra = sorted(set(entries or {}) - set(tasks))
        raise HarnessError(
            "question-plan task coverage mismatch: "
            f"missing={missing[:5]} extra={extra[:5]}"
        )

    total_questions = 0
    for task_id, task in tasks.items():
        entry = entries[task_id]
        if not isinstance(entry, dict):
            raise HarnessError(f"{task_id}: plan entry is not an object")
        if (
            entry.get("question_id") != task.get("question_id")
            or entry.get("statement_sha256")
            != sha256_text(str(task.get("modified_content") or ""))
        ):
            raise HarnessError(f"{task_id}: plan/data binding mismatch")
        questions = entry.get("questions")
        if not isinstance(questions, list):
            raise HarnessError(f"{task_id}: questions is not a list")
        seen: set[str] = set()
        for position, question in enumerate(questions):
            if not isinstance(question, dict):
                raise HarnessError(
                    f"{task_id}: question {position} is not an object"
                )
            candidate_id = question.get("candidate_id")
            text = question.get("text")
            if (
                not isinstance(candidate_id, str)
                or not candidate_id
                or candidate_id in seen
                or not isinstance(text, str)
                or not text.strip()
                or question.get("queue_position") != position
            ):
                raise HarnessError(
                    f"{task_id}: invalid question at position {position}"
                )
            seen.add(candidate_id)
        total_questions += len(questions)
    if plan.get("question_count") != total_questions:
        raise HarnessError(f"{path}: question-count mismatch")
    return plan


def run_question_task(
    *,
    interact: Any,
    client: Any,
    task: Mapping[str, Any],
    plan_entry: Mapping[str, Any],
    judge_model: str,
    max_turns: int = 6,
    consistency: int = 3,
    max_retries: int = 10,
) -> dict[str, Any]:
    """Replay one frozen question queue through the official matcher."""
    if max_turns < 1:
        raise HarnessError("max_turns must be positive")
    qa_pairs = list(task.get("qa_pairs") or [])
    statement = str(task["modified_content"])
    queue = list(plan_entry["questions"])
    question_budget = max_turns - 1
    scheduled = queue[:question_budget]

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": interact.SYSTEM_PROMPT},
        {"role": "user", "content": statement},
    ]
    response_contents: list[str] = []
    hit_questions: set[str] = set()
    hit_dict: dict[str, int] = {}
    asked_questions: list[dict[str, Any]] = []
    failure: str | None = None

    for turn, question in enumerate(scheduled):
        text = str(question["text"]).strip()
        content = f"[QUESTION]\n{text}\n[/QUESTION]"
        messages.append({"role": "assistant", "content": content})
        response_contents.append(content)
        hit, reply, matched = interact._judge_answer(
            client,
            text,
            qa_pairs,
            hit_questions,
            judge_model,
            consistency,
            max_retries,
        )
        if hit == -1:
            failure = "official clarification matcher failed"
            break
        hit_dict[str(turn)] = hit
        if matched is not None:
            hit_questions.add(matched)
        messages.append(
            {"role": "user", "content": reply + interact.FOLLOWUP}
        )
        asked_questions.append(
            {
                "turn": turn,
                "candidate_id": question["candidate_id"],
                "queue_position": question["queue_position"],
                "question": text,
            }
        )

    finished = failure is None
    logical_terminal = finished
    terminal_reason = (
        "question_budget_exhausted"
        if len(queue) > question_budget
        else "question_queue_exhausted"
    )
    question_turns = len(response_contents)
    return {
        "schema_version": RUN_SCHEMA,
        "id": task.get("question_id"),
        "task_id": task.get("task_id"),
        "key_question_num": len(qa_pairs),
        "prompt": statement,
        "messages": messages,
        "response_contents": response_contents,
        "hit_dict": hit_dict,
        "hit_questions": sorted(hit_questions),
        "question_turns": question_turns,
        "turns": question_turns + (1 if logical_terminal else 0),
        "final_code": "",
        "finished": finished,
        "failure": failure,
        "evaluation_scope": "clarification_only",
        "code_generation_evaluated": False,
        "logical_terminal_turn": logical_terminal,
        "terminal_reason": terminal_reason if logical_terminal else None,
        "n_questions_available": len(queue),
        "n_questions_scheduled": len(scheduled),
        "n_questions_asked": len(asked_questions),
        "n_questions_omitted_by_turn_budget": max(
            0, len(queue) - question_budget
        ),
        "asked_questions": asked_questions,
    }
