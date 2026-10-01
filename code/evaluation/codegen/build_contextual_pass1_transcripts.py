#!/usr/bin/env python3
"""Freeze public dialogue histories for contextual code generation."""
from __future__ import annotations

import argparse
import importlib.util
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from contextual_pass1_common import (
    CODE_NUDGE,
    FOLLOWUP,
    METHOD_ORDER,
    MODEL_ORDER,
    ExperimentError,
    atomic_json,
    atomic_jsonl,
    extract_question,
    index_unique,
    read_json,
    read_jsonl,
    render_question,
    sha256_file,
    stable_hash,
    task_sort_key,
    utc_now,
)


CONFIG_SCHEMA = "contextual-pass1-source-config-v1"
TRANSCRIPT_SCHEMA = "contextual-pass1-transcript-v1"
MANIFEST_SCHEMA = "contextual-pass1-transcript-manifest-v1"
MAX_QA_TURNS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def load_system_prompt(path: Path) -> str:
    spec = importlib.util.spec_from_file_location(
        "contextual_pass1_prompts", path
    )
    if spec is None or spec.loader is None:
        raise ExperimentError(f"cannot import prompt module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    value = getattr(module, "SYSTEM_PROMPT", None)
    if not isinstance(value, str) or not value.strip():
        raise ExperimentError(f"{path}: SYSTEM_PROMPT is missing")
    return value


def standard_pairs(row: Mapping[str, Any]) -> list[tuple[str, str]]:
    messages = row.get("messages")
    responses = row.get("response_contents")
    hit_dict = row.get("hit_dict")
    if (
        not isinstance(messages, list)
        or len(messages) < 2
        or not isinstance(responses, list)
        or not isinstance(hit_dict, dict)
    ):
        raise ExperimentError("interactive source record has invalid history")
    assistant_positions = [
        index
        for index, message in enumerate(messages[2:], 2)
        if isinstance(message, Mapping) and message.get("role") == "assistant"
    ]
    try:
        response_indices = sorted(int(key) for key in hit_dict)
    except (TypeError, ValueError) as exc:
        raise ExperimentError("interactive hit_dict has nonnumeric keys") from exc
    if response_indices != list(range(len(response_indices))):
        raise ExperimentError("interactive question turns are not contiguous")

    pairs: list[tuple[str, str]] = []
    for response_index in response_indices:
        if (
            response_index >= len(responses)
            or response_index >= len(assistant_positions)
        ):
            raise ExperimentError("interactive response index is out of range")
        position = assistant_positions[response_index]
        assistant = messages[position]
        raw_response = responses[response_index]
        if (
            not isinstance(assistant, Mapping)
            or not isinstance(raw_response, str)
            or assistant.get("content") != raw_response
            or position + 1 >= len(messages)
        ):
            raise ExperimentError("interactive response/message alignment drift")
        user = messages[position + 1]
        if not isinstance(user, Mapping) or user.get("role") != "user":
            raise ExperimentError("interactive question has no user reply")
        reply_with_followup = user.get("content")
        if (
            not isinstance(reply_with_followup, str)
            or not reply_with_followup.endswith(FOLLOWUP)
        ):
            raise ExperimentError("interactive reply suffix changed")
        reply = reply_with_followup[: -len(FOLLOWUP)]
        pairs.append((extract_question(raw_response), reply))
    return pairs


def clarifygpt_pairs(row: Mapping[str, Any]) -> list[tuple[str, str]]:
    questions = row.get("questions")
    returned = row.get("qa_returned")
    if not isinstance(questions, list) or not isinstance(returned, list):
        raise ExperimentError("ClarifyGPT record has no public Q/A history")
    pairs: list[tuple[str, str]] = []
    for item in returned:
        if not isinstance(item, Mapping):
            raise ExperimentError("ClarifyGPT Q/A item is not an object")
        question = item.get("q")
        answer = item.get("a")
        if (
            not isinstance(question, str)
            or not question.strip()
            or not isinstance(answer, str)
        ):
            raise ExperimentError("ClarifyGPT Q/A item is malformed")
        pairs.append((question.strip(), answer))
    if questions != [question for question, _ in pairs]:
        raise ExperimentError("ClarifyGPT questions and returned Q/A drift")
    return pairs


def public_messages(
    system_prompt: str,
    requirement: str,
    pairs: Sequence[tuple[str, str]],
) -> list[dict[str, str]]:
    if len(pairs) > MAX_QA_TURNS:
        raise ExperimentError("public history exceeds the five-turn Q/A budget")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": requirement},
    ]
    for question, answer in pairs:
        messages.append(
            {"role": "assistant", "content": render_question(question)}
        )
        messages.append(
            {"role": "user", "content": answer + FOLLOWUP}
        )
    messages.append({"role": "user", "content": CODE_NUDGE})
    return messages


def validate_source_row(
    *,
    task: Mapping[str, Any],
    row: Mapping[str, Any],
    kind: str,
    model: str,
) -> None:
    task_id = task["task_id"]
    question_id = task["question_id"]
    if row.get("task_id") != task_id:
        raise ExperimentError(f"{model}/{task_id}: source task ID drift")
    source_question_id = (
        row.get("question_id") if kind == "clarifygpt" else row.get("id")
    )
    if source_question_id != question_id:
        raise ExperimentError(f"{model}/{task_id}: source question ID drift")
    if kind == "clarifygpt":
        if row.get("status") != "ok" or row.get("model") != model:
            raise ExperimentError(
                f"{model}/{task_id}: invalid ClarifyGPT source cell"
            )
    else:
        if row.get("finished") is not True:
            raise ExperimentError(
                f"{model}/{task_id}: unfinished interactive source"
            )
        if row.get("prompt") != task["modified_content"]:
            raise ExperimentError(
                f"{model}/{task_id}: source requirement drift"
            )


def main() -> int:
    args = parse_args()
    config = read_json(args.config)
    if not isinstance(config, dict) or config.get("schema_version") != CONFIG_SCHEMA:
        raise ExperimentError("invalid source configuration")
    data_path = Path(str(config.get("data")))
    prompt_path = Path(str(config.get("prompt_module")))
    system_prompt = load_system_prompt(prompt_path)
    tasks = read_jsonl(data_path)
    task_index = index_unique(tasks, "task_id", "V3 actionable tasks")
    if len(task_index) != 328:
        raise ExperimentError(f"expected 328 actionable tasks, got {len(tasks)}")
    for task_id, task in task_index.items():
        if (
            task.get("schema_version") != "ClarifyCodeBench-v3-actionable"
            or not isinstance(task.get("modified_content"), str)
            or not isinstance(task.get("question_id"), str)
        ):
            raise ExperimentError(f"{task_id}: invalid actionable task")

    raw_models = config.get("models")
    if not isinstance(raw_models, dict) or tuple(raw_models) != MODEL_ORDER:
        raise ExperimentError("source models are missing or out of frozen order")

    source_manifest: dict[str, Any] = {}
    model_manifest: dict[str, Any] = {}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ordered_tasks = sorted(task_index.values(), key=lambda row: task_sort_key(row["task_id"]))

    for model in MODEL_ORDER:
        source_specs = raw_models[model]
        if not isinstance(source_specs, dict) or tuple(source_specs) != METHOD_ORDER:
            raise ExperimentError(f"{model}: methods are missing or out of order")
        loaded: dict[str, tuple[str, dict[str, dict[str, Any]]]] = {}
        for method in METHOD_ORDER:
            spec = source_specs[method]
            if not isinstance(spec, dict):
                raise ExperimentError(f"{model}/{method}: invalid source spec")
            kind = spec.get("kind")
            path = Path(str(spec.get("path")))
            if kind not in {"interactive", "clarifygpt"}:
                raise ExperimentError(f"{model}/{method}: unknown source kind")
            rows = read_jsonl(path)
            index = index_unique(rows, "task_id", f"{model}/{method} source")
            if not set(task_index).issubset(index):
                missing = sorted(set(task_index) - set(index))
                raise ExperimentError(
                    f"{model}/{method}: missing task IDs {missing[:5]}"
                )
            loaded[method] = (kind, index)
            source_manifest[f"{model}/{method}"] = {
                "kind": kind,
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "rows": len(rows),
            }

        output_rows: list[dict[str, Any]] = []
        question_counts: Counter[str] = Counter()
        source_question_counts: Counter[str] = Counter()
        truncated_task_counts: Counter[str] = Counter()
        task_counts: Counter[str] = Counter()
        for task in ordered_tasks:
            task_id = task["task_id"]
            requirement = task["modified_content"]
            for method in METHOD_ORDER:
                kind, index = loaded[method]
                source = index[task_id]
                validate_source_row(
                    task=task,
                    row=source,
                    kind=kind,
                    model=model,
                )
                source_pairs = (
                    clarifygpt_pairs(source)
                    if kind == "clarifygpt"
                    else standard_pairs(source)
                )
                pairs = source_pairs[:MAX_QA_TURNS]
                messages = public_messages(system_prompt, requirement, pairs)
                row = {
                    "schema_version": TRANSCRIPT_SCHEMA,
                    "model": model,
                    "method": method,
                    "task_id": task_id,
                    "question_id": task["question_id"],
                    "question_count": len(pairs),
                    "source_question_count": len(source_pairs),
                    "history_truncated": len(source_pairs) > MAX_QA_TURNS,
                    "messages": messages,
                    "public_dialogue_sha256": stable_hash(messages[:-1]),
                    "generation_messages_sha256": stable_hash(messages),
                }
                if set(row) & {
                    "qa_pairs",
                    "hit_dict",
                    "hit_questions",
                    "matched_key_questions",
                    "private_test_cases",
                    "public_test_cases",
                }:
                    raise AssertionError("private evaluation fields leaked")
                output_rows.append(row)
                question_counts[method] += len(pairs)
                source_question_counts[method] += len(source_pairs)
                truncated_task_counts[method] += int(
                    len(source_pairs) > MAX_QA_TURNS
                )
                task_counts[method] += 1

        expected_rows = len(task_index) * len(METHOD_ORDER)
        if len(output_rows) != expected_rows:
            raise ExperimentError(f"{model}: transcript census mismatch")
        output_path = args.out_dir / f"{model}.jsonl"
        atomic_jsonl(output_path, output_rows)
        model_manifest[model] = {
            "path": str(output_path.resolve()),
            "sha256": sha256_file(output_path),
            "rows": len(output_rows),
            "tasks_per_method": dict(task_counts),
            "questions_per_method": dict(question_counts),
            "source_questions_per_method": dict(source_question_counts),
            "truncated_tasks_per_method": dict(truncated_task_counts),
        }

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "created_at": utc_now(),
        "protocol": {
            "models": list(MODEL_ORDER),
            "methods": list(METHOD_ORDER),
            "actionable_tasks": len(task_index),
            "maximum_qa_turns": MAX_QA_TURNS,
            "one_generation_per_cell": True,
            "question_selection": "first five in native method order",
            "question_history_truncated_at_five": True,
            "code_nudge": CODE_NUDGE,
            "followup": FOLLOWUP,
            "system_prompt_sha256": stable_hash(system_prompt),
            "generation_context_fields": [
                "system_prompt",
                "requirement",
                "observed_questions",
                "public_user_replies",
                "code_nudge",
            ],
            "excluded_from_generation": [
                "Gold questions",
                "question-match labels",
                "judge decisions",
                "public tests",
                "private tests",
                "expected outputs",
                "previously generated final code",
            ],
        },
        "inputs": {
            "config": {
                "path": str(args.config.resolve()),
                "sha256": sha256_file(args.config),
            },
            "data": {
                "path": str(data_path.resolve()),
                "sha256": sha256_file(data_path),
            },
            "prompt_module": {
                "path": str(prompt_path.resolve()),
                "sha256": sha256_file(prompt_path),
            },
            "dialogue_sources": source_manifest,
        },
        "models": model_manifest,
        "checks": {
            "all_four_models_present": tuple(model_manifest) == MODEL_ORDER,
            "all_cells_have_public_only_context": True,
            "all_models_have_1312_cells": all(
                value["rows"] == 1312 for value in model_manifest.values()
            ),
            "all_methods_have_328_tasks": all(
                set(value["tasks_per_method"].values()) == {328}
                for value in model_manifest.values()
            ),
        },
    }
    if not all(manifest["checks"].values()):
        raise ExperimentError("transcript manifest checks failed")
    manifest["manifest_identity"] = stable_hash(manifest)
    atomic_json(args.out_dir / "MANIFEST.json", manifest)
    print(
        {
            "manifest_identity": manifest["manifest_identity"],
            "models": {
                model: model_manifest[model]["questions_per_method"]
                for model in MODEL_ORDER
            },
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
