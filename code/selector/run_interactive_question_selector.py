#!/usr/bin/env python3
"""Run a dialogue-conditioned clarification policy inside ClarifyCodeBench.

At each round, the policy either selects one immutable candidate ID from the
remaining frozen queue or terminates clarification. Selected question text is
always replayed verbatim from the queue.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


METHOD = "CONTRA+Dialogue-Conditioned-Question-Selector"
SCHEMA_MANIFEST = "contra-interactive-question-selector-manifest-v3"
SCHEMA_RESULT = "contra-interactive-question-selector-result-v3"
SCHEMA_EVENT = "contra-interactive-question-selector-event-v2"
SCHEMA_COMPLETE = "contra-interactive-question-selector-complete-v3"
PLAN_SCHEMA = "clarifycodebench-pregenerated-question-plan-v1"
SELECT_TOOL_NAME = "select_question"
FINISH_TOOL_NAME = "finish_clarification"
HARNESS_QUESTION_ROUND_LIMIT = 5
CODE_NUDGE = (
    "Clarification is complete. Emit the final program in the required "
    "[CODE] format."
)

SELECTOR_SYSTEM_PROMPT = """You will receive a system prompt, a requirement,
the conversation history, and a list of potentially relevant clarification
questions.

Update the question set using the requirement and conversation history. A
question is resolved when its answer is already determined or when prior
answers make it inapplicable. Select the most useful unresolved question. If
no question remains unresolved, finish."""


class SelectorError(RuntimeError):
    """A frozen input, selector response, or protocol invariant is invalid."""


class SelectorResponseError(SelectorError):
    """The selector did not return one allowed candidate ID."""


def _json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def stable_hash(value: object) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SelectorError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise SelectorError(f"{path}: expected an object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            raise SelectorError(f"{path}:{line_number}: blank line")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SelectorError(
                f"{path}:{line_number}: invalid JSON"
            ) from exc
        if not isinstance(value, dict):
            raise SelectorError(f"{path}:{line_number}: expected object")
        rows.append(value)
    return rows


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.new-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.new-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def validate_frozen_plan(
    *,
    plan: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]],
    benchmark_sha256: str,
) -> dict[str, tuple[dict[str, Any], ...]]:
    """Validate and normalize immutable per-task candidate queues."""
    task_by_id: dict[str, Mapping[str, Any]] = {}
    for task in tasks:
        task_id = task.get("task_id")
        if (
            not isinstance(task_id, str)
            or not task_id
            or task_id in task_by_id
        ):
            raise SelectorError("benchmark has missing or duplicate task_id")
        task_by_id[task_id] = task
    plan_tasks = plan.get("tasks")
    if (
        plan.get("schema_version") != PLAN_SCHEMA
        or plan.get("benchmark_sha256") != benchmark_sha256
        or plan.get("task_count") != len(task_by_id)
        or not isinstance(plan_tasks, Mapping)
        or set(plan_tasks) != set(task_by_id)
    ):
        raise SelectorError("question plan does not match the benchmark")

    queues: dict[str, tuple[dict[str, Any], ...]] = {}
    all_candidate_ids: set[str] = set()
    for task_id in sorted(task_by_id):
        task = task_by_id[task_id]
        task_plan = plan_tasks[task_id]
        if not isinstance(task_plan, Mapping):
            raise SelectorError(f"{task_id}: invalid plan task")
        statement = task.get("modified_content")
        if not isinstance(statement, str) or not statement:
            raise SelectorError(f"{task_id}: missing task statement")
        expected_statement_hash = hashlib.sha256(
            statement.encode("utf-8")
        ).hexdigest()
        if (
            task_plan.get("question_id") != task.get("question_id")
            or task_plan.get("statement_sha256")
            != expected_statement_hash
        ):
            raise SelectorError(f"{task_id}: task lineage mismatch")
        raw_questions = task_plan.get("questions")
        if not isinstance(raw_questions, list):
            raise SelectorError(f"{task_id}: questions are not a list")

        queue: list[dict[str, Any]] = []
        task_candidate_ids: set[str] = set()
        task_question_texts: set[str] = set()
        for position, raw in enumerate(raw_questions):
            if not isinstance(raw, Mapping):
                raise SelectorError(
                    f"{task_id}: invalid question at position {position}"
                )
            candidate_id = raw.get("candidate_id")
            text = raw.get("text")
            queue_position = raw.get("queue_position", position)
            if (
                not isinstance(candidate_id, str)
                or not candidate_id
                or candidate_id in task_candidate_ids
                or candidate_id in all_candidate_ids
                or not isinstance(text, str)
                or not text.strip()
                or text != text.strip()
                or text in task_question_texts
                or queue_position != position
                or "[QUESTION]" in text
                or "[/QUESTION]" in text
            ):
                raise SelectorError(
                    f"{task_id}: invalid immutable question at position "
                    f"{position}"
                )
            task_candidate_ids.add(candidate_id)
            all_candidate_ids.add(candidate_id)
            task_question_texts.add(text)
            queue.append(
                {
                    "candidate_id": candidate_id,
                    "text": text,
                    "original_queue_position": position,
                }
            )
        queues[task_id] = tuple(queue)
    return queues


def build_selector_messages(
    *,
    task_system_prompt: str,
    conversation: Sequence[Mapping[str, Any]],
    remaining_questions: Sequence[Mapping[str, Any]],
    round_index: int,
    selector_system_prompt: str = SELECTOR_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    """Build a Gold-blind selector request from public dialogue only."""
    if not 1 <= round_index <= HARNESS_QUESTION_ROUND_LIMIT:
        raise SelectorError("selector round is outside the harness limit")
    if (
        len(conversation) < 2
        or conversation[0].get("role") != "system"
        or conversation[0].get("content") != task_system_prompt
        or conversation[1].get("role") != "user"
        or not isinstance(conversation[1].get("content"), str)
    ):
        raise SelectorError("dialogue history has an invalid task context")
    if not remaining_questions:
        raise SelectorError("selector cannot run with an empty queue")
    if not selector_system_prompt.strip():
        raise SelectorError("selector system prompt cannot be empty")

    normalized_questions: list[dict[str, Any]] = []
    candidate_ids: set[str] = set()
    for question in remaining_questions:
        candidate_id = question.get("candidate_id")
        text = question.get("text")
        position = question.get("original_queue_position")
        if (
            not isinstance(candidate_id, str)
            or not candidate_id
            or candidate_id in candidate_ids
            or not isinstance(text, str)
            or not text
            or not isinstance(position, int)
            or position < 0
        ):
            raise SelectorError("remaining queue is malformed")
        candidate_ids.add(candidate_id)
        normalized_questions.append(
            {
                "candidate_id": candidate_id,
                "question": text,
            }
        )

    public_history: list[dict[str, str]] = []
    for message in conversation[2:]:
        role = message.get("role")
        content = message.get("content")
        if (
            role not in {"user", "assistant"}
            or not isinstance(content, str)
        ):
            raise SelectorError("conversation contains a non-public message")
        public_history.append({"role": str(role), "content": content})

    payload = {
        "system_prompt": task_system_prompt,
        "requirement": str(conversation[1]["content"]),
        "conversation_history": public_history,
        "remaining_questions": normalized_questions,
    }
    return [
        {"role": "system", "content": selector_system_prompt},
        {
            "role": "user",
            "content": json.dumps(
                payload, ensure_ascii=False, sort_keys=True
            ),
        },
    ]


def build_selector_tool(
    remaining_questions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    candidate_ids = [
        str(question["candidate_id"]) for question in remaining_questions
    ]
    if not candidate_ids or len(set(candidate_ids)) != len(candidate_ids):
        raise SelectorError("selector tool requires unique candidate IDs")
    return {
        "type": "function",
        "function": {
            "name": SELECT_TOOL_NAME,
            "description": (
                "Select the most useful question that remains unresolved."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["candidate_id"],
                "properties": {
                    "candidate_id": {
                        "type": "string",
                        "enum": candidate_ids,
                    }
                },
            },
        },
    }


def build_finish_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": FINISH_TOOL_NAME,
            "description": "Finish when no question remains unresolved.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {},
            },
        },
    }


def format_selector_tools(
    tools: Sequence[Mapping[str, Any]],
    *,
    schema: str,
) -> list[dict[str, Any]]:
    if schema == "chat":
        return [dict(tool) for tool in tools]
    if schema != "responses":
        raise SelectorError(f"unsupported selector tool schema: {schema}")
    formatted: list[dict[str, Any]] = []
    for tool in tools:
        function = tool.get("function")
        if tool.get("type") != "function" or not isinstance(
            function, Mapping
        ):
            raise SelectorError(
                "Responses tool conversion requires Chat function tools"
            )
        formatted.append(
            {
                "type": "function",
                "name": function["name"],
                "description": function.get("description", ""),
                "parameters": function.get("parameters")
                or {"type": "object"},
            }
        )
    return formatted


def parse_selection_response(
    response: Any,
    *,
    allowed_candidate_ids: Sequence[str],
) -> dict[str, str | None]:
    """Parse one forced question-selection or clarification-finish action."""
    choices = getattr(response, "choices", None) or []
    if len(choices) != 1:
        raise SelectorResponseError(
            f"expected one response choice, got {len(choices)}"
        )
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        raise SelectorResponseError(
            "selector returned free text beside the forced tool call"
        )
    calls = getattr(message, "tool_calls", None) or []
    if len(calls) != 1:
        raise SelectorResponseError(f"expected one tool call, got {len(calls)}")
    function = getattr(calls[0], "function", None)
    function_name = getattr(function, "name", None)
    if function_name not in {SELECT_TOOL_NAME, FINISH_TOOL_NAME}:
        raise SelectorResponseError("selector called an unknown tool")
    arguments = getattr(function, "arguments", None)
    if not isinstance(arguments, str):
        raise SelectorResponseError("tool arguments are not JSON text")
    try:
        value = json.loads(arguments)
    except json.JSONDecodeError as exc:
        raise SelectorResponseError("tool arguments are invalid JSON") from exc
    if not isinstance(value, dict):
        raise SelectorResponseError("tool arguments must be an object")
    if function_name == FINISH_TOOL_NAME:
        if value:
            raise SelectorResponseError(
                "finish_clarification does not accept arguments"
            )
        return {"action": "finish", "candidate_id": None}
    if set(value) != {"candidate_id"}:
        raise SelectorResponseError(
            "selector output must contain only candidate_id"
        )
    candidate_id = value["candidate_id"]
    allowed = set(allowed_candidate_ids)
    if not isinstance(candidate_id, str) or candidate_id not in allowed:
        raise SelectorResponseError(
            "selector chose a candidate outside the remaining queue"
        )
    return {"action": "ask", "candidate_id": candidate_id}


def _response_record(response: Any) -> dict[str, Any]:
    choices = getattr(response, "choices", None) or []
    choice = choices[0] if choices else None
    message = getattr(choice, "message", None)
    raw = (
        response.model_dump()
        if callable(getattr(response, "model_dump", None))
        else {}
    )
    return {
        "response_id": getattr(response, "id", None),
        "response_model": getattr(response, "model", None),
        "finish_reason": getattr(choice, "finish_reason", None),
        "content": getattr(message, "content", None),
        "tool_calls": [
            {
                "name": getattr(call.function, "name", None),
                "arguments": getattr(call.function, "arguments", None),
            }
            for call in (getattr(message, "tool_calls", None) or [])
        ],
        "usage": raw.get("usage") or {},
        "route_meta": raw.get("_route_meta") or {},
    }


def _is_permanent(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and 400 <= status < 500 and status != 429


class DialogueQuestionSelector:
    """LLM policy for selecting a remaining question or finishing."""

    def __init__(
        self,
        client: Any,
        *,
        model: str,
        max_tokens: int,
        reasoning_effort: str,
        max_retries: int,
        tool_schema: str = "chat",
        temperature: float | None = 0.0,
        system_prompt: str = SELECTOR_SYSTEM_PROMPT,
        followup_system_prompt: str | None = None,
    ):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.max_retries = max_retries
        self.tool_schema = tool_schema
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.followup_system_prompt = followup_system_prompt

    def select(
        self,
        *,
        task_system_prompt: str,
        conversation: Sequence[Mapping[str, Any]],
        remaining_questions: Sequence[Mapping[str, Any]],
        round_index: int,
    ) -> dict[str, Any]:
        selector_system_prompt = (
            self.followup_system_prompt
            if round_index > 1 and self.followup_system_prompt is not None
            else self.system_prompt
        )
        messages = build_selector_messages(
            task_system_prompt=task_system_prompt,
            conversation=conversation,
            remaining_questions=remaining_questions,
            round_index=round_index,
            selector_system_prompt=selector_system_prompt,
        )
        tools = format_selector_tools(
            [
                build_selector_tool(remaining_questions),
                build_finish_tool(),
            ],
            schema=self.tool_schema,
        )
        request_kwargs = {
            "max_tokens": self.max_tokens,
            "reasoning_effort": self.reasoning_effort,
            "stream": False,
            "tools": tools,
            "tool_choice": "required",
        }
        if self.temperature is not None:
            request_kwargs["temperature"] = self.temperature
        attempts: list[dict[str, Any]] = []
        response: Any | None = None
        for attempt in range(self.max_retries):
            started = time.monotonic()
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **request_kwargs,
                )
                break
            except Exception as exc:  # noqa: BLE001
                permanent = _is_permanent(exc)
                attempts.append(
                    {
                        "attempt": attempt + 1,
                        "latency_seconds": round(
                            time.monotonic() - started, 6
                        ),
                        "error": str(exc)[:4000],
                        "permanent": permanent,
                    }
                )
                if permanent or attempt + 1 >= self.max_retries:
                    raise SelectorError(
                        f"selector transport failed after "
                        f"{attempt + 1} attempt(s): {exc}"
                    ) from exc
                time.sleep(min(2**attempt, 30))
        if response is None:
            raise SelectorError("selector transport returned no response")
        allowed = [
            str(question["candidate_id"])
            for question in remaining_questions
        ]
        decision = parse_selection_response(
            response,
            allowed_candidate_ids=allowed,
        )
        request_identity = {
            "messages": messages,
            "tools": tools,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "reasoning_effort": self.reasoning_effort,
        }
        return {
            **decision,
            "trace": {
                "round": round_index,
                "request_sha256": stable_hash(request_identity),
                "remaining_candidate_ids": allowed,
                "attempts": attempts,
                "response": _response_record(response),
            },
        }


def generate_final_content(
    client: Any,
    *,
    model: str,
    messages: Sequence[Mapping[str, Any]],
    max_tokens: int,
    reasoning_effort: str,
    max_retries: int,
    temperature: float | None,
) -> tuple[str, dict[str, Any]]:
    """Generate only the terminal code response after selection has stopped."""
    request_kwargs = {
        "max_tokens": max_tokens,
        "reasoning_effort": reasoning_effort,
        "stream": False,
    }
    if temperature is not None:
        request_kwargs["temperature"] = temperature
    attempts: list[dict[str, Any]] = []
    for attempt in range(max_retries):
        started = time.monotonic()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=list(messages),
                **request_kwargs,
            )
            choices = getattr(response, "choices", None) or []
            if len(choices) != 1:
                raise SelectorError(
                    f"final code call returned {len(choices)} choices"
                )
            content = getattr(choices[0].message, "content", None)
            if not isinstance(content, str) or not content.strip():
                raise SelectorError("final code call returned empty content")
            return content, {
                "request_sha256": stable_hash(
                    {
                        "messages": list(messages),
                        "model": model,
                        **request_kwargs,
                    }
                ),
                "attempts": attempts,
                "response": _response_record(response),
            }
        except SelectorError:
            raise
        except Exception as exc:  # noqa: BLE001
            permanent = _is_permanent(exc)
            attempts.append(
                {
                    "attempt": attempt + 1,
                    "latency_seconds": round(
                        time.monotonic() - started, 6
                    ),
                    "error": str(exc)[:4000],
                    "permanent": permanent,
                }
            )
            if permanent or attempt + 1 >= max_retries:
                raise SelectorError(
                    f"final code transport failed after "
                    f"{attempt + 1} attempt(s): {exc}"
                ) from exc
            time.sleep(min(2**attempt, 30))
    raise AssertionError("unreachable")


def index_resume_events(
    events: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[int, Mapping[str, Any]],
    dict[int, Mapping[str, Any]],
    Mapping[str, Any] | None,
]:
    selections: dict[int, Mapping[str, Any]] = {}
    answers: dict[int, Mapping[str, Any]] = {}
    final: Mapping[str, Any] | None = None
    for event in events:
        event_type = event.get("event")
        if event_type in {"selection", "answer"}:
            round_index = event.get("round")
            target = selections if event_type == "selection" else answers
            if (
                not isinstance(round_index, int)
                or not 1 <= round_index <= HARNESS_QUESTION_ROUND_LIMIT
                or round_index in target
            ):
                raise SelectorError("resume journal has an invalid round event")
            target[round_index] = event
        elif event_type == "final_generation":
            if final is not None:
                raise SelectorError(
                    "resume journal has duplicate final generation"
                )
            final = event
        else:
            raise SelectorError("resume journal has an unknown event")
    if any(round_index not in selections for round_index in answers):
        raise SelectorError("resume journal has an answer without selection")
    selection_rounds = sorted(selections)
    if selection_rounds != list(range(1, len(selection_rounds) + 1)):
        raise SelectorError("resume selection rounds are not contiguous")
    for round_index in selection_rounds[:-1]:
        if (
            selections[round_index].get("action") != "ask"
            or round_index not in answers
        ):
            raise SelectorError(
                "resume journal advances past a terminal or unanswered action"
            )
    for round_index, selection in selections.items():
        action = selection.get("action")
        candidate_id = selection.get("candidate_id")
        if (
            action == "ask"
            and (not isinstance(candidate_id, str) or not candidate_id)
        ):
            raise SelectorError("resume journal has an invalid ask action")
        if action == "finish" and candidate_id is not None:
            raise SelectorError("resume journal has an invalid finish action")
        if action not in {"ask", "finish"}:
            raise SelectorError("resume journal has an unknown selector action")
        if action == "finish" and round_index in answers:
            raise SelectorError("resume journal answers a finish action")
    if final is not None and selection_rounds:
        last_round = selection_rounds[-1]
        if (
            selections[last_round].get("action") == "ask"
            and last_round not in answers
        ):
            raise SelectorError(
                "resume journal generated code before answering a question"
            )
    return selections, answers, final


def exact_code_block(
    content: str,
    *,
    extract_code: Callable[[str], str],
) -> str | None:
    """Accept one nonempty ClarifyCodeBench code block with no surrounding prose."""
    stripped = content.strip()
    opening = "[CODE]"
    closing = "[/CODE]"
    if (
        not stripped.startswith(opening)
        or not stripped.endswith(closing)
        or stripped.count(opening) != 1
        or stripped.count(closing) != 1
    ):
        return None
    before_close = stripped[: -len(closing)]
    inner = before_close[len(opening) :].strip()
    if not inner or extract_code(stripped).strip() != inner:
        return None
    return inner


def run_task(
    *,
    harness: tuple[Any, Any, Any, Any, Any],
    task: Mapping[str, Any],
    frozen_queue: Sequence[Mapping[str, Any]],
    selector: Any,
    generate_final: Callable[
        [Sequence[Mapping[str, Any]]], tuple[str, dict[str, Any]]
    ],
    judge_client: Any,
    judge_model: str,
    consistency: int,
    max_retries: int,
    resume_events: Sequence[Mapping[str, Any]] = (),
    event_sink: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Execute one queue under a dialogue-conditioned termination policy."""
    interact, prompts, extract_code, extract_question, is_code = harness
    qa_pairs = list(task.get("qa_pairs") or [])
    statement = task.get("modified_content")
    if not isinstance(statement, str) or not statement:
        raise SelectorError("task has no statement")
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": prompts.SYSTEM_PROMPT},
        {"role": "user", "content": statement},
    ]
    remaining = [dict(question) for question in frozen_queue]
    response_contents: list[str] = []
    hit_questions: set[str] = set()
    hit_dict: dict[str, int] = {}
    asked_questions: list[dict[str, Any]] = []
    selector_decisions: list[dict[str, Any]] = []
    selector_finished = False
    failure_reason: str | None = None
    selection_events, answer_events, final_event = index_resume_events(
        resume_events
    )
    used_events: set[tuple[str, int]] = set()

    round_index = 1
    while remaining and round_index <= HARNESS_QUESTION_ROUND_LIMIT:
        conversation_sha256 = stable_hash(messages)
        remaining_ids = [
            str(question["candidate_id"]) for question in remaining
        ]
        saved_selection = selection_events.get(round_index)
        if saved_selection is not None:
            if (
                saved_selection.get("conversation_sha256")
                != conversation_sha256
                or saved_selection.get("remaining_candidate_ids")
                != remaining_ids
                or not isinstance(saved_selection.get("selector_trace"), Mapping)
            ):
                raise SelectorError(
                    f"round {round_index}: saved selection context changed"
                )
            selection = {
                "action": saved_selection.get("action"),
                "candidate_id": saved_selection.get("candidate_id"),
                "trace": dict(saved_selection["selector_trace"]),
            }
            used_events.add(("selection", round_index))
        else:
            try:
                selection = selector.select(
                    task_system_prompt=prompts.SYSTEM_PROMPT,
                    conversation=messages,
                    remaining_questions=remaining,
                    round_index=round_index,
                )
            except Exception as exc:  # noqa: BLE001
                failure_reason = (
                    f"selector_failed:{type(exc).__name__}:"
                    f"{str(exc)[:1000]}"
                )
                break
            selection_event = {
                "event": "selection",
                "round": round_index,
                "conversation_sha256": conversation_sha256,
                "remaining_candidate_ids": remaining_ids,
                "action": selection.get("action"),
                "candidate_id": selection.get("candidate_id"),
                "selector_trace": dict(selection.get("trace") or {}),
            }
            if event_sink is not None:
                event_sink(selection_event)
        action = selection.get("action")
        candidate_id = selection.get("candidate_id")
        selector_decisions.append(
            {
                "round": round_index,
                "action": action,
                "candidate_id": candidate_id,
                "trace": dict(selection.get("trace") or {}),
            }
        )
        if action == "finish" and candidate_id is None:
            selector_finished = True
            break
        if action == "finish":
            failure_reason = "selector_finish_returned_candidate"
            break
        if action != "ask":
            failure_reason = "selector_returned_invalid_action"
            break
        selected_index = next(
            (
                index
                for index, question in enumerate(remaining)
                if question["candidate_id"] == candidate_id
            ),
            None,
        )
        if selected_index is None:
            failure_reason = "selector_returned_nonremaining_candidate"
            break
        question = remaining.pop(selected_index)
        content = f"[QUESTION]\n{question['text']}\n[/QUESTION]"
        turn = len(response_contents)
        messages.append({"role": "assistant", "content": content})
        response_contents.append(content)
        asked_questions.append(
            {
                "turn": turn,
                "round": round_index,
                "candidate_id": question["candidate_id"],
                "original_queue_position": question[
                    "original_queue_position"
                ],
                "question": question["text"],
                "render": "frozen_queue_verbatim",
            }
        )

        model_question = extract_question(content) or content
        question_sha256 = hashlib.sha256(
            model_question.encode("utf-8")
        ).hexdigest()
        saved_answer = answer_events.get(round_index)
        if saved_answer is not None:
            if (
                saved_answer.get("candidate_id") != candidate_id
                or saved_answer.get("question_sha256") != question_sha256
                or type(saved_answer.get("hit")) is not int
                or not isinstance(saved_answer.get("reply"), str)
                or (
                    saved_answer.get("matched") is not None
                    and not isinstance(saved_answer.get("matched"), str)
                )
            ):
                raise SelectorError(
                    f"round {round_index}: saved judge answer changed"
                )
            hit = int(saved_answer["hit"])
            reply = str(saved_answer["reply"])
            matched = saved_answer.get("matched")
            used_events.add(("answer", round_index))
        else:
            hit, reply, matched = interact._judge_answer(
                judge_client,
                model_question,
                qa_pairs,
                hit_questions,
                judge_model,
                consistency,
                max_retries,
            )
            if hit != -1 and event_sink is not None:
                event_sink(
                    {
                        "event": "answer",
                        "round": round_index,
                        "candidate_id": candidate_id,
                        "question_sha256": question_sha256,
                        "hit": hit,
                        "reply": reply,
                        "matched": matched,
                    }
                )
        if hit == -1:
            failure_reason = "clarifycodebench_question_judge_failed"
            break
        hit_dict[str(turn)] = hit
        if matched is not None:
            hit_questions.add(matched)
        messages.append(
            {"role": "user", "content": reply + interact.FOLLOWUP}
        )
        round_index += 1

    final_trace: dict[str, Any] | None = None
    final_wrapper_added = False
    final_response_status: str | None = None
    if failure_reason is None:
        messages.append({"role": "user", "content": CODE_NUDGE})
        final_conversation_sha256 = stable_hash(messages)
        if final_event is not None:
            if (
                final_event.get("conversation_sha256")
                != final_conversation_sha256
                or not isinstance(final_event.get("raw_content"), str)
                or not isinstance(final_event.get("generation_trace"), Mapping)
            ):
                raise SelectorError("saved final generation context changed")
            raw_content = str(final_event["raw_content"])
            final_trace = dict(final_event["generation_trace"])
            used_events.add(("final_generation", 0))
        else:
            try:
                raw_content, final_trace = generate_final(messages)
            except Exception as exc:  # noqa: BLE001
                failure_reason = (
                    f"final_code_failed:{type(exc).__name__}:"
                    f"{str(exc)[:1000]}"
                )
            else:
                if event_sink is not None:
                    event_sink(
                        {
                            "event": "final_generation",
                            "conversation_sha256": (
                                final_conversation_sha256
                            ),
                            "raw_content": raw_content,
                            "generation_trace": final_trace,
                        }
                    )
        if failure_reason is None:
            code = exact_code_block(
                raw_content,
                extract_code=extract_code,
            )
            if code is None or not is_code(raw_content):
                final_response_status = "no_exact_code_block"
                final_content = raw_content
            else:
                final_response_status = "exact_code_block"
                final_content = f"[CODE]\n{code}\n[/CODE]"
            messages.append(
                {"role": "assistant", "content": final_content}
            )
            response_contents.append(final_content)

    expected_event_keys = {
        ("selection", round_index) for round_index in selection_events
    } | {
        ("answer", round_index) for round_index in answer_events
    }
    if final_event is not None:
        expected_event_keys.add(("final_generation", 0))
    if failure_reason is None and used_events != expected_event_keys:
        raise SelectorError("resume journal contains unused future events")

    final_content = response_contents[-1] if response_contents else ""
    final_code = (
        extract_code(final_content)
        if final_response_status == "exact_code_block"
        else ""
    )
    stop_reason = (
        "failure"
        if failure_reason is not None
        else "selector_finish"
        if selector_finished
        else "queue_exhausted"
        if not remaining
        else "harness_round_limit"
    )
    return {
        "schema_version": SCHEMA_RESULT,
        "method": METHOD,
        "id": task.get("question_id"),
        "task_id": task.get("task_id"),
        "key_question_num": len(qa_pairs),
        "prompt": statement,
        "messages": messages,
        "response_contents": response_contents,
        "hit_dict": hit_dict,
        "hit_questions": sorted(hit_questions),
        "turns": len(response_contents),
        "final_code": final_code,
        "finished": failure_reason is None,
        "failure_reason": failure_reason,
        "selector_model": getattr(selector, "model", None),
        "selector_stop_action_available": True,
        "selector_question_generation_available": False,
        "selector_stop_reason": stop_reason,
        "harness_question_round_limit": HARNESS_QUESTION_ROUND_LIMIT,
        "initial_queue_size": len(frozen_queue),
        "asked_questions": asked_questions,
        "remaining_candidate_ids": [
            question["candidate_id"] for question in remaining
        ],
        "selector_decisions": selector_decisions,
        "final_generation_trace": final_trace,
        "final_wrapper_added": final_wrapper_added,
        "final_response_status": final_response_status,
    }


def _bootstrap(release: Path) -> tuple[Any, Any, Any, Any, Any]:
    if not release.is_dir():
        raise SelectorError(f"ClarifyCodeBench release missing: {release}")
    sys.path.insert(0, str(release))
    from clarifycodebench import interact, prompts  # noqa: PLC0415
    from clarifycodebench.extract import (  # noqa: PLC0415
        extract_code,
        extract_question,
        is_code,
    )

    return interact, prompts, extract_code, extract_question, is_code


def _load_gateway_client(
    *,
    transport_path: Path,
    base_url: str,
) -> Any:
    spec = importlib.util.spec_from_file_location(
        "interactive_selector_transport", transport_path
    )
    if spec is None or spec.loader is None:
        raise SelectorError(f"cannot import transport: {transport_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    gateway = getattr(module, "GatewayClient", None)
    if gateway is None:
        raise SelectorError("transport does not expose GatewayClient")
    api_key = (
        os.environ.get("CONTRA_API_KEY")
        or os.environ.get("CLARIFY_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    ).strip()
    if not api_key:
        raise SelectorError(
            "set CONTRA_API_KEY, CLARIFY_API_KEY, or OPENAI_API_KEY"
        )
    return gateway(api_key=api_key, base_url=base_url, timeout=900.0)


def _resume_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    rows: list[dict[str, Any]] = []
    valid_bytes = 0
    for index, raw in enumerate(lines):
        try:
            value = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            torn_tail = index == len(lines) - 1 and not raw.endswith(b"\n")
            if not torn_tail:
                raise SelectorError(
                    f"{path}:{index + 1}: invalid JSONL"
                ) from exc
            with path.open("r+b") as handle:
                handle.truncate(valid_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            break
        if not isinstance(value, dict):
            raise SelectorError(f"{path}:{index + 1}: expected object")
        rows.append(value)
        valid_bytes += len(raw)
        if index == len(lines) - 1 and not raw.endswith(b"\n"):
            with path.open("ab") as handle:
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
    return rows


class DurableWriter:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def append(self, row: Mapping[str, Any]) -> None:
        payload = json.dumps(
            row, ensure_ascii=False, sort_keys=True
        ) + "\n"
        with self.lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())


class TaskEventJournal:
    """Append-only per-task checkpoint journal for exact round resumption."""

    def __init__(
        self,
        path: Path,
        *,
        manifest_identity: str,
        task_id: str,
    ):
        self.path = path
        self.manifest_identity = manifest_identity
        self.task_id = task_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows = _resume_rows(path)
        for sequence, row in enumerate(self.rows):
            if (
                row.get("schema_version") != SCHEMA_EVENT
                or row.get("manifest_identity") != manifest_identity
                or row.get("task_id") != task_id
                or row.get("sequence") != sequence
            ):
                raise SelectorError(
                    f"{task_id}: event journal identity changed"
                )
        index_resume_events(self.rows)

    def append(self, event: Mapping[str, Any]) -> None:
        row = {
            "schema_version": SCHEMA_EVENT,
            "manifest_identity": self.manifest_identity,
            "task_id": self.task_id,
            "sequence": len(self.rows),
            **dict(event),
        }
        payload = json.dumps(
            row, ensure_ascii=False, sort_keys=True
        ) + "\n"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        self.rows.append(row)


def questions_are_frozen_verbatim(
    *,
    row: Mapping[str, Any],
    frozen_queues: Mapping[str, Sequence[Mapping[str, Any]]],
) -> bool:
    task_id = row.get("task_id")
    if not isinstance(task_id, str) or task_id not in frozen_queues:
        return False
    queue = {
        str(question["candidate_id"]): question
        for question in frozen_queues[task_id]
    }
    asked = row.get("asked_questions")
    if not isinstance(asked, list):
        return False
    seen: set[str] = set()
    for question in asked:
        if not isinstance(question, Mapping):
            return False
        candidate_id = question.get("candidate_id")
        source = queue.get(str(candidate_id))
        if (
            source is None
            or candidate_id in seen
            or question.get("question") != source.get("text")
            or question.get("original_queue_position")
            != source.get("original_queue_position")
            or question.get("render") != "frozen_queue_verbatim"
        ):
            return False
        seen.add(str(candidate_id))
    return True


def selector_decisions_are_valid(
    *,
    row: Mapping[str, Any],
    frozen_queues: Mapping[str, Sequence[Mapping[str, Any]]],
) -> bool:
    task_id = row.get("task_id")
    if not isinstance(task_id, str) or task_id not in frozen_queues:
        return False
    remaining_ids: list[str] = []
    for question in frozen_queues[task_id]:
        if not isinstance(question, Mapping):
            return False
        candidate_id = question.get("candidate_id")
        if (
            not isinstance(candidate_id, str)
            or not candidate_id
            or candidate_id in remaining_ids
        ):
            return False
        remaining_ids.append(candidate_id)
    decisions = row.get("selector_decisions")
    asked = row.get("asked_questions")
    if (
        row.get("selector_stop_action_available") is not True
        or row.get("selector_question_generation_available") is not False
        or row.get("harness_question_round_limit")
        != HARNESS_QUESTION_ROUND_LIMIT
        or row.get("initial_queue_size") != len(remaining_ids)
        or not isinstance(decisions, list)
        or not isinstance(asked, list)
        or len(decisions) > HARNESS_QUESTION_ROUND_LIMIT
        or len(asked) > HARNESS_QUESTION_ROUND_LIMIT
    ):
        return False
    asked_index = 0
    finished = False
    for expected_round, decision in enumerate(decisions, 1):
        if (
            not isinstance(decision, Mapping)
            or decision.get("round") != expected_round
            or finished
        ):
            return False
        action = decision.get("action")
        candidate_id = decision.get("candidate_id")
        trace = decision.get("trace")
        if not isinstance(trace, Mapping):
            return False
        response = trace.get("response")
        if not isinstance(response, Mapping):
            return False
        content = response.get("content")
        tool_calls = response.get("tool_calls")
        if (
            (isinstance(content, str) and content.strip())
            or not isinstance(tool_calls, list)
            or len(tool_calls) != 1
        ):
            return False
        tool_call = tool_calls[0]
        if not isinstance(tool_call, Mapping):
            return False
        arguments = tool_calls[0].get("arguments")
        try:
            parsed_arguments = json.loads(arguments)
        except (TypeError, json.JSONDecodeError):
            return False
        if action == "ask":
            asked_question = (
                asked[asked_index] if asked_index < len(asked) else None
            )
            if (
                not isinstance(asked_question, Mapping)
                or candidate_id != asked_question.get("candidate_id")
                or candidate_id not in remaining_ids
                or tool_call.get("name") != SELECT_TOOL_NAME
                or not isinstance(parsed_arguments, dict)
                or set(parsed_arguments) != {"candidate_id"}
                or parsed_arguments["candidate_id"] != candidate_id
            ):
                return False
            remaining_ids.remove(str(candidate_id))
            asked_index += 1
        elif action == "finish":
            if (
                candidate_id is not None
                or not remaining_ids
                or tool_call.get("name") != FINISH_TOOL_NAME
                or parsed_arguments != {}
            ):
                return False
            finished = True
        else:
            return False
    if asked_index != len(asked):
        return False
    if row.get("remaining_candidate_ids") != remaining_ids:
        return False
    stop_reason = row.get("selector_stop_reason")
    if stop_reason == "selector_finish":
        return finished and len(asked) < HARNESS_QUESTION_ROUND_LIMIT
    if stop_reason == "queue_exhausted":
        return not finished and not remaining_ids
    if stop_reason == "harness_round_limit":
        return (
            not finished
            and len(asked) == HARNESS_QUESTION_ROUND_LIMIT
            and bool(remaining_ids)
        )
    return False


def write_completion_if_full(
    *,
    output_path: Path,
    manifest_path: Path,
    manifest_identity: str,
    expected_task_ids: Sequence[str],
    frozen_queues: Mapping[str, Sequence[Mapping[str, Any]]],
) -> bool:
    """Write or restore the completion receipt without making model calls."""
    final_rows = _resume_rows(output_path)
    final_ids = [str(row.get("task_id")) for row in final_rows]
    if (
        len(final_ids) != len(expected_task_ids)
        or len(final_ids) != len(set(final_ids))
        or set(final_ids) != set(expected_task_ids)
    ):
        return False
    by_task_id = {
        str(row["task_id"]): row for row in final_rows
    }
    canonical_rows = [
        by_task_id[task_id] for task_id in sorted(expected_task_ids)
    ]
    canonical_payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in canonical_rows
    ).encode("utf-8")
    if output_path.read_bytes() != canonical_payload:
        atomic_jsonl(output_path, canonical_rows)
    final_rows = canonical_rows
    checks = {
        "task_coverage_exact": True,
        "all_tasks_finished": all(
            row.get("finished") is True
            and row.get("manifest_identity") == manifest_identity
            for row in final_rows
        ),
        "question_texts_from_frozen_queue_only": all(
            questions_are_frozen_verbatim(
                row=row,
                frozen_queues=frozen_queues,
            )
            for row in final_rows
        ),
        "selector_decision_protocol_valid": all(
            selector_decisions_are_valid(
                row=row,
                frozen_queues=frozen_queues,
            )
            for row in final_rows
        ),
        "harness_question_round_limit_respected": all(
            len(row.get("asked_questions") or ())
            <= HARNESS_QUESTION_ROUND_LIMIT
            for row in final_rows
        ),
    }
    if not all(checks.values()):
        raise SelectorError("completion checks failed")
    complete = {
        "schema_version": SCHEMA_COMPLETE,
        "manifest_identity": manifest_identity,
        "tasks": len(final_rows),
        "manifest_sha256": sha256_file(manifest_path),
        "results_sha256": sha256_file(output_path),
        "checks": checks,
    }
    complete_path = output_path.with_suffix(
        output_path.suffix + ".complete.json"
    )
    if complete_path.exists() and read_json(complete_path) != complete:
        raise SelectorError("existing completion receipt changed")
    if not complete_path.exists():
        atomic_json(complete_path, complete)
    return True


def build_manifest(
    *,
    args: argparse.Namespace,
    harness: tuple[Any, Any, Any, Any, Any],
    tasks: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
    selector_prompt: str,
    selector_followup_prompt: str | None,
) -> dict[str, Any]:
    interact, prompts, extract_code, _extract_question, _is_code = harness
    task_ids = sorted(str(task["task_id"]) for task in tasks)
    payload = {
        "schema_version": SCHEMA_MANIFEST,
        "method": METHOD,
        "model": args.model,
        "judge_model": args.judge_model,
        "selector_max_tokens": args.selector_max_tokens,
        "code_max_tokens": args.code_max_tokens,
        "reasoning_effort": args.reasoning_effort,
        "selector_tool_schema": args.selector_tool_schema,
        "temperature": args.model_temperature,
        "consistency": args.consistency,
        "max_retries": args.max_retries,
        "protocol": {
            "harness_question_round_limit": (
                HARNESS_QUESTION_ROUND_LIMIT
            ),
            "automatic_stop_after_harness_round_limit": True,
            "automatic_stop_on_queue_exhaustion": True,
            "model_stop_action_available": True,
            "termination_policy": (
                "dialogue_conditioned_positive_marginal_utility"
            ),
            "question_generation_available": False,
            "question_rewriting_available": False,
            "selection_output": (
                "one_remaining_candidate_id_or_finish_clarification"
            ),
            "selector_inputs": (
                "system_prompt+requirement+conversation_history"
                "+remaining_frozen_queue"
            ),
            "gold_visible_to_selector": False,
            "same_backbone_for_selector_and_final_code": True,
            "selector_prompt_schedule": (
                "initial_then_followup"
                if selector_followup_prompt is not None
                else "single"
            ),
        },
        "sources": {
            "data": {
                "path": str(args.data.resolve()),
                "sha256": sha256_file(args.data),
            },
            "plan": {
                "path": str(args.plan.resolve()),
                "sha256": sha256_file(args.plan),
                "schema_version": plan.get("schema_version"),
            },
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "transport": {
                "path": str(args.transport.resolve()),
                "sha256": sha256_file(args.transport),
            },
            "clarifycodebench": {
                "release": str(args.release.resolve()),
                "system_prompt_sha256": hashlib.sha256(
                    prompts.SYSTEM_PROMPT.encode("utf-8")
                ).hexdigest(),
                "judge_prompt_sha256": hashlib.sha256(
                    prompts.JUDGE_PROMPT.encode("utf-8")
                ).hexdigest(),
                "interact_sha256": sha256_file(Path(interact.__file__)),
                "extract_sha256": sha256_file(
                    Path(sys.modules[extract_code.__module__].__file__)
                ),
            },
            "selector_prompt_sha256": hashlib.sha256(
                selector_prompt.encode("utf-8")
            ).hexdigest(),
            "selector_prompt_path": (
                str(args.selector_prompt_file.resolve())
                if args.selector_prompt_file is not None
                else None
            ),
            "selector_followup_prompt_sha256": (
                hashlib.sha256(
                    selector_followup_prompt.encode("utf-8")
                ).hexdigest()
                if selector_followup_prompt is not None
                else None
            ),
            "selector_followup_prompt_path": (
                str(args.selector_followup_prompt_file.resolve())
                if args.selector_followup_prompt_file is not None
                else None
            ),
        },
        "scope": {
            "tasks": len(tasks),
            "task_ids_sha256": hashlib.sha256(
                "\n".join(task_ids).encode("utf-8")
            ).hexdigest(),
            "queue_questions": sum(
                len(task["questions"])
                for task in plan["tasks"].values()
            ),
        },
    }
    return {**payload, "manifest_identity": stable_hash(payload)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument(
        "--transport",
        type=Path,
        default=Path(__file__).with_name("openai_transport.py"),
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--judge-model", default="gpt-4o")
    parser.add_argument(
        "--base-url",
        default=(
            os.environ.get("CONTRA_BASE_URL")
            or os.environ.get("CLARIFY_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        ),
    )
    parser.add_argument("--selector-max-tokens", type=int, default=8192)
    parser.add_argument("--selector-prompt-file", type=Path)
    parser.add_argument("--selector-followup-prompt-file", type=Path)
    parser.add_argument("--code-max-tokens", type=int, default=64000)
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument(
        "--model-temperature",
        choices=("0", "none"),
        default="0",
        help=(
            "Use 0 for deterministic sampling or none when the model API "
            "does not support a temperature parameter."
        ),
    )
    parser.add_argument(
        "--selector-tool-schema",
        choices=("chat", "responses"),
        default="chat",
        help=(
            "Function-tool schema expected by the configured transport "
            "endpoint."
        ),
    )
    parser.add_argument("--consistency", type=int, default=3)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    if args.reasoning_effort != "low":
        parser.error("the selector experiment requires reasoning_effort=low")
    args.model_temperature = (
        None if args.model_temperature == "none" else 0.0
    )
    if args.consistency != 3:
        parser.error("the frozen ClarifyCodeBench judge requires three votes")
    if (
        args.selector_max_tokens < 1
        or args.code_max_tokens < 1
        or args.max_retries < 1
        or args.workers < 1
        or (args.limit is not None and args.limit < 1)
    ):
        parser.error("token, retry, worker, and limit values must be positive")
    if not args.dry_run and not args.confirm:
        parser.error("--confirm is required for API execution")
    return args


def main() -> int:
    args = parse_args()
    selector_prompt = SELECTOR_SYSTEM_PROMPT
    if args.selector_prompt_file is not None:
        selector_prompt = args.selector_prompt_file.read_text(
            encoding="utf-8"
        ).strip()
        if not selector_prompt:
            raise SelectorError("selector prompt file is empty")
    selector_followup_prompt: str | None = None
    if args.selector_followup_prompt_file is not None:
        selector_followup_prompt = (
            args.selector_followup_prompt_file.read_text(
                encoding="utf-8"
            ).strip()
        )
        if not selector_followup_prompt:
            raise SelectorError("selector followup prompt file is empty")
    harness = _bootstrap(args.release)
    interact = harness[0]
    tasks = interact.load_tasks(args.data)
    plan = read_json(args.plan)
    queues = validate_frozen_plan(
        plan=plan,
        tasks=tasks,
        benchmark_sha256=sha256_file(args.data),
    )
    manifest = build_manifest(
        args=args,
        harness=harness,
        tasks=tasks,
        plan=plan,
        selector_prompt=selector_prompt,
        selector_followup_prompt=selector_followup_prompt,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    if args.dry_run:
        return 0

    lock_path = args.out.with_suffix(args.out.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    run_lock = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(run_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        run_lock.close()
        raise SelectorError(
            f"another selector runner owns {lock_path}"
        ) from exc

    manifest_path = args.out.with_suffix(args.out.suffix + ".manifest.json")
    if manifest_path.exists():
        if read_json(manifest_path) != manifest:
            raise SelectorError("refusing to resume: manifest changed")
    else:
        atomic_json(manifest_path, manifest)

    existing_rows = _resume_rows(args.out)
    done: set[str] = set()
    for row in existing_rows:
        task_id = row.get("task_id")
        if (
            row.get("schema_version") != SCHEMA_RESULT
            or row.get("finished") is not True
            or row.get("manifest_identity")
            != manifest["manifest_identity"]
            or not isinstance(task_id, str)
            or task_id in done
        ):
            raise SelectorError("existing output contains invalid final rows")
        done.add(task_id)
    todo = [task for task in tasks if task["task_id"] not in done]
    if args.limit is not None:
        todo = todo[: args.limit]
    if not todo:
        if args.limit is None:
            expected_ids = [str(task["task_id"]) for task in tasks]
            if not write_completion_if_full(
                output_path=args.out,
                manifest_path=manifest_path,
                manifest_identity=manifest["manifest_identity"],
                expected_task_ids=expected_ids,
                frozen_queues=queues,
            ):
                raise SelectorError(
                    "no pending tasks but output coverage is incomplete"
                )
        return 0

    client = _load_gateway_client(
        transport_path=args.transport,
        base_url=args.base_url,
    )
    selector = DialogueQuestionSelector(
        client,
        model=args.model,
        max_tokens=args.selector_max_tokens,
        reasoning_effort=args.reasoning_effort,
        max_retries=args.max_retries,
        tool_schema=args.selector_tool_schema,
        temperature=args.model_temperature,
        system_prompt=selector_prompt,
        followup_system_prompt=selector_followup_prompt,
    )
    output_writer = DurableWriter(args.out)
    attempt_writer = DurableWriter(
        args.out.with_suffix(args.out.suffix + ".attempts.jsonl")
    )

    def execute(task: Mapping[str, Any]) -> dict[str, Any]:
        task_id = str(task["task_id"])
        state_dir = args.out.with_suffix(args.out.suffix + ".state")
        event_path = state_dir / (
            hashlib.sha256(task_id.encode("utf-8")).hexdigest() + ".jsonl"
        )
        journal = TaskEventJournal(
            event_path,
            manifest_identity=manifest["manifest_identity"],
            task_id=task_id,
        )

        def generate_final(
            messages: Sequence[Mapping[str, Any]],
        ) -> tuple[str, dict[str, Any]]:
            return generate_final_content(
                client,
                model=args.model,
                messages=messages,
                max_tokens=args.code_max_tokens,
                reasoning_effort=args.reasoning_effort,
                max_retries=args.max_retries,
                temperature=args.model_temperature,
            )

        row = run_task(
            harness=harness,
            task=task,
            frozen_queue=queues[task_id],
            selector=selector,
            generate_final=generate_final,
            judge_client=client,
            judge_model=args.judge_model,
            consistency=args.consistency,
            max_retries=args.max_retries,
            resume_events=journal.rows,
            event_sink=journal.append,
        )
        row["manifest_identity"] = manifest["manifest_identity"]
        row["model"] = args.model
        row["reasoning_effort"] = args.reasoning_effort
        row["completed_at"] = utc_now()
        return row

    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(execute, task): task for task in todo}
        for future in as_completed(futures):
            task = futures[future]
            try:
                row = future.result()
            except Exception as exc:  # noqa: BLE001
                row = {
                    "schema_version": SCHEMA_RESULT,
                    "manifest_identity": manifest["manifest_identity"],
                    "task_id": task["task_id"],
                    "finished": False,
                    "failure_reason": (
                        f"{type(exc).__name__}:{str(exc)[:2000]}"
                    ),
                    "completed_at": utc_now(),
                }
            if row.get("finished") is True:
                output_writer.append(row)
            else:
                failures += 1
                attempt_writer.append(row)

    if failures:
        return 1
    if args.limit is None:
        expected_ids = [str(task["task_id"]) for task in tasks]
        if not write_completion_if_full(
            output_path=args.out,
            manifest_path=manifest_path,
            manifest_identity=manifest["manifest_identity"],
            expected_task_ids=expected_ids,
            frozen_queues=queues,
        ):
            raise SelectorError("completed output does not cover all tasks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
