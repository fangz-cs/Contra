#!/usr/bin/env python3
"""Run the resumable Decision-Relevance + Resolution-Span AV replay."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import random
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


RAW_FIELDS = frozenset(
    {
        "candidate_id",
        "clarification_question",
        "complete_requirement",
        "interpretation_a",
        "interpretation_b",
    }
)
SCHEMA_MANIFEST = "av-decision-resolution-manifest-v1"
SCHEMA_CHUNK = "av-decision-resolution-chunk-v1"
SCHEMA_CALL = "av-decision-resolution-call-v1"
SCHEMA_CHUNK_RESULT = "av-decision-resolution-chunk-result-v1"
SCHEMA_RESULT = "av-decision-resolution-candidate-result-v1"
SCHEMA_COMPLETE = "av-decision-resolution-complete-v1"
MAX_PROTOCOL_ATTEMPTS = 3
MAX_TRANSPORT_RETRIES = 3
MAX_RETRY_SLEEP_SECONDS = 30.0
DEFAULT_TIMEOUT_SECONDS = 300.0
TRANSIENT_HTTP_STATUSES = frozenset(
    {408, 409, 425, 429, 499, 500, 502, 503, 504, 520, 521, 522, 524, 529}
)
PERMANENT_HTTP_STATUSES = frozenset(
    {400, 401, 402, 403, 404, 405, 413, 415, 422}
)

AGENT1_CORE_PROMPT = """\
You are a Decision-Relevance Classifier for requirements clarification in code generation.

Given a task specification S, a clarification question q, and two answer hypotheses h_A and h_B, determine whether q concerns task-required solution behavior.

A clarification question is decision-relevant exactly when:

1. S requires a correct solution to define the behavior affected by q; and
2. choosing h_A rather than h_B changes that required behavior.

Assess decision relevance independently of whether S already resolves q:
criterion 2 asks whether h_A and h_B prescribe different task behaviors, not
whether both comply with S.

Evaluate only behavior that S requires a correct solution to define."""

AGENT2_CORE_PROMPT = """\
You are a Specification-Resolution Extractor for requirements clarification in code generation.

Given a task specification S, a clarification question q, and two answer hypotheses h_A and h_B, extract the shortest contiguous span in S that, by itself, directly selects exactly one of the two answer hypotheses.

Return the span verbatim. Return null when no such span exists."""

AGENT1_BATCH_PROTOCOL = """\

The user may provide multiple candidates from the same task specification. Evaluate each candidate independently; candidates do not provide evidence for one another.

Return strict JSON only:
{"decisions": [{"candidate_id": "<provided local ID>", "decision_relevant": true | false}]}

Return exactly one decision for every provided candidate ID and no other IDs."""

AGENT2_BATCH_PROTOCOL = """\

The user may provide multiple candidates from the same task specification. Evaluate each candidate independently; candidates do not provide evidence for one another.

Return strict JSON only:
{"resolutions": [{"candidate_id": "<provided local ID>", "resolution_span": "<verbatim span from S>" | null}]}

Return exactly one resolution for every provided candidate ID and no other IDs."""

DEMO_UNRESOLVED_SPECIFICATION = """\
Write a program that reads N followed by N integers. Print the square of each integer.

Input
The input contains N integers.

Output
Print the square of each integer."""

DEMO_RESOLVED_SPECIFICATION = """\
Write a program that reads N followed by N integers.

Input
The input contains N integers.

Output
Print all N squares on one line, separated by single spaces."""

DEMO_FORMAT_QUESTION = (
    "Should the N squares be printed on one line separated by spaces, "
    "or should each square be printed on a separate line?"
)
DEMO_FORMAT_A = "Print all N squares on one line separated by spaces."
DEMO_FORMAT_B = "Print each square on a separate line."
DEMO_INVALID_QUESTION = (
    "If an input token is not an integer, should the program print ERROR "
    "or terminate without output?"
)
DEMO_INVALID_A = "Print ERROR."
DEMO_INVALID_B = "Terminate without output."
DEMO_RESOLUTION_SPAN = (
    "Print all N squares on one line, separated by single spaces."
)


class ReplayError(RuntimeError):
    """A frozen input, protocol, durability, or coverage invariant failed."""


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    group_id: str
    requirement: str
    question: str
    hypothesis_a: str
    hypothesis_b: str


@dataclass(frozen=True)
class ChunkItem:
    local_id: str
    candidate: Candidate


@dataclass(frozen=True)
class Chunk:
    stage: str
    group_id: str
    ordinal: int
    requirement: str
    items: tuple[ChunkItem, ...]
    input_sha256: str
    chunk_id: str


@dataclass(frozen=True)
class ModelConfig:
    name: str
    model_id: str
    base_url_env: str
    api_key_env: str
    max_tokens: int
    temperature: float
    workers: int
    reasoning_effort: str


@dataclass(frozen=True)
class TransportResult:
    status: str
    text: str | None
    finish_reason: str | None
    error: str | None
    retry_reasons: tuple[Mapping[str, Any], ...]
    latency_seconds: float
    usage: Mapping[str, Any]
    provider_metadata: Mapping[str, Any]


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def stable_json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    ).encode("utf-8")


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.new-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def write_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise ReplayError(f"{path}: immutable artifact mismatch")
        return
    atomic_bytes(path, payload)


def repair_torn_tail(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    raw = path.read_bytes()
    if raw.endswith(b"\n"):
        return
    start = raw.rfind(b"\n") + 1
    tail = raw[start:]
    try:
        complete = isinstance(json.loads(tail), dict)
    except (UnicodeDecodeError, json.JSONDecodeError):
        complete = False
    with path.open("r+b") as handle:
        if complete:
            handle.seek(0, os.SEEK_END)
            handle.write(b"\n")
        else:
            handle.truncate(start)
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(
    path: Path, *, repair_output_journal: bool = False
) -> list[dict[str, Any]]:
    if repair_output_journal:
        repair_torn_tail(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8", errors="strict").splitlines(), 1
    ):
        if not line.strip():
            raise ReplayError(f"{path}:{line_number}: blank line")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReplayError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(value, dict):
            raise ReplayError(f"{path}:{line_number}: expected object")
        rows.append(value)
    return rows


def append_jsonl(
    path: Path, row: Mapping[str, Any], *, lock: threading.Lock
) -> None:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        repair_torn_tail(path)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())


def load_candidates(path: Path) -> list[Candidate]:
    rows = read_jsonl(path)
    if not rows:
        raise ReplayError("candidate input is empty")
    candidates: list[Candidate] = []
    seen: set[str] = set()
    group_requirements: dict[str, str] = {}
    for index, row in enumerate(rows):
        if set(row) != RAW_FIELDS:
            raise ReplayError(f"candidate {index}: field isolation violation")
        values: dict[str, str] = {}
        for field in RAW_FIELDS:
            value = row[field]
            if not isinstance(value, str) or not value.strip():
                raise ReplayError(f"candidate {index}: {field} is empty")
            values[field] = value
        candidate_id = values["candidate_id"]
        if candidate_id in seen:
            raise ReplayError(f"duplicate candidate ID: {candidate_id}")
        seen.add(candidate_id)
        group_id = candidate_id.split("::", 1)[0]
        if group_id == candidate_id:
            raise ReplayError(f"candidate ID lacks task prefix: {candidate_id}")
        requirement = values["complete_requirement"]
        prior = group_requirements.setdefault(group_id, requirement)
        if prior != requirement:
            raise ReplayError(f"{group_id}: requirement drift")
        candidates.append(
            Candidate(
                candidate_id=candidate_id,
                group_id=group_id,
                requirement=requirement,
                question=values["clarification_question"],
                hypothesis_a=values["interpretation_a"],
                hypothesis_b=values["interpretation_b"],
            )
        )
    return candidates


def candidate_payload(local_id: str, candidate: Candidate) -> dict[str, str]:
    return {
        "candidate_id": local_id,
        "clarification_question": candidate.question,
        "answer_hypothesis_A": candidate.hypothesis_a,
        "answer_hypothesis_B": candidate.hypothesis_b,
    }


def chunk_payload(
    stage: str,
    group_id: str,
    ordinal: int,
    requirement: str,
    items: Sequence[ChunkItem],
) -> dict[str, Any]:
    return {
        "stage": stage,
        "group_id": group_id,
        "ordinal": ordinal,
        "task_specification": requirement,
        "candidates": [
            {
                "local_id": item.local_id,
                "candidate_id": item.candidate.candidate_id,
                "question": item.candidate.question,
                "hypothesis_a": item.candidate.hypothesis_a,
                "hypothesis_b": item.candidate.hypothesis_b,
            }
            for item in items
        ],
    }


def build_chunks(
    candidates: Sequence[Candidate],
    *,
    stage: str,
    max_candidates: int,
    max_prompt_chars: int,
) -> list[Chunk]:
    if stage not in {"decision_relevance", "specification_resolution"}:
        raise ReplayError(f"unknown stage: {stage}")
    if max_candidates < 1 or max_prompt_chars < 1:
        raise ReplayError("invalid chunk limits")
    groups: OrderedDict[str, list[Candidate]] = OrderedDict()
    for candidate in candidates:
        groups.setdefault(candidate.group_id, []).append(candidate)
    chunks: list[Chunk] = []
    for group_id, group_candidates in groups.items():
        requirement = group_candidates[0].requirement
        current: list[Candidate] = []
        ordinal = 0

        def emit(values: Sequence[Candidate], chunk_ordinal: int) -> None:
            items = tuple(
                ChunkItem(local_id=f"q{index:03d}", candidate=candidate)
                for index, candidate in enumerate(values, 1)
            )
            payload = chunk_payload(
                stage, group_id, chunk_ordinal, requirement, items
            )
            input_sha256 = stable_json_sha256(payload)
            chunks.append(
                Chunk(
                    stage=stage,
                    group_id=group_id,
                    ordinal=chunk_ordinal,
                    requirement=requirement,
                    items=items,
                    input_sha256=input_sha256,
                    chunk_id=f"{stage[:2]}_{input_sha256[:24]}",
                )
            )

        for candidate in group_candidates:
            proposed = [*current, candidate]
            proposed_items = tuple(
                ChunkItem(local_id=f"q{index:03d}", candidate=value)
                for index, value in enumerate(proposed, 1)
            )
            prompt_size = len(
                canonical_json(
                    {
                        "task_specification": requirement,
                        "candidates": [
                            candidate_payload(item.local_id, item.candidate)
                            for item in proposed_items
                        ],
                    }
                )
            )
            if current and (
                len(proposed) > max_candidates
                or prompt_size > max_prompt_chars
            ):
                emit(current, ordinal)
                ordinal += 1
                current = [candidate]
            else:
                current = proposed
            single_size = len(
                canonical_json(
                    {
                        "task_specification": requirement,
                        "candidates": [
                            candidate_payload("q001", current[-1])
                        ],
                    }
                )
            )
            if single_size > max_prompt_chars:
                raise ReplayError(
                    f"{candidate.candidate_id}: exceeds prompt character limit"
                )
        if current:
            emit(current, ordinal)
    if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
        raise ReplayError("chunk ID collision")
    emitted_ids = [
        item.candidate.candidate_id
        for chunk in chunks
        for item in chunk.items
    ]
    expected_ids = [candidate.candidate_id for candidate in candidates]
    if emitted_ids != expected_ids:
        raise ReplayError("chunking changed candidate order or coverage")
    return chunks


def chunk_record(chunk: Chunk) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_CHUNK,
        "chunk_id": chunk.chunk_id,
        "chunk_input_sha256": chunk.input_sha256,
        "stage": chunk.stage,
        "group_id": chunk.group_id,
        "ordinal": chunk.ordinal,
        "candidate_count": len(chunk.items),
        "candidate_ids": [
            item.candidate.candidate_id for item in chunk.items
        ],
    }


def demo_user(
    specification: str,
    question: str,
    hypothesis_a: str,
    hypothesis_b: str,
) -> str:
    return json.dumps(
        {
            "task_specification": specification,
            "candidates": [
                {
                    "candidate_id": "q001",
                    "clarification_question": question,
                    "answer_hypothesis_A": hypothesis_a,
                    "answer_hypothesis_B": hypothesis_b,
                }
            ],
        },
        ensure_ascii=False,
        indent=2,
    )


def build_messages(chunk: Chunk) -> list[dict[str, str]]:
    final_user = json.dumps(
        {
            "task_specification": chunk.requirement,
            "candidates": [
                candidate_payload(item.local_id, item.candidate)
                for item in chunk.items
            ],
        },
        ensure_ascii=False,
        indent=2,
    )
    if chunk.stage == "decision_relevance":
        return [
            {
                "role": "system",
                "content": AGENT1_CORE_PROMPT + AGENT1_BATCH_PROTOCOL,
            },
            {
                "role": "user",
                "content": demo_user(
                    DEMO_RESOLVED_SPECIFICATION,
                    DEMO_FORMAT_QUESTION,
                    DEMO_FORMAT_A,
                    DEMO_FORMAT_B,
                ),
            },
            {
                "role": "assistant",
                "content": (
                    '{"decisions":[{"candidate_id":"q001",'
                    '"decision_relevant":true}]}'
                ),
            },
            {
                "role": "user",
                "content": demo_user(
                    DEMO_UNRESOLVED_SPECIFICATION,
                    DEMO_INVALID_QUESTION,
                    DEMO_INVALID_A,
                    DEMO_INVALID_B,
                ),
            },
            {
                "role": "assistant",
                "content": (
                    '{"decisions":[{"candidate_id":"q001",'
                    '"decision_relevant":false}]}'
                ),
            },
            {"role": "user", "content": final_user},
        ]
    return [
        {
            "role": "system",
            "content": AGENT2_CORE_PROMPT + AGENT2_BATCH_PROTOCOL,
        },
        {
            "role": "user",
            "content": demo_user(
                DEMO_UNRESOLVED_SPECIFICATION,
                DEMO_FORMAT_QUESTION,
                DEMO_FORMAT_A,
                DEMO_FORMAT_B,
            ),
        },
        {
            "role": "assistant",
            "content": (
                '{"resolutions":[{"candidate_id":"q001",'
                '"resolution_span":null}]}'
            ),
        },
        {
            "role": "user",
            "content": demo_user(
                DEMO_RESOLVED_SPECIFICATION,
                DEMO_FORMAT_QUESTION,
                DEMO_FORMAT_A,
                DEMO_FORMAT_B,
            ),
        },
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "resolutions": [
                        {
                            "candidate_id": "q001",
                            "resolution_span": DEMO_RESOLUTION_SPAN,
                        }
                    ]
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
        {"role": "user", "content": final_user},
    ]


def response_schema(chunk: Chunk) -> dict[str, Any]:
    local_ids = [item.local_id for item in chunk.items]
    common_properties: dict[str, Any] = {
        "candidate_id": {"type": "string", "enum": local_ids}
    }
    if chunk.stage == "decision_relevance":
        container = "decisions"
        value_field = "decision_relevant"
        common_properties[value_field] = {"type": "boolean"}
    else:
        container = "resolutions"
        value_field = "resolution_span"
        common_properties[value_field] = {
            "anyOf": [{"type": "string"}, {"type": "null"}]
        }
    return {
        "type": "object",
        "properties": {
            container: {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": common_properties,
                    "required": ["candidate_id", value_field],
                    "additionalProperties": False,
                },
                "minItems": len(local_ids),
                "maxItems": len(local_ids),
            }
        },
        "required": [container],
        "additionalProperties": False,
    }


def parse_response(chunk: Chunk, text: str) -> list[dict[str, Any]]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ReplayError(f"response is not strict JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ReplayError("response root is not an object")
    if chunk.stage == "decision_relevance":
        container = "decisions"
        result_field = "decision_relevant"
    else:
        container = "resolutions"
        result_field = "resolution_span"
    if set(value) != {container} or not isinstance(value[container], list):
        raise ReplayError(f"response must contain only {container}")
    expected = [item.local_id for item in chunk.items]
    expected_set = set(expected)
    parsed: dict[str, Any] = {}
    for index, row in enumerate(value[container]):
        if not isinstance(row, dict) or set(row) != {
            "candidate_id",
            result_field,
        }:
            raise ReplayError(f"{container}[{index}]: invalid fields")
        candidate_id = row["candidate_id"]
        if not isinstance(candidate_id, str) or candidate_id not in expected_set:
            raise ReplayError(f"{container}[{index}]: unknown candidate ID")
        if candidate_id in parsed:
            raise ReplayError(f"{container}[{index}]: duplicate candidate ID")
        result = row[result_field]
        if chunk.stage == "decision_relevance":
            if not isinstance(result, bool):
                raise ReplayError(
                    f"{container}[{index}]: decision is not boolean"
                )
        else:
            if result is not None and (
                not isinstance(result, str) or not result
            ):
                raise ReplayError(
                    f"{container}[{index}]: span is not string or null"
                )
            if isinstance(result, str) and result not in chunk.requirement:
                raise ReplayError(
                    f"{container}[{index}]: span is not verbatim in S"
                )
        parsed[candidate_id] = result
    if set(parsed) != expected_set:
        missing = sorted(expected_set - set(parsed))
        raise ReplayError(f"response candidate coverage mismatch: {missing}")
    return [
        {"candidate_id": local_id, result_field: parsed[local_id]}
        for local_id in expected
    ]


def load_model_config(path: Path, model_name: str) -> ModelConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or not isinstance(raw.get("models"), list):
        raise ReplayError(f"{path}: expected a top-level models list")
    matches = [
        row
        for row in raw["models"]
        if isinstance(row, Mapping) and row.get("name") == model_name
    ]
    if len(matches) != 1:
        raise ReplayError(f"{path}: model {model_name!r} is not unique")
    row = matches[0]
    allowed = {
        "name",
        "model_id",
        "base_url_env",
        "api_key_env",
        "max_tokens",
        "temperature",
        "workers",
        "reasoning_effort",
    }
    if set(row) != allowed:
        raise ReplayError(
            f"{path}: model fields differ: {sorted(set(row) ^ allowed)}"
        )
    config = ModelConfig(
        name=str(row["name"]),
        model_id=str(row["model_id"]),
        base_url_env=str(row["base_url_env"]),
        api_key_env=str(row["api_key_env"]),
        max_tokens=int(row["max_tokens"]),
        temperature=float(row["temperature"]),
        workers=int(row["workers"]),
        reasoning_effort=str(row["reasoning_effort"]),
    )
    if (
        not config.name
        or not config.model_id
        or config.max_tokens < 1
        or config.workers < 1
    ):
        raise ReplayError(f"{path}: invalid model configuration")
    return config


def usage_to_dict(usage: object) -> dict[str, Any]:
    if usage is None:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
        }
    if hasattr(usage, "model_dump"):
        raw = usage.model_dump()
    elif isinstance(usage, Mapping):
        raw = dict(usage)
    else:
        raw = {}
    details = raw.get("completion_tokens_details")
    reasoning_tokens = (
        details.get("reasoning_tokens")
        if isinstance(details, Mapping)
        else raw.get("reasoning_tokens")
    )
    return {
        "input_tokens": int(raw.get("prompt_tokens") or 0),
        "output_tokens": int(raw.get("completion_tokens") or 0),
        "reasoning_tokens": int(reasoning_tokens or 0),
        "total_tokens": int(raw.get("total_tokens") or 0),
    }


def classify_exception(exc: BaseException) -> tuple[str, int | None, str]:
    status_code = getattr(exc, "status_code", None)
    message = f"{type(exc).__name__}: {exc}"
    if status_code in PERMANENT_HTTP_STATUSES:
        return "permanent", status_code, message
    if status_code in TRANSIENT_HTTP_STATUSES:
        return "transient", status_code, message
    if isinstance(status_code, int) and 400 <= status_code < 500:
        return "permanent", status_code, message
    return "transient", status_code, message


class OpenAITransport:
    def __init__(self, config: ModelConfig):
        base_url = os.environ.get(config.base_url_env)
        api_key = os.environ.get(config.api_key_env)
        if not base_url:
            raise ReplayError(f"{config.base_url_env} is not set")
        if not api_key:
            raise ReplayError(f"{config.api_key_env} is not set")
        try:
            timeout = float(
                os.environ.get(
                    "AV_V4_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS
                )
            )
        except ValueError as exc:
            raise ReplayError("AV_V4_TIMEOUT_SECONDS must be numeric") from exc
        import openai

        self.config = config
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,
        )

    def call(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        schema: Mapping[str, Any],
        schema_name: str,
    ) -> TransportResult:
        started = time.monotonic()
        retries: list[dict[str, Any]] = []
        for attempt in range(MAX_TRANSPORT_RETRIES + 1):
            attempt_started = time.monotonic()
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model_id,
                    messages=list(messages),
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    reasoning_effort=self.config.reasoning_effort,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": schema_name,
                            "strict": True,
                            "schema": schema,
                        },
                    },
                )
                if not response.choices:
                    raise RuntimeError("provider returned no choices")
                choice = response.choices[0]
                text = getattr(choice.message, "content", None) or ""
                finish_reason = getattr(choice, "finish_reason", None)
                if not text.strip() and finish_reason != "length":
                    raise RuntimeError("provider returned empty completion")
                return TransportResult(
                    status="ok",
                    text=text,
                    finish_reason=finish_reason,
                    error=None,
                    retry_reasons=tuple(retries),
                    latency_seconds=round(time.monotonic() - started, 3),
                    usage=usage_to_dict(getattr(response, "usage", None)),
                    provider_metadata={
                        "id": getattr(response, "id", None),
                        "model": getattr(response, "model", None),
                        "created": getattr(response, "created", None),
                    },
                )
            except Exception as exc:
                kind, status_code, message = classify_exception(exc)
                record = {
                    "attempt": attempt + 1,
                    "at": utc_now(),
                    "kind": kind,
                    "status_code": status_code,
                    "message": message[:8000],
                    "latency_seconds": round(
                        time.monotonic() - attempt_started, 3
                    ),
                }
                retries.append(record)
                if kind == "permanent" or attempt >= MAX_TRANSPORT_RETRIES:
                    return TransportResult(
                        status=f"failed_{kind}",
                        text=None,
                        finish_reason=None,
                        error=message,
                        retry_reasons=tuple(retries),
                        latency_seconds=round(
                            time.monotonic() - started, 3
                        ),
                        usage={},
                        provider_metadata={},
                    )
                sleep_seconds = min(
                    MAX_RETRY_SLEEP_SECONDS,
                    (2**attempt) * (0.75 + random.random() * 0.5),
                )
                record["sleep_seconds"] = round(sleep_seconds, 3)
                time.sleep(sleep_seconds)
        raise AssertionError("unreachable")


def execute_chunk(
    *,
    chunk: Chunk,
    transport: OpenAITransport,
    calls_path: Path,
    results_path: Path,
    journal_lock: threading.Lock,
) -> dict[str, Any]:
    messages = build_messages(chunk)
    schema = response_schema(chunk)
    for protocol_attempt in range(1, MAX_PROTOCOL_ATTEMPTS + 1):
        result = transport.call(
            messages=messages,
            schema=schema,
            schema_name=f"{chunk.stage}_n{len(chunk.items)}",
        )
        call_id = stable_json_sha256(
            {
                "chunk_id": chunk.chunk_id,
                "protocol_attempt": protocol_attempt,
                "started_sequence": len(messages),
                "at": utc_now(),
            }
        )[:24]
        call_row: dict[str, Any] = {
            "schema_version": SCHEMA_CALL,
            "call_id": call_id,
            "chunk_id": chunk.chunk_id,
            "chunk_input_sha256": chunk.input_sha256,
            "stage": chunk.stage,
            "protocol_attempt": protocol_attempt,
            "candidate_count": len(chunk.items),
            "candidate_ids": [
                item.candidate.candidate_id for item in chunk.items
            ],
            "message_sha256": stable_json_sha256(messages),
            "response_schema_sha256": stable_json_sha256(schema),
            "status": result.status,
            "completion_text": result.text,
            "finish_reason": result.finish_reason,
            "error": result.error,
            "retry_reasons": list(result.retry_reasons),
            "latency_seconds": result.latency_seconds,
            "usage": dict(result.usage),
            "provider_metadata": dict(result.provider_metadata),
            "completed_at": utc_now(),
        }
        parse_error: str | None = None
        outputs: list[dict[str, Any]] | None = None
        if result.status == "ok" and result.text is not None:
            try:
                outputs = parse_response(chunk, result.text)
            except ReplayError as exc:
                parse_error = str(exc)
        call_row["parse_error"] = parse_error
        append_jsonl(calls_path, call_row, lock=journal_lock)
        if outputs is not None:
            row = {
                "schema_version": SCHEMA_CHUNK_RESULT,
                "chunk_id": chunk.chunk_id,
                "chunk_input_sha256": chunk.input_sha256,
                "stage": chunk.stage,
                "group_id": chunk.group_id,
                "ordinal": chunk.ordinal,
                "candidate_count": len(chunk.items),
                "outputs": outputs,
                "logical_call_attempts": protocol_attempt,
                "completed_at": utc_now(),
            }
            append_jsonl(results_path, row, lock=journal_lock)
            return row
        if result.status != "ok":
            raise ReplayError(
                f"{chunk.chunk_id}: transport failed: {result.error}"
            )
        if protocol_attempt < MAX_PROTOCOL_ATTEMPTS:
            messages = [
                *messages,
                {"role": "assistant", "content": result.text or ""},
                {
                    "role": "user",
                    "content": (
                        "The response violated the required protocol: "
                        f"{parse_error}. Return a corrected JSON response "
                        "covering exactly the provided candidate IDs."
                    ),
                },
            ]
    raise ReplayError(
        f"{chunk.chunk_id}: exhausted protocol repair attempts"
    )


def validate_chunk_result(
    chunk: Chunk, row: Mapping[str, Any]
) -> dict[str, Any]:
    if (
        row.get("schema_version") != SCHEMA_CHUNK_RESULT
        or row.get("chunk_id") != chunk.chunk_id
        or row.get("chunk_input_sha256") != chunk.input_sha256
        or row.get("stage") != chunk.stage
        or row.get("group_id") != chunk.group_id
        or row.get("ordinal") != chunk.ordinal
        or row.get("candidate_count") != len(chunk.items)
    ):
        raise ReplayError(f"{chunk.chunk_id}: chunk result identity mismatch")
    outputs = row.get("outputs")
    if not isinstance(outputs, list):
        raise ReplayError(f"{chunk.chunk_id}: outputs are not a list")
    container = (
        "decisions"
        if chunk.stage == "decision_relevance"
        else "resolutions"
    )
    normalized = parse_response(
        chunk,
        json.dumps({container: outputs}, ensure_ascii=False),
    )
    if normalized != outputs:
        raise ReplayError(f"{chunk.chunk_id}: output order is not canonical")
    return dict(row)


def load_completed_chunks(
    path: Path, chunks: Sequence[Chunk]
) -> dict[str, dict[str, Any]]:
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    completed: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path, repair_output_journal=True):
        chunk_id = row.get("chunk_id")
        if not isinstance(chunk_id, str) or chunk_id not in chunk_by_id:
            raise ReplayError(f"{path}: unknown chunk result {chunk_id!r}")
        if chunk_id in completed:
            raise ReplayError(f"{path}: duplicate chunk result {chunk_id}")
        completed[chunk_id] = validate_chunk_result(
            chunk_by_id[chunk_id], row
        )
    return completed


def run_stage(
    *,
    chunks: Sequence[Chunk],
    transport: OpenAITransport,
    output_dir: Path,
    workers: int,
) -> dict[str, dict[str, Any]]:
    if not chunks:
        return {}
    stage = chunks[0].stage
    if any(chunk.stage != stage for chunk in chunks):
        raise ReplayError("run_stage received mixed stages")
    results_path = output_dir / f"{stage}_chunk_results.jsonl"
    calls_path = output_dir / "calls.jsonl"
    completed = load_completed_chunks(results_path, chunks)
    incomplete = [chunk for chunk in chunks if chunk.chunk_id not in completed]
    if not incomplete:
        return completed
    journal_lock = threading.Lock()
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                execute_chunk,
                chunk=chunk,
                transport=transport,
                calls_path=calls_path,
                results_path=results_path,
                journal_lock=journal_lock,
            ): chunk
            for chunk in incomplete
        }
        for index, future in enumerate(as_completed(futures), 1):
            chunk = futures[future]
            try:
                row = future.result()
                completed[chunk.chunk_id] = row
            except Exception as exc:
                failures.append(f"{chunk.chunk_id}: {type(exc).__name__}: {exc}")
            if index % 25 == 0 or index == len(incomplete):
                print(
                    f"{utc_now()} stage={stage} completed="
                    f"{len(completed)}/{len(chunks)} failures={len(failures)}",
                    flush=True,
                )
    if failures:
        preview = " | ".join(failures[:8])
        raise ReplayError(
            f"{stage}: {len(failures)} chunk failures; {preview}"
        )
    return load_completed_chunks(results_path, chunks)


def flatten_outputs(
    chunks: Sequence[Chunk],
    chunk_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    field = (
        "decision_relevant"
        if chunks and chunks[0].stage == "decision_relevance"
        else "resolution_span"
    )
    for chunk in chunks:
        row = chunk_results.get(chunk.chunk_id)
        if row is None:
            raise ReplayError(f"{chunk.chunk_id}: missing completed result")
        values = row["outputs"]
        if len(values) != len(chunk.items):
            raise ReplayError(f"{chunk.chunk_id}: output length drift")
        for item, value in zip(chunk.items, values, strict=True):
            if value["candidate_id"] != item.local_id:
                raise ReplayError(f"{chunk.chunk_id}: local ID order drift")
            candidate_id = item.candidate.candidate_id
            if candidate_id in output:
                raise ReplayError(f"duplicate flattened result: {candidate_id}")
            output[candidate_id] = value[field]
    return output


def manifest(
    *,
    input_path: Path,
    models_config_path: Path,
    config: ModelConfig,
    candidates: Sequence[Candidate],
    agent1_chunks: Sequence[Chunk],
    max_candidates: int,
    max_prompt_chars: int,
) -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    prompt_material = {
        "agent1_core": AGENT1_CORE_PROMPT,
        "agent2_core": AGENT2_CORE_PROMPT,
        "agent1_batch_protocol": AGENT1_BATCH_PROTOCOL,
        "agent2_batch_protocol": AGENT2_BATCH_PROTOCOL,
        "demonstrations": {
            "unresolved_specification": DEMO_UNRESOLVED_SPECIFICATION,
            "resolved_specification": DEMO_RESOLVED_SPECIFICATION,
            "format_question": DEMO_FORMAT_QUESTION,
            "format_hypothesis_a": DEMO_FORMAT_A,
            "format_hypothesis_b": DEMO_FORMAT_B,
            "invalid_input_question": DEMO_INVALID_QUESTION,
            "invalid_input_hypothesis_a": DEMO_INVALID_A,
            "invalid_input_hypothesis_b": DEMO_INVALID_B,
            "resolution_span": DEMO_RESOLUTION_SPAN,
        },
    }
    return {
        "schema_version": SCHEMA_MANIFEST,
        "method": (
            "Decision-Relevance Classification + "
            "Specification-Resolution Extraction"
        ),
        "method_version": "av-v4.0-post-diagnostic-frozen-r1",
        "model": config.name,
        "model_id": config.model_id,
        "model_parameters": {
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "reasoning_effort": config.reasoning_effort,
            "workers": config.workers,
        },
        "candidate_count": len(candidates),
        "task_count": len({candidate.group_id for candidate in candidates}),
        "candidate_input_path": str(input_path.resolve()),
        "candidate_input_sha256": sha256_file(input_path),
        "candidate_semantic_sha256": stable_json_sha256(
            [
                {
                    "candidate_id": candidate.candidate_id,
                    "complete_requirement": candidate.requirement,
                    "clarification_question": candidate.question,
                    "interpretation_a": candidate.hypothesis_a,
                    "interpretation_b": candidate.hypothesis_b,
                }
                for candidate in candidates
            ]
        ),
        "gold_visible": False,
        "benchmark_metadata_visible": False,
        "programs_visible": False,
        "execution_evidence_visible": False,
        "old_av_outputs_visible": False,
        "agent1_output_visible_to_agent2": False,
        "agent2_gate": "run only when decision_relevant is true",
        "retention_rule": (
            "decision_relevant AND resolution_span IS null"
        ),
        "missing_or_invalid_output_policy": (
            "fail the chunk; never map failure to a semantic decision"
        ),
        "resolution_span_validation": (
            "every non-null span must occur verbatim in the task specification"
        ),
        "batching": {
            "semantics": (
                "transport-only batching within one task; every candidate is "
                "judged independently and candidates are not evidence for "
                "one another"
            ),
            "max_candidates": max_candidates,
            "max_prompt_chars": max_prompt_chars,
            "agent1_chunk_count": len(agent1_chunks),
            "cross_task_batching": False,
        },
        "demonstration_source": "synthetic; no CCB examples",
        "prompt_material": prompt_material,
        "prompt_sha256": stable_json_sha256(prompt_material),
        "code_sha256": sha256_file(script_path),
        "models_config_path": str(models_config_path.resolve()),
        "models_config_sha256": sha256_file(models_config_path),
        "post_diagnostic_redesign": True,
    }


def verify_existing_complete(
    output_dir: Path, manifest_path: Path
) -> dict[str, Any] | None:
    complete_path = output_dir / "COMPLETE.json"
    if not complete_path.exists():
        return None
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if (
        not isinstance(complete, dict)
        or complete.get("schema_version") != SCHEMA_COMPLETE
        or complete.get("manifest_sha256") != sha256_file(manifest_path)
        or complete.get("results_sha256")
        != sha256_file(output_dir / "results.jsonl")
    ):
        raise ReplayError("existing COMPLETE.json failed integrity checks")
    return complete


def execute(args: argparse.Namespace) -> dict[str, Any] | None:
    candidates = load_candidates(args.input_jsonl)
    config = load_model_config(args.models_config, args.model)
    agent1_chunks = build_chunks(
        candidates,
        stage="decision_relevance",
        max_candidates=args.max_candidates_per_chunk,
        max_prompt_chars=args.max_prompt_chars,
    )
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_value = manifest(
        input_path=args.input_jsonl,
        models_config_path=args.models_config,
        config=config,
        candidates=candidates,
        agent1_chunks=agent1_chunks,
        max_candidates=args.max_candidates_per_chunk,
        max_prompt_chars=args.max_prompt_chars,
    )
    write_immutable(manifest_path, json_bytes(manifest_value))
    write_immutable(
        output_dir / "decision_relevance_chunks.jsonl",
        jsonl_bytes([chunk_record(chunk) for chunk in agent1_chunks]),
    )
    if args.prepare_only:
        return None
    if not args.confirm_live:
        raise ReplayError("live execution requires --confirm-live")
    existing_complete = verify_existing_complete(output_dir, manifest_path)
    if existing_complete is not None:
        return existing_complete

    transport = OpenAITransport(config)
    agent1_results = run_stage(
        chunks=agent1_chunks,
        transport=transport,
        output_dir=output_dir,
        workers=config.workers,
    )
    decisions = flatten_outputs(agent1_chunks, agent1_results)
    selected = [
        candidate
        for candidate in candidates
        if decisions[candidate.candidate_id] is True
    ]
    agent2_chunks = build_chunks(
        selected,
        stage="specification_resolution",
        max_candidates=args.max_candidates_per_chunk,
        max_prompt_chars=args.max_prompt_chars,
    )
    write_immutable(
        output_dir / "specification_resolution_chunks.jsonl",
        jsonl_bytes([chunk_record(chunk) for chunk in agent2_chunks]),
    )
    agent2_results = run_stage(
        chunks=agent2_chunks,
        transport=transport,
        output_dir=output_dir,
        workers=config.workers,
    )
    resolutions = flatten_outputs(agent2_chunks, agent2_results)
    if set(resolutions) != {
        candidate.candidate_id for candidate in selected
    }:
        raise ReplayError("Agent 2 gate or coverage mismatch")

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = candidate.candidate_id
        decision_relevant = decisions[candidate_id]
        resolution_checked = decision_relevant is True
        resolution_span = (
            resolutions[candidate_id] if resolution_checked else None
        )
        retained = decision_relevant is True and resolution_span is None
        rows.append(
            {
                "schema_version": SCHEMA_RESULT,
                "candidate_id": candidate_id,
                "decision_relevant": decision_relevant,
                "resolution_checked": resolution_checked,
                "resolution_span": resolution_span,
                "retained_for_behavioral_verification": retained,
            }
        )
    results_path = output_dir / "results.jsonl"
    write_immutable(results_path, jsonl_bytes(rows))
    calls = read_jsonl(
        output_dir / "calls.jsonl", repair_output_journal=True
    )
    counts = {
        "candidates": len(rows),
        "tasks": len({candidate.group_id for candidate in candidates}),
        "decision_relevant_true": sum(
            row["decision_relevant"] is True for row in rows
        ),
        "decision_relevant_false": sum(
            row["decision_relevant"] is False for row in rows
        ),
        "resolution_checked": sum(
            row["resolution_checked"] is True for row in rows
        ),
        "resolution_span_null": sum(
            row["resolution_checked"] is True
            and row["resolution_span"] is None
            for row in rows
        ),
        "resolution_span_nonnull": sum(
            isinstance(row["resolution_span"], str) for row in rows
        ),
        "retained_for_behavioral_verification": sum(
            row["retained_for_behavioral_verification"] is True
            for row in rows
        ),
        "agent1_chunks": len(agent1_chunks),
        "agent2_chunks": len(agent2_chunks),
        "physical_calls": len(calls),
        "protocol_repair_calls": sum(
            int(row.get("protocol_attempt") or 0) > 1 for row in calls
        ),
        "http_429_events": sum(
            reason.get("status_code") == 429
            for row in calls
            for reason in row.get("retry_reasons") or []
            if isinstance(reason, dict)
        ),
    }
    complete = {
        "schema_version": SCHEMA_COMPLETE,
        "method_version": manifest_value["method_version"],
        "completed_at": utc_now(),
        "manifest_sha256": sha256_file(manifest_path),
        "candidate_input_sha256": sha256_file(args.input_jsonl),
        "decision_relevance_chunks_sha256": sha256_file(
            output_dir / "decision_relevance_chunks.jsonl"
        ),
        "decision_relevance_chunk_results_sha256": sha256_file(
            output_dir / "decision_relevance_chunk_results.jsonl"
        ),
        "specification_resolution_chunks_sha256": sha256_file(
            output_dir / "specification_resolution_chunks.jsonl"
        ),
        "specification_resolution_chunk_results_sha256": sha256_file(
            output_dir / "specification_resolution_chunk_results.jsonl"
        ),
        "calls_sha256": sha256_file(output_dir / "calls.jsonl"),
        "results_sha256": sha256_file(results_path),
        "counts": counts,
        "checks": {
            "candidate_coverage_exact": len(rows) == len(candidates),
            "agent2_gate_exact": len(resolutions) == len(selected),
            "nonnull_spans_verbatim": all(
                not isinstance(row["resolution_span"], str)
                or row["resolution_span"]
                in candidate.requirement
                for row, candidate in zip(rows, candidates, strict=True)
            ),
            "no_semantic_fallbacks": True,
        },
    }
    if not all(complete["checks"].values()):
        raise ReplayError("final deterministic checks failed")
    atomic_bytes(output_dir / "COMPLETE.json", json_bytes(complete))
    return complete


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models-config", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--max-candidates-per-chunk", type=int, default=32)
    parser.add_argument("--max-prompt-chars", type=int, default=60000)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--confirm-live", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = args.output_dir / ".run.lock"
    lock_handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_handle.close()
        raise ReplayError("another AV v4 process owns the run lock") from exc
    try:
        result = execute(args)
        if result is not None:
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    finally:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
