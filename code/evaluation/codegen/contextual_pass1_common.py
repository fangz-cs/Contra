#!/usr/bin/env python3
"""Shared utilities for the dialogue-conditioned pass@1 experiment."""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


METHOD_ORDER = ("vanilla", "ask_or_assume", "clarifygpt", "ours")
MODEL_ORDER = (
    "gpt-5-mini",
    "gemini-3.1-pro-preview",
    "gpt-5.5",
    "grok-4.5",
)
FOLLOWUP = (
    "\nBased on our current conversation, please ask a clarifying question "
    "or generate the code."
)
CODE_NUDGE = (
    "Clarification is complete. Emit the final program in the required "
    "[CODE] format."
)
_QUESTION_RE = re.compile(
    r"\[QUESTION\](.*?)(?:\[/QUESTION\]|$)",
    re.DOTALL,
)
_CODE_TAG_RE = re.compile(
    r"\[CODE\](.*?)(?:\[/CODE\]|$)",
    re.DOTALL,
)
_CODE_FENCE_RE = re.compile(
    r"```[ \t]*(?P<lang>[A-Za-z0-9_+-]*)[ \t]*\n"
    r"(?P<code>.*?)(?:\n```|```|$)",
    re.DOTALL,
)


class ExperimentError(RuntimeError):
    """The frozen experiment inputs or outputs violate an invariant."""


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def read_json(path: str | Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentError(f"invalid JSON: {path}") from exc


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ExperimentError(f"cannot read JSONL: {path}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise ExperimentError(f"{path}:{line_number}: blank line")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ExperimentError(
                f"{path}:{line_number}: invalid JSON"
            ) from exc
        if not isinstance(value, dict):
            raise ExperimentError(
                f"{path}:{line_number}: expected a JSON object"
            )
        rows.append(value)
    return rows


def index_unique(
    rows: Iterable[Mapping[str, Any]],
    key: str,
    label: str,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for raw in rows:
        value = raw.get(key)
        if not isinstance(value, str) or not value or value in output:
            raise ExperimentError(f"{label}: missing or duplicate {key}")
        output[value] = dict(raw)
    return output


def atomic_json(path: str | Path, value: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    temporary = target.with_name(f".{target.name}.new-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def atomic_jsonl(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    ).encode("utf-8")
    temporary = target.with_name(f".{target.name}.new-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def task_sort_key(task_id: str) -> tuple[int, str]:
    match = re.fullmatch(r"task_(\d+)", task_id)
    if match:
        return int(match.group(1)), task_id
    return 10**12, task_id


def model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-")


def extract_question(text: str) -> str:
    match = _QUESTION_RE.search(text or "")
    question = match.group(1).strip() if match else (text or "").strip()
    if not question:
        raise ExperimentError("empty clarification question")
    return question


def render_question(question: str) -> str:
    if not question.strip():
        raise ExperimentError("cannot render an empty question")
    return f"[QUESTION]\n{question.strip()}\n[/QUESTION]"


def extract_code(text: str) -> tuple[str, str]:
    """Apply the official CCB extraction order without resampling."""
    if not isinstance(text, str) or not text.strip():
        return "", "empty_response"
    stripped = text.strip()
    if (
        stripped.startswith("[CODE]")
        and stripped.endswith("[/CODE]")
        and stripped.count("[CODE]") == 1
        and stripped.count("[/CODE]") == 1
    ):
        inner = stripped[len("[CODE]") : -len("[/CODE]")].strip()
        if inner:
            return inner, "exact_code_block"

    tagged = _CODE_TAG_RE.search(text)
    fenced = None
    for match in _CODE_FENCE_RE.finditer(text):
        language = (match.group("lang") or "").strip().lower()
        if language in ("", "python", "py"):
            fenced = match
            break
    if tagged and (not fenced or tagged.start() <= fenced.start()):
        return tagged.group(1).strip(), "tagged_code_fallback"
    if fenced:
        return (fenced.group("code") or "").strip(), "fenced_code_fallback"
    return stripped, "raw_text_fallback"
