#!/usr/bin/env python3
"""Generate one final program per frozen dialogue, with durable resume state."""
from __future__ import annotations

import argparse
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

from openai import OpenAI

from contextual_pass1_common import (
    METHOD_ORDER,
    ExperimentError,
    atomic_json,
    atomic_jsonl,
    extract_code,
    model_slug,
    read_json,
    read_jsonl,
    sha256_file,
    stable_hash,
    task_sort_key,
    utc_now,
)


RESULT_SCHEMA = "contextual-pass1-generation-result-v1"
SUMMARY_SCHEMA = "contextual-pass1-generation-summary-v1"
_THREAD_LOCAL = threading.local()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript-manifest", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=64_000)
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--temperature", default="none")
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    if not args.confirm:
        parser.error("--confirm is required for API execution")
    return args


def parse_temperature(value: str) -> float | None:
    if value.lower() in {"none", "null", "omit"}:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ExperimentError(f"invalid temperature: {value}") from exc


def client(args: argparse.Namespace, api_key: str) -> OpenAI:
    cached = getattr(_THREAD_LOCAL, "client", None)
    if cached is None:
        cached = OpenAI(
            base_url=args.base_url,
            api_key=api_key,
            timeout=args.request_timeout,
        )
        _THREAD_LOCAL.client = cached
    return cached


def state_path(root: Path, row: Mapping[str, Any]) -> Path:
    return root / "state" / str(row["method"]) / f"{row['task_id']}.json"


def request_identity(
    row: Mapping[str, Any],
    *,
    model: str,
    max_tokens: int,
    reasoning_effort: str,
    temperature: float | None,
) -> dict[str, Any]:
    return {
        "messages": row["messages"],
        "model": model,
        "max_tokens": max_tokens,
        "reasoning_effort": reasoning_effort,
        "temperature": temperature,
        "stream": False,
    }


def reusable_state(
    path: Path,
    *,
    request_sha256: str,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = read_json(path)
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != RESULT_SCHEMA
        or value.get("request_sha256") != request_sha256
    ):
        raise ExperimentError(f"incompatible generation state: {path}")
    if value.get("transport_status") == "ok":
        return value
    return None


def response_record(response: Any) -> dict[str, Any]:
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
        "usage": raw.get("usage") or {},
        "route_meta": raw.get("_route_meta") or {},
        "content": getattr(message, "content", None),
    }


def generate_one(
    args: argparse.Namespace,
    *,
    row: Mapping[str, Any],
    api_key: str,
    temperature: float | None,
    root: Path,
) -> dict[str, Any]:
    identity = request_identity(
        row,
        model=args.model,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
        temperature=temperature,
    )
    request_sha256 = stable_hash(identity)
    path = state_path(root, row)
    existing = reusable_state(path, request_sha256=request_sha256)
    if existing is not None:
        return existing

    request_kwargs: dict[str, Any] = {
        "model": args.model,
        "messages": row["messages"],
        "max_tokens": args.max_tokens,
        "reasoning_effort": args.reasoning_effort,
        "stream": False,
    }
    if temperature is not None:
        request_kwargs["temperature"] = temperature
    attempts: list[dict[str, Any]] = []
    response: Any | None = None
    for attempt in range(1, args.max_retries + 1):
        started = time.monotonic()
        try:
            response = client(args, api_key).chat.completions.create(
                **request_kwargs
            )
            break
        except Exception as exc:  # noqa: BLE001
            status = getattr(exc, "status_code", None)
            permanent = (
                isinstance(status, int)
                and 400 <= status < 500
                and status != 429
            )
            attempts.append(
                {
                    "attempt": attempt,
                    "latency_seconds": round(time.monotonic() - started, 6),
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:4000],
                    "status_code": status,
                    "permanent": permanent,
                }
            )
            if permanent or attempt == args.max_retries:
                break
            time.sleep(min(2 ** (attempt - 1), 30))

    base = {
        "schema_version": RESULT_SCHEMA,
        "model": args.model,
        "method": row["method"],
        "task_id": row["task_id"],
        "question_id": row["question_id"],
        "question_count": row["question_count"],
        "generation_messages_sha256": row["generation_messages_sha256"],
        "request_sha256": request_sha256,
        "request_config": {
            "base_url": args.base_url,
            "max_tokens": args.max_tokens,
            "reasoning_effort": args.reasoning_effort,
            "temperature": temperature,
            "stream": False,
        },
        "transport_attempts": attempts,
        "completed_at": utc_now(),
    }
    if response is None:
        result = {
            **base,
            "transport_status": "failed",
            "raw_content": "",
            "code": "",
            "parse_status": "not_generated",
            "response": None,
        }
    else:
        record = response_record(response)
        content = record.pop("content")
        raw_content = content if isinstance(content, str) else ""
        code, parse_status = extract_code(raw_content)
        result = {
            **base,
            "transport_status": "ok",
            "raw_content": raw_content,
            "code": code,
            "parse_status": parse_status,
            "response": record,
        }
    atomic_json(path, result)
    return result


def write_progress(
    root: Path,
    *,
    model: str,
    expected: int,
    results: list[Mapping[str, Any]],
) -> None:
    counts = Counter(str(row.get("transport_status")) for row in results)
    atomic_json(
        root / "PROGRESS.json",
        {
            "schema_version": "contextual-pass1-generation-progress-v1",
            "updated_at": utc_now(),
            "model": model,
            "expected": expected,
            "observed": len(results),
            "transport_status": dict(counts),
        },
    )


def main() -> int:
    args = parse_args()
    if args.workers < 1 or args.max_retries < 1:
        raise ExperimentError("workers and max retries must be positive")
    temperature = parse_temperature(args.temperature)
    manifest = read_json(args.transcript_manifest)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version")
        != "contextual-pass1-transcript-manifest-v1"
    ):
        raise ExperimentError("invalid transcript manifest")
    model_record = (manifest.get("models") or {}).get(args.model)
    if not isinstance(model_record, dict):
        raise ExperimentError(f"model not present in transcript manifest: {args.model}")
    transcript_path = Path(str(model_record["path"]))
    if sha256_file(transcript_path) != model_record.get("sha256"):
        raise ExperimentError("transcript file hash changed")
    rows = read_jsonl(transcript_path)
    if (
        len(rows) != 1312
        or any(row.get("model") != args.model for row in rows)
    ):
        raise ExperimentError("generation transcript census mismatch")
    cell_keys = {(row.get("method"), row.get("task_id")) for row in rows}
    if len(cell_keys) != len(rows):
        raise ExperimentError("duplicate generation cells")

    api_key = args.api_key_file.read_text(encoding="utf-8").strip()
    if not api_key:
        raise ExperimentError("API key file is empty")
    root = args.out_dir / model_slug(args.model)
    root.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        rows,
        key=lambda row: (
            task_sort_key(str(row["task_id"])),
            METHOD_ORDER.index(str(row["method"])),
        ),
    )
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                generate_one,
                args,
                row=row,
                api_key=api_key,
                temperature=temperature,
                root=root,
            ): row
            for row in ordered
        }
        for completed, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            if completed % 20 == 0 or completed == len(futures):
                write_progress(
                    root,
                    model=args.model,
                    expected=len(rows),
                    results=results,
                )
                print(
                    f"{args.model}: {completed}/{len(rows)} "
                    f"transport={Counter(r['transport_status'] for r in results)}",
                    flush=True,
                )

    results.sort(
        key=lambda row: (
            METHOD_ORDER.index(str(row["method"])),
            task_sort_key(str(row["task_id"])),
        )
    )
    failed = [row for row in results if row["transport_status"] != "ok"]
    if failed:
        write_progress(
            root,
            model=args.model,
            expected=len(rows),
            results=results,
        )
        print(
            f"{args.model}: {len(failed)} transport failures remain; rerun to recover",
            flush=True,
        )
        return 2

    results_path = root / "results.jsonl"
    atomic_jsonl(results_path, results)
    parse_counts = {
        method: dict(
            Counter(
                row["parse_status"]
                for row in results
                if row["method"] == method
            )
        )
        for method in METHOD_ORDER
    }
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "completed_at": utc_now(),
        "model": args.model,
        "rows": len(results),
        "tasks": len({row["task_id"] for row in results}),
        "methods": list(METHOD_ORDER),
        "generation_policy": {
            "one_successful_response_per_cell": True,
            "no_parse_or_semantic_resampling": True,
            "transport_retries_only": True,
            "workers": args.workers,
            "max_tokens": args.max_tokens,
            "reasoning_effort": args.reasoning_effort,
            "temperature": temperature,
            "base_url": args.base_url,
        },
        "parse_status_by_method": parse_counts,
        "inputs": {
            "transcript_manifest_path": str(args.transcript_manifest.resolve()),
            "transcript_manifest_sha256": sha256_file(
                args.transcript_manifest
            ),
            "transcript_path": str(transcript_path.resolve()),
            "transcript_sha256": sha256_file(transcript_path),
        },
        "results": {
            "path": str(results_path.resolve()),
            "sha256": sha256_file(results_path),
        },
    }
    summary["summary_identity"] = stable_hash(summary)
    atomic_json(root / "SUMMARY.json", summary)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
