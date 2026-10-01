"""HTTP layer: retry classification, usage extraction, reasoning traces.

The single most expensive bug this file exists to prevent is retrying a
permanent failure. A rejected parameter that comes back 400 and is re-sent on
every call and every retry fails slowly, expensively, and without an obvious
error.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

MAX_RETRIES = 8
BACKOFF_CAP_S = 60.0

# Retry these. Anything else is permanent until proven otherwise.
TRANSIENT_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}
# Never retry these. Re-sending a rejected parameter cannot make it accepted.
PERMANENT_STATUS = {400, 401, 403, 404, 405, 413, 422}


class PermanentError(RuntimeError):
    def __init__(self, message: str, *, status: int | None, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body


class TransientError(RuntimeError):
    def __init__(self, message: str, *, status: int | None, body: str = "", retry_after: float | None = None):
        super().__init__(message)
        self.status = status
        self.body = body
        self.retry_after = retry_after


@dataclass
class Attempt:
    n: int
    error_type: str
    status: int | None
    message: str
    at: str


@dataclass
class CallResult:
    text: str
    finish_reason: str | None
    response_id: str | None
    usage: dict[str, Any]
    reasoning_text: str | None
    reasoning_opaque: str | None
    latency_s: float
    attempts: list[Attempt] = field(default_factory=list)
    raw: dict[str, Any] | None = None


def _status_of(exc: BaseException) -> int | None:
    for attr in ("status_code", "status", "http_status", "code"):
        v = getattr(exc, attr, None)
        if isinstance(v, int):
            return v
    resp = getattr(exc, "response", None)
    if resp is not None:
        v = getattr(resp, "status_code", None)
        if isinstance(v, int):
            return v
    return None


def _body_of(exc: BaseException) -> str:
    resp = getattr(exc, "response", None)
    if resp is not None:
        try:
            return resp.text[:4000]
        except Exception:
            pass
    return str(exc)[:4000]


def classify(exc: BaseException) -> PermanentError | TransientError:
    """Map any exception onto exactly one of the two buckets."""
    status = _status_of(exc)
    body = _body_of(exc)
    name = type(exc).__name__
    if status in PERMANENT_STATUS:
        return PermanentError(f"{name}: HTTP {status}", status=status, body=body)
    if status in TRANSIENT_STATUS:
        retry_after = None
        resp = getattr(exc, "response", None)
        if resp is not None:
            try:
                retry_after = float(resp.headers.get("retry-after"))
            except Exception:
                retry_after = None
        return TransientError(f"{name}: HTTP {status}", status=status, body=body, retry_after=retry_after)
    if status is not None and 400 <= status < 500:
        # An unknown 4xx is the client's fault. Do not pay to find out twice.
        return PermanentError(f"{name}: HTTP {status}", status=status, body=body)
    # Wrapper exceptions bury the status code in the message text. LiteLLM
    # raises ConversationRunError("... litellm.BadRequestError: OpenAIException
    # - \"auto\" tool choice requires --enable-auto-tool-choice ..."), which has
    # no status attribute at all -- so a hard 400 was retried 8 times against an
    # endpoint that could never accept the request. Read the text as a last
    # resort, but only for markers that are unambiguously permanent.
    lowered = f"{name}: {body}".lower()
    _PERMANENT_MARKERS = (
        "badrequesterror", "bad_request_error", "invalid_request_error",
        "authenticationerror", "permissiondeniederror", "notfounderror",
        "unsupportedparamserror", "http 400", "http 401", "http 403", "http 404",
    )
    if any(marker in lowered for marker in _PERMANENT_MARKERS):
        return PermanentError(f"{name}: {body[:200]}", status=status, body=body)

    # A bug in this process, not a hiccup on the wire. Retrying a TypeError
    # cannot make a wrong keyword argument right, and a misconfiguration that
    # looks transient is a misconfiguration that gets retried for a whole
    # campaign.
    if isinstance(exc, (TypeError, AttributeError, KeyError, IndexError,
                        NameError, ImportError, NotImplementedError,
                        AssertionError)):
        return PermanentError(f"{name}: {body[:200]}", status=status, body=body)
    # No status at all: connection reset, DNS failure, read timeout, malformed
    # body from a gateway. All worth another attempt.
    return TransientError(f"{name}: {body[:200]}", status=status, body=body)


def with_retries(fn: Callable[[], Any], *, max_retries: int = MAX_RETRIES,
                 on_attempt: Callable[[Attempt], None] | None = None) -> tuple[Any, list[Attempt]]:
    """Run ``fn``, retrying transient failures only."""
    attempts: list[Attempt] = []
    for n in range(1, max_retries + 1):
        try:
            return fn(), attempts
        except (PermanentError, TransientError):
            raise
        except BaseException as exc:  # noqa: BLE001
            err = classify(exc)
            at = Attempt(
                n=n,
                error_type=type(err).__name__,
                status=err.status,
                message=str(err)[:500],
                at=utcnow(),
            )
            attempts.append(at)
            if on_attempt:
                on_attempt(at)
            if isinstance(err, PermanentError):
                err.attempts = attempts  # type: ignore[attr-defined]
                raise err from exc
            if n == max_retries:
                err.attempts = attempts  # type: ignore[attr-defined]
                raise err from exc
            wait = min(BACKOFF_CAP_S, 2.0 ** (n - 1)) * (0.5 + random.random())
            if getattr(err, "retry_after", None):
                wait = max(wait, min(BACKOFF_CAP_S, err.retry_after))
            time.sleep(wait)
    raise AssertionError("unreachable")


def utcnow() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def extract_usage(usage_obj: Any) -> dict[str, Any]:
    """Normalise a usage payload, reading reasoning tokens from BOTH locations.

    Gateways disagree with the spec: the OpenAI spec puts reasoning tokens under
    ``completion_tokens_details``, while some proxies put them at the top level
    of ``usage``. Reading only the spec location silently records ``None`` on
    every row of an entire campaign while the model is in fact spending
    thousands of reasoning tokens per call.
    """
    if usage_obj is None:
        return {}
    u = usage_obj if isinstance(usage_obj, dict) else _to_dict(usage_obj)
    ctd = u.get("completion_tokens_details") or {}
    if not isinstance(ctd, dict):
        ctd = _to_dict(ctd)
    ptd = u.get("prompt_tokens_details") or {}
    if not isinstance(ptd, dict):
        ptd = _to_dict(ptd)

    reasoning = ctd.get("reasoning_tokens")
    if reasoning is None:
        reasoning = u.get("reasoning_tokens")

    cached = ptd.get("cached_tokens")
    if cached is None:
        cached = u.get("cached_tokens")

    return {
        "prompt_tokens": u.get("prompt_tokens"),
        "completion_tokens": u.get("completion_tokens"),
        "total_tokens": u.get("total_tokens"),
        "reasoning_tokens": reasoning,
        "cached_tokens": cached,
        "raw": u,
    }


def extract_reasoning(message: Any) -> tuple[str | None, str | None]:
    """Pull a reasoning trace off a message, wherever the gateway parked it.

    Returns ``(text, opaque)``. A ``None`` trace means the gateway did not
    expose one -- never that the model did not think. The token count in
    ``usage`` is the authority on that.
    """
    if message is None:
        return None, None
    extra = getattr(message, "model_extra", None) or {}
    if not isinstance(extra, dict):
        extra = {}
    d = extra if extra else (message if isinstance(message, dict) else _to_dict(message))
    text = d.get("reasoning_text") or d.get("reasoning_content") or d.get("reasoning")
    opaque = d.get("reasoning_opaque") or d.get("encrypted_reasoning")
    if isinstance(text, list):  # Anthropic-style content blocks
        parts = [b.get("thinking") for b in text if isinstance(b, dict) and b.get("type") == "thinking"]
        text = "\n".join(p for p in parts if p) or None
    return (text if isinstance(text, str) else None,
            opaque if isinstance(opaque, str) else None)


def _to_dict(obj: Any) -> dict[str, Any]:
    for attr in ("model_dump", "to_dict", "dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass
    return dict(getattr(obj, "__dict__", {}) or {})
