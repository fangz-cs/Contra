#!/usr/bin/env python3
"""Everything a user might reasonably want to change, in one place.

Three sources, in order: the environment, then a config file, then the default.
The environment wins because `settings.json` can set it per project, while the
file is one answer for the whole machine.

A setting that arrives unreadable -- `CONTRA_MAX_ITEMS=lots`, a config file with
a stray comma -- falls back to the default and says so on stderr. It must never
silently become zero or empty: every defect this tool has had degraded into
reporting less, and a misread setting is the cheapest possible way to do that
again.
"""
from __future__ import annotations

import json
import locale
import os
import sys
from pathlib import Path

# The value is also the type. A default of None means "leave it to whatever
# runs underneath" -- for a model, the CLI's own choice; for a language, the
# locale.
DEFAULTS: dict[str, object] = {
    # Presentation
    "lang": None,               # None -> follow the locale; "en" or "zh"
    "max_items": 8,             # how many entries the view may hold

    # Which model answers which call. None means the `claude` CLI's default.
    # Measured on one turn: the review call at Sonnet took 3.6x the latency of
    # Opus for 40% of the price, and it sits on the path the reader waits on.
    "extract_model": "haiku",
    "probe_model": None,
    "build_model": None,

    # Seconds
    "timeout_extract": 180,
    "timeout_build": 300,
    "test_timeout": 300,

    # What counts as a change worth reading. Reporting spans all of these;
    # only `runnable_suffixes` can be settled by running a test.
    "source_suffixes": [
        ".py", ".pyi", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx",
        ".go", ".rs", ".rb", ".java", ".kt", ".swift",
        ".c", ".h", ".cc", ".cpp", ".hpp", ".cs", ".php", ".scala", ".sh",
    ],
    "runnable_suffixes": [".py"],
    # Skip a Python change whose syntax tree is unchanged: comments, blank
    # lines, quote style. Turning this off makes every reformat cost a turn.
    "skip_unchanged_ast": True,

    # Character-bigram overlap above which two entries are one decision.
    # Deliberately conservative: merging two different decisions destroys one
    # silently, while failing to merge shows the reader one thing twice.
    "merge_threshold": 0.28,

    # "required" refuses to run model-written tests without bubblewrap;
    # "allow_host" accepts running them with your permissions.
    "sandbox": "required",

    # Baseline ceilings, so a huge tree cannot stall a turn.
    "max_files": 2000,
    "max_bytes": 512 * 1024,
}

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}


def _config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME")
    return (Path(base) if base else Path.home() / ".config") / "contra" / "config.json"


def _warn(msg: str) -> None:
    # stderr, not the view: a bad setting is the operator's problem, and the
    # view belongs to the person reading about their code.
    print(f"contra: {msg}", file=sys.stderr)


def _load_file() -> dict:
    path = Path(os.environ.get("CONTRA_CONFIG") or _config_path())
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError, OSError) as exc:
        _warn(f"ignoring {path}: {exc}")
        return {}
    if not isinstance(loaded, dict):
        _warn(f"ignoring {path}: expected an object")
        return {}
    for key in loaded:
        if key not in DEFAULTS:
            _warn(f"unknown setting in {path}: {key}")
    return loaded


_FILE = _load_file()


def _coerce(key: str, raw: object, default: object, source: str):
    """Turn a string from the environment into the default's type."""
    if not isinstance(raw, str):
        return raw
    text = raw.strip()
    try:
        if isinstance(default, bool):
            low = text.lower()
            if low in _TRUE:
                return True
            if low in _FALSE:
                return False
            raise ValueError("expected a boolean")
        if isinstance(default, int):
            return int(text)
        if isinstance(default, float):
            return float(text)
        if isinstance(default, list):
            items = [p.strip() for p in text.replace(",", " ").split()]
            if not items:
                raise ValueError("empty list")
            return items
    except (TypeError, ValueError) as exc:
        _warn(f"{source} {key}={raw!r} is not usable ({exc}); using {default!r}")
        return default
    return text


def get(key: str):
    """The effective value of one setting."""
    default = DEFAULTS[key]
    env = os.environ.get("CONTRA_" + key.upper())
    if env is not None:
        return _coerce(key, env, default, "environment")
    if key in _FILE:
        value = _FILE[key]
        if default is not None and value is not None and not isinstance(
                value, type(default)) and not (
                isinstance(default, float) and isinstance(value, int)):
            _warn(f"config {key}={value!r} is not a {type(default).__name__}; "
                  f"using {default!r}")
            return default
        return value
    return default


def language() -> str:
    """"en" or "zh". An explicit setting wins; otherwise the locale decides."""
    chosen = get("lang")
    if chosen:
        low = str(chosen).lower()
        if low.startswith("zh"):
            return "zh"
        if low.startswith("en"):
            return "en"
        _warn(f"lang={chosen!r} is not supported; using en")
        return "en"
    for var in ("LC_ALL", "LC_MESSAGES", "LANG"):
        value = os.environ.get(var)
        if value:
            return "zh" if value.lower().startswith("zh") else "en"
    try:
        code = locale.getlocale()[0] or ""
    except (ValueError, TypeError):
        code = ""
    return "zh" if code.lower().startswith("zh") else "en"


def suffixes(key: str) -> tuple[str, ...]:
    """Normalised, so `py,js` and `.py .js` both work."""
    raw = get(key)
    out = []
    for item in raw if isinstance(raw, (list, tuple)) else [raw]:
        text = str(item).strip()
        if text:
            out.append(text if text.startswith(".") else "." + text)
    return tuple(out) or tuple(DEFAULTS[key])


def describe() -> str:
    """Every setting and where its value came from. For `--config`."""
    rows = []
    for key, default in DEFAULTS.items():
        value = get(key)
        if os.environ.get("CONTRA_" + key.upper()) is not None:
            src = "env"
        elif key in _FILE:
            src = "file"
        else:
            src = "default"
        rows.append(f"  {key:<20} {json.dumps(value, ensure_ascii=False):<42} {src}")
    return (f"config file: {os.environ.get('CONTRA_CONFIG') or _config_path()}\n"
            + "\n".join(rows))


if __name__ == "__main__":
    print(describe())
