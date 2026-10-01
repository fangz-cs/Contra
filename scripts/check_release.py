#!/usr/bin/env python3
"""Check the publication payload for data files and sensitive text patterns."""

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


SOURCE_SUFFIXES = {".py", ".md", ".toml", ".json", ".yaml", ".yml"}
SPECIAL_FILES = {"LICENSE", ".gitignore", ".env.example", "MANIFEST.in"}
JSON_FILES = {
    ".claude-plugin/marketplace.json",
    "plugin/.claude-plugin/plugin.json",
    "plugin/hooks/hooks.json",
    "plugin/hook_settings_eval.json",
}
GENERATED_PARTS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", "build", "dist",
}
FORBIDDEN_PARTS = {
    "data", "data_v3", "datasets", "inputs", "outputs", "results",
    "results_stub", "workspaces", "state", ".claude",
}
TEXT_PATTERNS = {
    "personal home path": re.compile(r"/(?:home|Users)/[A-Za-z0-9_.-]+"),
    "email address": re.compile(r"\b[A-Za-z0-9_.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "private network address": re.compile(
        r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
        r"|192\.168\.\d{1,3}\.\d{1,3}"
        r"|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b"
    ),
    "credential token": re.compile(
        r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}"
        r"|github_pat_[A-Za-z0-9_]{20,}"
        r"|sk-(?:proj-)?[A-Za-z0-9_-]{20,}"
        r"|AIza[A-Za-z0-9_-]{30,})\b"
    ),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "credential in URL": re.compile(r"https?://[^/\s:@]+:[^/\s@]+@"),
}


@dataclass(frozen=True)
class Finding:
    path: str
    category: str
    line: int | None = None


def publication_paths(root: Path) -> list[Path]:
    tracked = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--cached", "-z"],
        capture_output=True,
        check=False,
    )
    if tracked.returncode == 0:
        paths = [
            root / value.decode("utf-8")
            for value in tracked.stdout.split(b"\0")
            if value
        ]
        if not paths:
            raise ValueError("the Git publication index is empty")
        return sorted(paths)
    return sorted(
        path for path in root.rglob("*")
        if (path.is_file() or path.is_symlink())
        and not any(part in GENERATED_PARTS or part.endswith(".egg-info")
                    for part in path.relative_to(root).parts)
    )


def scan_paths(
    root: Path,
    paths: Sequence[Path],
    deny_tokens: Sequence[str] = (),
) -> list[Finding]:
    findings: list[Finding] = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        if any(token.casefold() in relative.casefold() for token in deny_tokens):
            findings.append(Finding(relative, "explicitly excluded filename"))
        if path.is_symlink():
            findings.append(Finding(relative, "symbolic link"))
            continue
        if any(part in FORBIDDEN_PARTS for part in Path(relative).parts):
            findings.append(Finding(relative, "data or private-state directory"))
        if path.name not in SPECIAL_FILES and path.suffix not in SOURCE_SUFFIXES:
            findings.append(Finding(relative, "non-source payload"))
        if path.suffix == ".json" and relative not in JSON_FILES:
            findings.append(Finding(relative, "unapproved JSON payload"))
        if path.stat().st_size > 250_000:
            findings.append(Finding(relative, "oversized payload"))
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings.append(Finding(relative, "binary payload"))
            continue
        for line_number, line in enumerate(text.splitlines(), 1):
            for category, pattern in TEXT_PATTERNS.items():
                if pattern.search(line):
                    findings.append(Finding(relative, category, line_number))
            if any(token.casefold() in line.casefold() for token in deny_tokens):
                findings.append(Finding(relative, "explicitly excluded identifier", line_number))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--deny-token", action="append", default=[])
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        parser.error("release root is not a directory")
    if any(not value.strip() for value in args.deny_token):
        parser.error("excluded identifiers must not be empty")
    paths = publication_paths(root)
    findings = scan_paths(root, paths, args.deny_token)
    for finding in findings:
        location = f"{finding.path}:{finding.line}" if finding.line else finding.path
        print(f"{location}: {finding.category}")
    if findings:
        print(f"Release audit failed: {len(findings)} finding(s).")
        return 1
    print(f"Release audit passed: {len(paths)} source/configuration files; no sensitive-text matches.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
