#!/usr/bin/env python3
"""What this turn changed, as opposed to what the working tree happens to hold.

The hook used to ask git for `git diff -- '*.py'` and treat the answer as the
assistant's work. Three things are wrong with that, and all three show up on the
first real repository:

  * `git diff` omits staged changes and, worse, omits new files entirely -- so a
    file the assistant had just written was invisible to the reviewer;
  * it is measured against HEAD, so uncommitted work from last week is re-read
    as this turn's decisions, once per turn, forever;
  * it cannot tell a turn that changed code from one that only answered a
    question, so every prompt paid the full cost.

They are one problem: the hook had no notion of *this turn*. So a
UserPromptSubmit hook records the tree as it stood when the prompt arrived, and
Stop diffs against that.

The snapshot is cheap because most files are unchanged and git already stores
them: for those we keep the blob hash and read the content back out of the
object store. Only files git cannot hand back -- modified, staged, untracked --
carry their text inline.
"""
from __future__ import annotations

import ast
import difflib
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

# Reporting what a change decided needs only the diff, so it is not tied to a
# language. Running a test to settle one is -- but that is a third of the
# output; the two thirds that report work beyond the request, or a scope line
# drawn silently, are language-agnostic and include the strongest signal there
# is (work beyond the request separates underspecified requests from settled
# ones by 6.5x).
SOURCE_SUFFIXES = config.suffixes("source_suffixes")
MAX_FILES = config.get("max_files")
MAX_BYTES = config.get("max_bytes")
# Only reached outside a git repository, where there is no object store to hand
# the contents to. Without a ceiling a large tree would write its whole source
# into the state file on every prompt.
MAX_INLINE_TOTAL = 2 * 1024 * 1024
def _default_state_dir() -> Path:
    """Somewhere that survives a reboot.

    `/tmp` was the first choice and it was wrong: `verifications.jsonl` is the
    running ledger, not scratch, and a machine that restarts takes the whole
    history with it. That has already happened here and cost a day's results.
    """
    base = os.environ.get("XDG_STATE_HOME")
    return (Path(base) if base else Path.home() / ".local" / "state") / "contra"


STATE_DIR = Path(os.environ.get("CONTRA_STATE_DIR") or _default_state_dir())


def baseline_path(session: str) -> Path:
    return STATE_DIR / (session + ".baseline.json")


def _git(cwd: Path, *args: str, stdin: str | None = None) -> str:
    """Git's stdout, or "" if it could not be asked. Callers that cannot tell
    those two apart should use `_git_ok` instead."""
    return _git_ok(cwd, *args, stdin=stdin) or ""


def _git_ok(cwd: Path, *args: str, stdin: str | None = None) -> str | None:
    """None means git could not answer -- not that the answer was empty."""
    try:
        done = subprocess.run(["git", *args], cwd=str(cwd), input=stdin,
                              capture_output=True, text=True,
                              errors="replace", timeout=30)
    except Exception:
        return None
    return done.stdout if done.returncode == 0 else None


def source_files(cwd: Path) -> list[Path]:
    """Every source file that counts, asking git first so .gitignore is honoured."""
    listed = _git(cwd, "ls-files", "--cached", "--others",
                  "--exclude-standard", "-z", "--",
                  *(f"*{s}" for s in SOURCE_SUFFIXES))
    if listed:
        rels = [r for r in listed.split("\0") if r]
    else:
        skip = {".git", "__pycache__", ".venv", "venv", "node_modules",
                ".mypy_cache", ".pytest_cache", ".tox"}
        rels = [str(p.relative_to(cwd)) for p in cwd.rglob("*")
                if p.suffix in SOURCE_SUFFIXES
                and not skip & set(p.relative_to(cwd).parts)]
    out = []
    for rel in sorted(rels)[:MAX_FILES]:
        p = cwd / rel
        if p.is_file():
            out.append(p)
    return out


def blob_sha(data: bytes) -> str:
    """The name git would give this content, computed without calling git."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _known_blobs(cwd: Path, shas: list[str]) -> set[str]:
    """Which of these are already in the object store, in one call."""
    if not shas:
        return set()
    out = _git(cwd, "cat-file", "--batch-check=%(objectname) %(objecttype)",
               stdin="\n".join(shas) + "\n")
    known = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == "blob":
            known.add(parts[0])
    return known


def _read(p: Path) -> str | None:
    try:
        if p.stat().st_size > MAX_BYTES:
            return None
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _store_blobs(cwd: Path, rels: list[str]) -> dict[str, str]:
    """Hand contents to git so the snapshot can be a list of hashes.

    A repository with many untracked Python files -- a research tree, a
    scratch directory -- would otherwise make every prompt write megabytes of
    inlined source. Git already solves this: it compresses, it deduplicates,
    and writing the same content twice produces the same object, so a file
    that does not change across turns is stored once however many turns pass.

    This writes loose objects and nothing else: no index, no refs, no working
    tree. They are unreachable, so `git gc` collects them on its own schedule,
    exactly as it does for the objects `git stash` leaves behind.
    """
    if not rels:
        return {}
    out = _git(cwd, "hash-object", "-w", "--stdin-paths",
               stdin="\n".join(rels) + "\n")
    shas = out.split()
    if len(shas) != len(rels):
        return {}          # no git, a read-only .git, anything: fall back
    return dict(zip(rels, shas))


def snapshot(cwd: Path) -> dict:
    """The tree as it stands, recorded so a later turn can diff against it."""
    seen: dict[str, str] = {}
    for p in source_files(cwd):
        text = _read(p)
        if text is None:
            continue
        seen[str(p.relative_to(cwd))] = blob_sha(text.encode("utf-8"))

    known = _known_blobs(cwd, list(seen.values()))
    absent = [rel for rel, sha in seen.items() if sha not in known]
    stored = _store_blobs(cwd, absent)

    files: dict[str, dict] = {}
    inlined = 0
    dropped = 0
    for rel, sha in seen.items():
        if sha in known or stored.get(rel):
            files[rel] = {"blob": stored.get(rel, sha)}
            continue
        text = _read(cwd / rel) or ""
        if inlined + len(text) > MAX_INLINE_TOTAL:
            # Recording the hash alone would let the diff say a file changed
            # without saying how, which is worse than saying nothing: the
            # reviewer would be handed a gap it cannot read.
            dropped += 1
            continue
        files[rel] = {"text": text}
        inlined += len(text)
    out: dict = {"files": files}
    if dropped:
        out["dropped"] = dropped
    return out


def _same_behaviour(before: str, after: str) -> bool:
    """Whether the two texts can only differ in ways no execution can see.

    A comment, a blank line, a reformatting pass: the parse tree is identical,
    so nothing the code does can have changed, and there is no decision in the
    change to report. Comparing the trees is exact and free -- far better than
    guessing from the diff which lines "look like" comments.

    Docstrings ARE in the tree, so rewriting one still counts as a change. That
    is the right side to err on: a docstring states a contract.
    """
    try:
        return ast.dump(ast.parse(before)) == ast.dump(ast.parse(after))
    except (SyntaxError, ValueError, RecursionError):
        return False        # cannot prove they match, so assume they differ


def _before(cwd: Path, entry: dict) -> str | None:
    """The file as the snapshot recorded it, or None if it cannot be recovered.

    The distinction matters: an unreadable blob returned as "" makes the whole
    file look newly added, and the reviewer is handed a diff full of decisions
    this turn never made. A blob can genuinely go missing -- `git gc` collects
    unreachable objects, and these are unreachable by design.
    """
    if "text" in entry:
        return entry["text"]
    blob = entry.get("blob")
    if not blob:
        return None
    return _git_ok(cwd, "cat-file", "-p", blob)


def turn_diff(cwd: Path, snap: dict | None) -> str:
    """A unified diff of what changed since the snapshot, in git's own format.

    With no snapshot -- the turn began before the hook was installed, or the
    file was lost -- this returns nothing rather than falling back to the whole
    working tree. A silent fallback is how the old behaviour would creep back.
    """
    if not snap:
        return ""
    before_files: dict[str, dict] = snap.get("files") or {}
    now = {str(p.relative_to(cwd)): p for p in source_files(cwd)}

    chunks = []
    for rel in sorted(set(before_files) | set(now)):
        old = _before(cwd, before_files[rel]) if rel in before_files else ""
        new = _read(now[rel]) if rel in now else ""
        if old is None:
            # Say nothing about a file whose earlier state is unrecoverable.
            # Guessing "" here would report the whole file as this turn's work.
            continue
        if new is None or old == new:
            continue
        if (rel.endswith(".py") and config.get("skip_unchanged_ast")
                and _same_behaviour(old, new)):
            # Formatting only. Reporting it would cost the reader a hundred
            # seconds to be told nothing happened.
            continue
        chunks.append("".join(difflib.unified_diff(
            old.splitlines(keepends=True), new.splitlines(keepends=True),
            fromfile=f"a/{rel}" if rel in before_files else "/dev/null",
            tofile=f"b/{rel}" if rel in now else "/dev/null")))
    return "\n".join(c for c in chunks if c.strip())


def load(session: str) -> dict | None:
    try:
        return json.loads(baseline_path(session).read_text(encoding="utf-8"))
    except Exception:
        return None


def sweep(older_than: float = 86400) -> None:
    """Drop state from sessions that have long since ended.

    A baseline is written on every prompt and only ever overwritten by the same
    session, so without this the directory grows for as long as the plugin is
    installed. Verification results are kept: they are the record.
    """
    import time
    cutoff = time.time() - older_than
    for pattern in ("*.baseline.json", "*.pending.json", "*.part", "*.done"):
        for f in STATE_DIR.glob(pattern):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
            except OSError:
                pass


def save(session: str, data: dict) -> None:
    """Atomic, because Stop may read this while UserPromptSubmit still writes."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    final = baseline_path(session)
    tmp = final.with_suffix(".part")
    with tmp.open("w", encoding="utf-8") as h:
        json.dump(data, h, ensure_ascii=False)
        h.flush(); os.fsync(h.fileno())
    tmp.replace(final)
    sweep()
