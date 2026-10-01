"""Durable result storage: fsynced write-through, torn-line-tolerant loading,
and a hard wall between stub and real output.

Three invariants live here, and every read path in the campaign goes through
this module so none of them can be bypassed:

R2  one finished cell -> one complete JSON line -> flush() + fsync(), written by
    a single writer thread fed from a queue.
R3  resume is the default: a cell counts as done only when its status is
    terminal (``ok`` or ``failed_permanent``).
R8  stub rows and real rows never share a directory, AND every reader checks the
    provenance tag and refuses to mix. Tagging alone is not a defence.
"""
from __future__ import annotations

import errno
import fcntl
import json
import os
import queue
import threading
from pathlib import Path
from typing import Any, Iterator

SCHEMA_VERSION = 1

REAL_ROOT = Path("results/harness")
STUB_ROOT = Path("results_stub/harness")


class ProvenanceError(RuntimeError):
    """Raised when stub and real rows would be mixed."""


def root_for(stub: bool) -> Path:
    """Physical separation is the default, not a flag the operator must set."""
    return STUB_ROOT if stub else REAL_ROOT


def cell_key(model: str, arm: str, task_id: str, sample: int) -> str:
    return f"{model}||{arm}||{task_id}||{sample}"


def rows_path(run_dir: Path, model: str, arm: str) -> Path:
    return run_dir / "rows" / f"{model}__{arm}.jsonl"


def iter_rows(path: Path) -> Iterator[tuple[dict[str, Any] | None, str]]:
    """Yield ``(row, raw_line)``; ``row`` is None for an unparseable line.

    A kill -9 mid-write leaves a torn final line. A loader that raises on it
    turns a one-cell loss into a total loss, so bad lines are reported, not
    fatal.
    """
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line), line
            except json.JSONDecodeError:
                yield None, line


def load_done(path: Path, *, stub: bool, log=print) -> set[str]:
    """Cell keys already in a terminal state, with the provenance guard applied."""
    done: set[str] = set()
    torn = 0
    for row, raw in iter_rows(path):
        if row is None:
            torn += 1
            continue
        assert_provenance(row, path, stub=stub)
        if row.get("status") in ("ok", "failed_permanent"):
            done.add(row["cell_key"])
    if torn:
        log(f"  discarded {torn} unparseable line(s) in {path}; those cells will re-run")
    return done


def assert_provenance(row: dict[str, Any], path: Path, *, stub: bool) -> None:
    """Refuse to read a fabricated row as if it were real, and vice versa.

    This is the check that R8 says must exist on *every* read path. Without it a
    free smoke test can convince a real run that the whole campaign is done.
    """
    row_stub = bool(row.get("stub", False))
    if row_stub != stub:
        kind = "stub" if row_stub else "real"
        want = "stub" if stub else "real"
        raise ProvenanceError(
            f"{path}: found a {kind} row (cell {row.get('cell_key')!r}) while running in "
            f"{want} mode.\n"
            f"  Stub output belongs in {STUB_ROOT}/, real output in {REAL_ROOT}/.\n"
            f"  Fix: delete or move the offending file; do not merge the two."
        )


def load_rows(path: Path, *, stub: bool) -> list[dict[str, Any]]:
    """Read rows for analysis, with the same guard the runner uses."""
    out = []
    for row, _ in iter_rows(path):
        if row is None:
            continue
        assert_provenance(row, path, stub=stub)
        out.append(row)
    return out


class SingleWriterLock:
    """An exclusive flock on the run directory.

    The writer thread keeps lines intact within one process and does nothing
    about a second process. Two copies of the same runner appending to one file
    is how tasks get paid for twice.
    """

    def __init__(self, run_dir: Path):
        self.path = run_dir / ".writer.lock"
        self._fd: int | None = None

    def __enter__(self) -> "SingleWriterLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(self._fd)
            self._fd = None
            if exc.errno in (errno.EAGAIN, errno.EACCES):
                raise SystemExit(
                    f"Another runner already holds {self.path}. Refusing to be the "
                    f"second writer: two processes appending to one file interleave "
                    f"partial lines and pay for the same cells twice."
                ) from exc
            raise
        os.write(self._fd, f"{os.getpid()}\n".encode())
        return self

    def __exit__(self, *exc_info) -> None:
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None


class Writer:
    """One writer thread per process, fed by a queue, fsyncing every line."""

    _SENTINEL = object()

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self._q: queue.Queue = queue.Queue()
        self._handles: dict[Path, Any] = {}
        self._thread = threading.Thread(target=self._drain, name="writer", daemon=False)
        self._written = 0

    def start(self) -> "Writer":
        (self.run_dir / "rows").mkdir(parents=True, exist_ok=True)
        self._thread.start()
        return self

    def put(self, path: Path, obj: dict[str, Any]) -> None:
        self._q.put((path, obj))

    def close(self) -> None:
        self._q.put(self._SENTINEL)
        self._thread.join()

    @property
    def written(self) -> int:
        return self._written

    def _handle(self, path: Path):
        h = self._handles.get(path)
        if h is None:
            path.parent.mkdir(parents=True, exist_ok=True)
            # A kill -9 mid-write leaves a final line with no newline. Appending
            # straight onto it would glue the next row to the torn one, so the
            # torn line would cost two cells instead of one. Terminate it first.
            needs_nl = False
            if path.exists() and path.stat().st_size:
                with path.open("rb") as probe:
                    probe.seek(-1, os.SEEK_END)
                    needs_nl = probe.read(1) != b"\n"
            h = path.open("a", encoding="utf-8")
            if needs_nl:
                h.write("\n")
                h.flush()
                os.fsync(h.fileno())
            self._handles[path] = h
        return h

    def _drain(self) -> None:
        while True:
            item = self._q.get()
            if item is self._SENTINEL:
                break
            path, obj = item
            h = self._handle(path)
            h.write(json.dumps(obj, ensure_ascii=False) + "\n")
            h.flush()
            os.fsync(h.fileno())   # the one that actually matters
            self._written += 1
        for h in self._handles.values():
            h.flush()
            os.fsync(h.fileno())
            h.close()


def write_json_atomic(path: Path, obj: dict[str, Any]) -> None:
    """For manifest.json, which is rewritten in place as spend accumulates."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
