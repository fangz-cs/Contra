#!/usr/bin/env python3
"""Confirm a disclosed assumption by running a test that separates the readings.

The earlier verifier compared return values, so anything a function did other
than return -- warnings raised, descriptors leaked, work done at import time,
state left on an app object -- looked to it like no difference at all. It
reported "identical behaviour" for changes that were plainly not identical, and
on this repository it reached only four of twenty-six disclosures.

A test is a general observer. Whatever surface the difference shows up on,
pytest can already assert against it, and the assistant writing the test knows
which surface that is. So the question becomes: write a test that passes on the
implemented reading; does it still pass on the rival one? A test that fails
there IS the evidence, and it is evidence the user can keep.

Reads one JSON object on stdin:
  {"root": "/repo", "file": "pkg/mod.py",
   "test_code": "def test_x(): ...",           # asserts the implemented reading
   "rival_edits": [{"old": "...", "new": "..."}]}   # the other reading, as a patch

The rival is a patch rather than a replacement function because a change is not
always inside a function: a default constant, an import, a class attribute can
each carry a decision, and often the most consequential one.

Prints a verdict. Never claims equivalence it cannot show: a rival that also
passes is reported as "not separated", not as "no difference".
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hooks"))
import config  # noqa: E402
import exec_patch  # noqa: E402

TIMEOUT = config.get("test_timeout")
BWRAP = "/usr/bin/bwrap"
# pytest's own exit codes, which carry the distinction this file depends on.
# Anything outside these two means the run did not happen: a collection error,
# an import that failed, an interrupted session. Treating those as "the test
# failed" is how a rival patch that merely breaks the file gets reported as a
# behavioural difference.
PYTEST_PASSED, PYTEST_FAILED = 0, 1


def _sandbox_roots() -> list[str]:
    """Where the interpreter's own libraries live, and nothing else.

    Binding every `sys.path` entry was the obvious way to make third-party
    imports work, and it silently mounted the repository: this script inserts
    its own directory on `sys.path`, and a project whose cwd or PYTHONPATH is on
    it would follow. A test written by a model could then read the very source
    the sandbox exists to keep out of its reach.

    So only the interpreter prefixes, plus the `sys.path` entries that live
    underneath them -- that covers an ordinary pip install. An editable install
    pointing outside the prefix does not come in, and an import that needs it
    fails; the caller reports that as unverifiable, which is visible. The
    project's own code needs none of this: the copy is mounted at /work and
    pytest puts its rootdir on the path.
    """
    prefixes = []
    for base in (sys.prefix, sys.base_prefix, os.path.dirname(sys.executable)):
        real = os.path.realpath(base) if base else ""
        if real and os.path.isdir(real) and real != "/":
            prefixes.append(real)

    out: list[str] = []
    for r in sorted(set(prefixes) | {
            os.path.realpath(p) for p in sys.path
            if p and os.path.isdir(p)
            and any(os.path.realpath(p).startswith(x + os.sep) for x in prefixes)}):
        if not any(r != o and r.startswith(o + os.sep) for o in out):
            out.append(r)
    return out


def _sandboxed(repo: Path, test_name: str) -> list[str] | None:
    """The bwrap invocation, or None when bubblewrap is not installed."""
    if not (os.path.isfile(BWRAP) and os.access(BWRAP, os.X_OK)):
        return None
    cmd = [BWRAP, "--unshare-all", "--die-with-parent", "--new-session",
           "--ro-bind", "/usr", "/usr", "--ro-bind-try", "/lib", "/lib",
           "--ro-bind-try", "/lib64", "/lib64", "--ro-bind-try", "/bin", "/bin",
           "--ro-bind-try", "/etc", "/etc",
           "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    for root in _sandbox_roots():
        cmd += ["--ro-bind-try", root, root]
    # The copy is writable: a test may legitimately create files next to itself.
    # The real tree is not mounted at all, and --unshare-all takes the network.
    cmd += ["--bind", str(repo), "/work", "--chdir", "/work",
            "--setenv", "HOME", "/work",
            "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
            sys.executable, "-m", "pytest", test_name, "-x", "-q",
            "-p", "no:cacheprovider"]
    return cmd


def run_pytest(repo: Path, test_file: Path) -> tuple[str, str]:
    """Run the test. Returns one of "pass", "fail", "error", plus the tail.

    This executes code a model just wrote, so it runs with no network and no
    view of the real tree. Without bubblewrap it does not run at all: the
    caller reports that as unverifiable, which is visible, rather than
    executing it on the host, which is not.
    """
    name = test_file.name
    cmd = _sandboxed(repo, name)
    if cmd is None:
        if config.get("sandbox") != "allow_host":
            return "error", (f"{BWRAP} not found; install bubblewrap, or set "
                             "sandbox=allow_host to accept running "
                             "model-written tests on the host")
        cmd = [sys.executable, "-m", "pytest", name, "-x", "-q",
               "-p", "no:cacheprovider"]
    try:
        proc = subprocess.run(cmd, cwd=str(repo), capture_output=True,
                              text=True, errors="replace", timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return "error", f"the test did not finish within {TIMEOUT}s"
    except OSError as exc:
        return "error", f"could not start the test run: {exc}"
    tail = (proc.stdout + proc.stderr)[-1500:]
    if proc.returncode == PYTEST_PASSED:
        return "pass", tail
    if proc.returncode == PYTEST_FAILED:
        return "fail", tail
    return "error", tail


def main() -> int:
    try:
        spec = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError) as exc:
        print(json.dumps({"verdict": "unverifiable",
                          "why": f"could not read the request: {exc}"[:200]}))
        return 2
    root = Path(spec["root"]).resolve()
    target = root / spec["file"]
    if not target.is_file():
        print(json.dumps({"verdict": "unverifiable", "why": "file not found"}))
        return 2

    work = Path(tempfile.mkdtemp(prefix="contra_disc_"))
    try:
        # A copy, so the rival implementation never touches the real tree.
        mirror = work / root.name
        shutil.copytree(root, mirror, symlinks=True,
                        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"))
        test_file = mirror / "test_contra_discriminating.py"
        test_file.write_text(spec["test_code"], encoding="utf-8")

        now, out_now = run_pytest(mirror, test_file)
        if now == "error":
            return _emit({"verdict": "unverifiable",
                          "why": "the test could not be run at all",
                          "detail": out_now})
        if now != "pass":
            # The test does not describe what was actually built, so it cannot
            # tell us anything about the rival either.
            return _emit({"verdict": "unverifiable",
                          "why": "the test does not pass on the implemented reading",
                          "detail": out_now})

        source = (mirror / spec["file"]).read_text(encoding="utf-8")
        edits = spec.get("rival_edits")
        try:
            if edits:
                patched = exec_patch.apply_edits(source, edits)
            else:
                # Older callers still send a whole replacement function.
                patched = exec_patch.splice(source, spec["function"], spec["rival_impl"])
        except Exception as exc:  # noqa: BLE001
            return _emit({"verdict": "unverifiable",
                          "why": f"could not apply the rival reading: {exc}"})
        (mirror / spec["file"]).write_text(patched, encoding="utf-8")

        rival, out_rival = run_pytest(mirror, test_file)
        if rival == "fail":
            return _emit({"verdict": "confirmed",
                          "evidence": "the test fails on the rival reading",
                          "test_code": spec["test_code"],
                          "detail": out_rival})
        if rival == "error":
            # The rival patch broke the file rather than changing its
            # behaviour -- a syntax error, a name that no longer resolves.
            # pytest cannot collect, and its exit code is not 1. Reading that
            # as "the test failed" would report a wrecked file as evidence
            # that the two readings differ, which is a false positive with a
            # green tick next to it.
            return _emit({"verdict": "unverifiable",
                          "why": "the rival reading does not run, so the test "
                                 "says nothing about how it would behave",
                          "detail": out_rival})
        return _emit({"verdict": "not_separated",
                      "why": "the test passes on both readings; it does not "
                             "capture whatever distinguishes them",
                      "test_code": spec["test_code"]})
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _emit(payload: dict) -> int:
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
