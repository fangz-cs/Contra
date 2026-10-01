"""The verifier executes code a model just wrote. These are its guarantees.

Each escape probe is a test that PASSES if the sandbox let it out. A blocked
probe therefore fails on the implemented reading, and the verifier reports
`unverifiable` -- so `unverifiable` here means the wall held.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

DISC = Path(__file__).resolve().parents[1] / "scripts" / "discriminate.py"
BWRAP = "/usr/bin/bwrap"
needs_bwrap = pytest.mark.skipif(
    not (os.path.isfile(BWRAP) and os.access(BWRAP, os.X_OK)),
    reason="bubblewrap not installed")

MODULE = '''def clamp(n, limit=512):
    if n > limit:
        return limit, True
    return n, False
'''
RIVAL = [{"old": "    if n > limit:", "new": "    if n >= limit:"}]
SEPARATES = ("from pkg.mod import clamp\n\n"
             "def test_the_boundary_is_not_clamped():\n"
             "    assert clamp(512) == (512, False)\n")


@pytest.fixture
def repo(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "mod.py").write_text(MODULE)
    return tmp_path


def judge(repo: Path, test_code: str, rival=None) -> dict:
    spec = {"root": str(repo), "file": "pkg/mod.py", "function": "clamp",
            "test_code": test_code, "rival_edits": rival or RIVAL}
    proc = subprocess.run([sys.executable, str(DISC)], input=json.dumps(spec),
                          capture_output=True, text=True, timeout=600)
    return json.loads(proc.stdout)


@needs_bwrap
def test_a_separating_test_is_confirmed(repo):
    assert judge(repo, SEPARATES)["verdict"] == "confirmed"


@needs_bwrap
def test_a_test_that_passes_both_ways_is_not_confirmed(repo):
    both = ("from pkg.mod import clamp\n\n"
            "def test_large_input_clamps():\n"
            "    assert clamp(9999)[0] == 512\n")
    assert judge(repo, both)["verdict"] == "not_separated"


@needs_bwrap
def test_a_rival_that_cannot_run_is_not_evidence(repo):
    """The false positive this exit-code split exists to stop.

    A rival patch that breaks the file makes pytest exit 2, not 1. Read as
    "the test failed", a wrecked file becomes a confirmed behavioural
    difference -- wrong, and displayed with a tick beside it.
    """
    broken = [{"old": "    if n > limit:", "new": "    if n > limit and:"}]
    v = judge(repo, SEPARATES, rival=broken)
    assert v["verdict"] == "unverifiable"
    assert "does not run" in v["why"]


# An interpreter installed under $HOME (conda, pyenv) makes bwrap create the
# home directory to mount its prefix, so `home` exists inside the sandbox and
# lists exactly that one entry. Probe a path that must not be reachable, not
# the directory that has to.
@needs_bwrap
@pytest.mark.parametrize("name,body", [
    ("network", "import socket\n\ndef test_out():\n"
                "    socket.create_connection(('1.1.1.1', 53), timeout=5).close()\n"),
    ("dotfiles", "import pathlib\n\ndef test_out():\n"
                  f"    assert pathlib.Path({str(Path.home() / '.claude')!r}).is_dir()\n"),
])
def test_the_sandbox_holds(repo, name, body):
    v = judge(repo, body)
    assert v["verdict"] == "unverifiable", f"{name}: the probe got out"
    assert "does not pass on the implemented reading" in v["why"]


@needs_bwrap
def test_the_real_tree_is_not_reachable(repo):
    """The copy is what runs; the working tree is not mounted at all."""
    outside = DISC.parent / "rank.py"
    body = (f"import pathlib\n\ndef test_out():\n"
            f"    assert pathlib.Path({str(outside)!r}).exists()\n")
    assert judge(repo, body)["verdict"] == "unverifiable"


def test_without_bubblewrap_it_refuses_rather_than_running_on_the_host(repo, monkeypatch):
    sys.path.insert(0, str(DISC.parent))
    import discriminate
    monkeypatch.setattr(discriminate, "BWRAP", "/nonexistent/bwrap")
    monkeypatch.delenv("CONTRA_ALLOW_UNSANDBOXED", raising=False)
    outcome, why = discriminate.run_pytest(repo, repo / "t.py")
    assert outcome == "error"
    assert "bubblewrap" in why
