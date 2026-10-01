"""The hook must not start a copy of itself, and its child must not run commands.

A Stop hook lives in settings.json, so it fires for every `claude -p` on the
machine -- including the three this hook starts. Without a stop, each of those
starts three more. The blast radius is the whole account balance, which is why
this is locked twice and both locks are tested.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HOOKS = Path(__file__).resolve().parents[1] / "hooks"
sys.path.insert(0, str(HOOKS))
import verify_disclosures as vd  # noqa: E402


def run_hook(script: str, payload: dict, env: dict) -> str:
    proc = subprocess.run(
        [sys.executable, str(HOOKS / script)],
        input=json.dumps(payload), capture_output=True, text=True,
        env={**os.environ, **env}, timeout=60)
    return proc.stdout


@pytest.mark.parametrize("script", ["verify_disclosures.py", "record_baseline.py"])
def test_hooks_do_nothing_inside_their_own_child(tmp_path, script):
    out = run_hook(script,
                   {"cwd": str(tmp_path), "session_id": "s", "prompt": "hi"},
                   {"CONTRA_INSIDE": "1", "CONTRA_STATE_DIR": str(tmp_path / "st")})
    assert json.loads(out) == {}
    # Nothing was written -- not even a baseline or a pending file.
    assert not (tmp_path / "st").exists()


def test_the_child_is_marked_and_cannot_run_commands(monkeypatch):
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        seen["env"] = kw.get("env") or {}
        class R: stdout = "{}"
        return R()

    monkeypatch.setattr(vd.subprocess, "run", fake_run)
    vd.claude("system", "payload", timeout=1, label="probe")

    cmd = seen["cmd"]
    assert "--restricted" in cmd, "the child must not get command-running tools"
    # Restricted mode rejects bypassPermissions, so the tools have to be allowed
    # by name -- and an unanswerable prompt under `-p` is a denial, so both
    # flags are needed or every read silently fails.
    assert "--allow-dangerously-skip-permissions" not in cmd
    assert cmd[cmd.index("--tools") + 1] == cmd[cmd.index("--allowed-tools") + 1]
    tools = cmd[cmd.index("--tools") + 1]
    assert "Bash" not in tools
    assert "Read" in tools, "reading the repository is where half the findings come from"
    assert seen["env"].get("CONTRA_INSIDE") == "1"


def test_state_survives_a_reboot(monkeypatch):
    """`verifications.jsonl` is the ledger, not scratch.

    `/tmp` took a day of results with it once already -- a probe prompt and a
    whole A/B run that cannot be recovered.
    """
    import baseline
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    assert not str(baseline._default_state_dir()).startswith("/tmp")

    monkeypatch.setenv("XDG_STATE_HOME", "/somewhere/state")
    assert baseline._default_state_dir() == Path("/somewhere/state/contra")
