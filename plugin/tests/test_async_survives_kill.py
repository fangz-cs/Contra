"""Killing the hook process must not lose the verification it was running.

The host kills the whole process group when the assistant stops, and it only
waits for the hook when a confirmed test wakes the assistant back up. So the
turns that confirm nothing are exactly the turns whose record disappears -- a
loss aimed at the quantity the plugin is measured on. 10 of 32 cells were lost
this way before the work was moved into a session of its own.
"""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HOOK = Path(__file__).resolve().parents[1] / "hooks" / "verify_disclosures.py"


def test_child_lands_results_after_parent_is_killed(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "m.py").write_text("def f(x):\n    return x\n")

    # One entry the hook cannot verify, so the child reaches write_log quickly
    # without needing a model. What matters here is that it writes at all.
    (state / "s1.pending.json").write_text(json.dumps({
        "cwd": str(repo),
        "verify": [{"file": "nonexistent.py", "function": "f", "gap": "g",
                    "chosen": "a", "rival": "b"}],
        "rest": [{"gap": "scope entry", "kind": "declined", "verdict": "scope"}],
    }), encoding="utf-8")

    env = {**os.environ, "CONTRA_STATE_DIR": str(state)}
    proc = subprocess.Popen(
        [sys.executable, str(HOOK), "verify"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=env, start_new_session=True)
    proc.stdin.write(json.dumps({"session_id": "s1", "cwd": str(repo)}))
    proc.stdin.close()

    time.sleep(1.5)
    os.kill(proc.pid, signal.SIGKILL)      # the host taking the group down
    proc.wait(timeout=10)

    log = state / "verifications.jsonl"
    for _ in range(40):
        if log.is_file():
            break
        time.sleep(0.5)
    assert log.is_file(), "the detached child did not write the log"
    row = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
    gaps = {r.get("gap") for r in row["results"]}
    assert "scope entry" in gaps, f"entries that never needed a test were lost: {gaps}"
