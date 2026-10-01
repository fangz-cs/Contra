"""What a turn changed, versus what the tree happens to hold.

Each test here is a failure the old implementation had on a real repository.
`git diff -- '*.py'` cannot see a new file, cannot see a staged change, and
measures against HEAD rather than against the start of the turn -- so it
reported last week's uncommitted work as this turn's decisions, every turn.
"""
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hooks"))
import baseline  # noqa: E402


def sh(cwd, *args):
    subprocess.run(args, cwd=str(cwd), capture_output=True, check=False)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setattr(baseline, "STATE_DIR", tmp_path / "state")
    d = tmp_path / "repo"
    d.mkdir()
    sh(d, "git", "init", "-q")
    (d / "a.py").write_text("def f():\n    return 1\n")
    (d / "noise.py").write_text("OLD = 1\n")
    sh(d, "git", "add", "-A")
    sh(d, "git", "-c", "user.email=a@b", "-c", "user.name=a", "commit", "-qm", "i")
    return d


def test_new_file_is_visible(repo):
    """The old diff omitted untracked files, so anything just written vanished."""
    snap = baseline.snapshot(repo)
    (repo / "b.py").write_text("def g():\n    return 2\n")
    diff = baseline.turn_diff(repo, snap)
    assert "b.py" in diff and "return 2" in diff


def test_staged_change_is_visible(repo):
    """`git diff` without HEAD hides whatever has been added to the index."""
    snap = baseline.snapshot(repo)
    (repo / "a.py").write_text("def f():\n    return 99\n")
    sh(repo, "git", "add", "a.py")
    assert "return 99" in baseline.turn_diff(repo, snap)


def test_earlier_uncommitted_work_is_excluded(repo):
    """The point of the baseline: only what moved after the prompt arrived."""
    (repo / "noise.py").write_text("OLD = 999\n")
    snap = baseline.snapshot(repo)
    (repo / "a.py").write_text("def f():\n    return 7\n")
    diff = baseline.turn_diff(repo, snap)
    assert "noise.py" not in diff
    assert "return 7" in diff


def test_deletion_counts(repo):
    snap = baseline.snapshot(repo)
    (repo / "a.py").unlink()
    assert "a.py" in baseline.turn_diff(repo, snap)


def test_idle_turn_produces_nothing(repo):
    """A turn that answered a question must not trigger the whole pipeline."""
    assert baseline.turn_diff(repo, baseline.snapshot(repo)).strip() == ""


def test_works_outside_git(tmp_path):
    d = tmp_path / "plain"
    d.mkdir()
    (d / "a.py").write_text("x = 1\n")
    snap = baseline.snapshot(d)
    (d / "a.py").write_text("x = 2\n")
    assert "x = 2" in baseline.turn_diff(d, snap)


def test_missing_baseline_claims_nothing(repo):
    """Falling back to the working tree is exactly the bug this replaced."""
    (repo / "a.py").write_text("def f():\n    return 5\n")
    assert baseline.turn_diff(repo, None) == ""


def test_gitignore_is_honoured(repo):
    (repo / ".gitignore").write_text("skip/\n")
    (repo / "skip").mkdir()
    (repo / "skip" / "junk.py").write_text("z = 1\n")
    sh(repo, "git", "add", ".gitignore")
    snap = baseline.snapshot(repo)
    (repo / "skip" / "junk.py").write_text("z = 2\n")
    (repo / "a.py").write_text("def f():\n    return 3\n")
    diff = baseline.turn_diff(repo, snap)
    assert "junk.py" not in diff
    assert "return 3" in diff


def test_snapshot_stores_hashes_not_source(repo):
    """Inside git, nothing is inlined -- not even a file git has never seen.

    A tree with many untracked Python files would otherwise write megabytes of
    source on every prompt.
    """
    (repo / "noise.py").write_text("OLD = 2\n")       # dirty, tracked
    (repo / "brand_new.py").write_text("NEW = 1\n")   # untracked
    files = baseline.snapshot(repo)["files"]
    assert all("blob" in v for v in files.values()), files


def test_snapshot_falls_back_to_text_without_git(tmp_path):
    d = tmp_path / "plain"
    d.mkdir()
    (d / "a.py").write_text("x = 1\n")
    assert "text" in baseline.snapshot(d)["files"]["a.py"]


def test_repeated_snapshots_are_idempotent(repo):
    """Same content, same object: turns do not accumulate storage."""
    (repo / "brand_new.py").write_text("NEW = 1\n")
    first = baseline.snapshot(repo)["files"]
    second = baseline.snapshot(repo)["files"]
    assert first == second


def test_inline_fallback_is_bounded(tmp_path, monkeypatch):
    """Outside git there is no object store, so the state file needs a ceiling."""
    monkeypatch.setattr(baseline, "MAX_INLINE_TOTAL", 2000)
    d = tmp_path / "plain"
    d.mkdir()
    for i in range(20):
        (d / f"m{i}.py").write_text("x = 1\n" * 100)
    snap = baseline.snapshot(d)
    assert snap.get("dropped", 0) > 0
    assert sum(len(v.get("text", "")) for v in snap["files"].values()) <= 2000


def test_comment_only_change_does_not_fire(repo):
    """A comment costs the reader nothing to add and 100 seconds to report."""
    snap = baseline.snapshot(repo)
    (repo / "a.py").write_text("def f():\n    # explain the return\n    return 1\n")
    assert baseline.turn_diff(repo, snap).strip() == ""


def test_reformatting_does_not_fire(repo):
    snap = baseline.snapshot(repo)
    (repo / "a.py").write_text("def f():\n\n\n    return 1\n")
    assert baseline.turn_diff(repo, snap).strip() == ""


def test_docstring_change_does_fire(repo):
    """A docstring states a contract, so rewriting one is a real change."""
    snap = baseline.snapshot(repo)
    (repo / "a.py").write_text('def f():\n    """Now promises something."""\n    return 1\n')
    assert "a.py" in baseline.turn_diff(repo, snap)


def test_one_real_change_among_comments_still_fires(repo):
    """Filtering is per file; a formatting pass must not hide a real edit."""
    snap = baseline.snapshot(repo)
    (repo / "a.py").write_text("def f():\n    # tidy\n    return 1\n")
    (repo / "noise.py").write_text("OLD = 2\n")
    diff = baseline.turn_diff(repo, snap)
    assert "noise.py" in diff
    assert "a.py" not in diff


def test_other_languages_are_tracked(tmp_path, monkeypatch):
    """Two thirds of what the plugin reports needs no test runner at all."""
    monkeypatch.setattr(baseline, "STATE_DIR", tmp_path / "state")
    d = tmp_path / "poly"
    d.mkdir()
    sh(d, "git", "init", "-q")
    for name, body in (("a.go", "package main\nfunc F() int { return 1 }\n"),
                       ("b.ts", "export const f = () => 1\n"),
                       ("c.rs", "pub fn f() -> i32 { 1 }\n")):
        (d / name).write_text(body)
    snap = baseline.snapshot(d)
    assert set(snap["files"]) == {"a.go", "b.ts", "c.rs"}
    (d / "a.go").write_text("package main\nfunc F() int { return 2 }\n")
    assert "a.go" in baseline.turn_diff(d, snap)


def test_unrecoverable_blob_reports_nothing(repo):
    """An unreadable earlier state must not be read as "the file was empty".

    Returning "" there makes the whole file look newly added, and the reviewer
    is handed a diff full of decisions this turn never made. Blobs really do go
    missing: `git gc` collects unreachable objects, and these are unreachable
    by design.
    """
    snap = baseline.snapshot(repo)
    snap["files"]["a.py"] = {"blob": "0" * 40}       # as if gc had run
    (repo / "a.py").write_text("def f():\n    return 999\n")
    assert baseline.turn_diff(repo, snap).strip() == ""


def test_a_genuinely_empty_file_still_diffs(repo):
    """The other side of the same distinction."""
    (repo / "blank.py").write_text("")
    snap = baseline.snapshot(repo)
    (repo / "blank.py").write_text("x = 1\n")
    assert "x = 1" in baseline.turn_diff(repo, snap)
