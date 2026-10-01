"""Where the plugin is willing to read from.

The path comes from a model that has been reading the repository, so it is
untrusted input: a suggestive comment in the tree is enough to make it name
somewhere else. Before this, both an absolute path and a `../` escape were
accepted, and the file was read and handed to the next model call.
"""
import importlib.util
import os
import sys
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "vd", Path(__file__).resolve().parents[1] / "hooks" / "verify_disclosures.py")
_vd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_vd)


@pytest.fixture
def tree(tmp_path):
    (tmp_path / "ok.py").write_text("x = 1\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "deep.py").write_text("y = 2\n")
    return tmp_path


@pytest.mark.parametrize("path", [
    "/etc/passwd",
    "../../../etc/passwd",
    "sub/../../outside.py",
])
def test_paths_outside_the_tree_are_refused(tree, path):
    target, why = _vd.locate({"file": path}, tree)
    assert target is None
    assert "outside" in why


def test_symlinks_are_refused(tree):
    """A link inside the tree can still point anywhere."""
    os.symlink("/etc/passwd", tree / "link.py")
    target, why = _vd.locate({"file": "link.py"}, tree)
    assert target is None


@pytest.mark.parametrize("path", ["ok.py", "sub/deep.py", "sub/../ok.py"])
def test_paths_inside_the_tree_are_accepted(tree, path):
    target, why = _vd.locate({"file": path}, tree)
    assert target is not None, why


def test_quoted_paths_still_work(tree):
    assert _vd.locate({"file": '"ok.py"'}, tree)[0] is not None
