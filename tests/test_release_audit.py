from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


AUDIT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_release.py"
SPEC = importlib.util.spec_from_file_location("release_audit", AUDIT_PATH)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def test_audit_accepts_plain_source(tmp_path) -> None:
    path = tmp_path / "method.py"
    path.write_text("def method():\n    return 1\n", encoding="utf-8")
    assert audit.scan_paths(tmp_path, [path]) == []


@pytest.mark.parametrize("filename", ["records.jsonl", "paper.zip", "table.csv", "weights.pt"])
def test_audit_rejects_data_and_archive_payloads(tmp_path, filename) -> None:
    path = tmp_path / filename
    path.write_text("synthetic", encoding="utf-8")
    assert any(item.category == "non-source payload"
               for item in audit.scan_paths(tmp_path, [path]))


def test_audit_rejects_personal_paths_without_printing_contents(tmp_path) -> None:
    path = tmp_path / "settings.py"
    private_path = str(Path("/home") / "example-user" / "private")
    path.write_text(f"ROOT = {private_path!r}\n", encoding="utf-8")
    findings = audit.scan_paths(tmp_path, [path])
    assert findings[0].category == "personal home path"
    assert findings[0].line == 1
    assert private_path not in repr(findings)


def test_audit_rejects_explicit_identifiers(tmp_path) -> None:
    path = tmp_path / "README.md"
    path.write_text("An excluded identity.\n", encoding="utf-8")
    assert audit.scan_paths(tmp_path, [path], ["excluded identity"])


def test_audit_rejects_symlinks(tmp_path) -> None:
    original = tmp_path / "method.py"
    original.write_text("pass\n", encoding="utf-8")
    alias = tmp_path / "alias.py"
    alias.symlink_to(original)
    assert audit.scan_paths(tmp_path, [alias])[0].category == "symbolic link"
