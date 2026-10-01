"""The two instruments must describe an entry the same way.

They are deliberately independent -- one reads the assistant's report, the other
reads the diff without seeing it -- but their output lands in one view. When the
probe stopped asking for `trigger`, `caller_hint` and `context`, entries found
by review reached the reader with blank lines exactly where entries the
assistant disclosed had detail. Nothing failed; the view just got quieter on
one side, which is the kind of defect that survives for months.
"""
import re
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[1] / "hooks" / "verify_disclosures.py"
SRC = HOOK.read_text(encoding="utf-8")

# Every field the view or the verifier reads off an entry.
CONTRACT = {"kind", "file", "function", "gap", "context", "chosen", "rival",
            "scope", "caller_hint", "trigger", "reversible"}


def fields(name: str) -> set:
    start = SRC.index(f'{name} = """')
    body = SRC[start:SRC.index('"""', start + len(name) + 7)]
    return set(re.findall(r'"([a-z_]+)":', body))


@pytest.mark.parametrize("name", ["PROBE_SYSTEM", "EXTRACT_SYSTEM"])
def test_prompt_asks_for_every_field_the_code_reads(name):
    missing = CONTRACT - fields(name)
    assert not missing, f"{name} never asks for: {sorted(missing)}"


def test_both_prompts_pin_the_output_language():
    """Mixed languages break the local merge outright: character overlap across
    scripts is zero, so the same decision is shown twice."""
    for name in ("PROBE_SYSTEM", "EXTRACT_SYSTEM"):
        start = SRC.index(f'{name} = """')
        body = SRC[start:SRC.index('"""', start + len(name) + 7)]
        assert "same language as the" in body, name
