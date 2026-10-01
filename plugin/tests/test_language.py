"""The view speaks the reader's language, and never speaks a key.

Only the frame is translated. Entry text comes from the model, which answers in
the language of the request -- so an English frame around Chinese entries is
correct, not a bug.
"""
import importlib
import sys
from pathlib import Path

import pytest

HOOKS = Path(__file__).resolve().parents[1] / "hooks"
sys.path.insert(0, str(HOOKS))
sys.path.insert(0, str(HOOKS.parent / "scripts"))
import config as _config  # noqa: E402
import text as _text  # noqa: E402

ITEM = {"kind": "implementation", "file": "pkg/mod.py", "function": "clamp",
        "gap": "is 512 itself allowed", "chosen": "> 512", "rival": ">= 512",
        "scope": "local", "trigger": "edge", "verdict": "confirmed",
        "source": "reviewer", "test_path": "tests/t.py"}


@pytest.fixture
def view(monkeypatch):
    def render_in(lang):
        if lang is None:
            monkeypatch.delenv("CONTRA_LANG", raising=False)
        else:
            monkeypatch.setenv("CONTRA_LANG", lang)
        importlib.reload(_config)
        importlib.reload(_text)
        import rank
        importlib.reload(rank)
        return rank.render([dict(ITEM), dict(ITEM, function="other")])
    return render_in


def test_english_frame(view):
    out = view("en")
    assert "2 decisions I made for you" in out
    assert "I chose" in out and "Other reading:" in out
    assert "found in review" in out
    assert "verified" in out


def test_chinese_frame(view):
    out = view("zh")
    assert "我替你做的" in out
    assert "我选了" in out and "另一读法" in out
    assert "审查发现" in out


def test_the_locale_decides_when_nothing_is_set(view, monkeypatch):
    monkeypatch.setenv("LANG", "zh_CN.UTF-8")
    monkeypatch.delenv("LC_ALL", raising=False)
    assert "我替你做的" in view(None)


def test_entry_text_is_left_alone(view):
    """A Chinese request inside an English frame keeps its own words."""
    out = view("en")
    assert "> 512" in out and ">= 512" in out


def test_both_tables_carry_the_same_keys():
    """A key added to one language and not the other renders as English.

    That is the right fallback, and it is also invisible -- so the tables are
    compared here instead of being discovered by a reader.
    """
    en, zh = set(_text.TEXT["en"]), set(_text.TEXT["zh"])
    assert en == zh, f"only in en: {en - zh}; only in zh: {zh - en}"


def test_a_missing_key_never_renders_empty():
    assert _text.t("no_such_key_at_all") == "no_such_key_at_all"


def test_a_template_with_a_wrong_slot_still_says_something():
    """Better a template than a blank line where a fact should be."""
    assert _text.t("confirmed", wrong_slot="x")


@pytest.mark.parametrize("n,expected", [(1, "1 decision "), (2, "2 decisions ")])
def test_english_counts_agree_with_their_noun(monkeypatch, n, expected):
    monkeypatch.setenv("CONTRA_LANG", "en")
    importlib.reload(_config)
    importlib.reload(_text)
    assert _text.headline(n).startswith(expected)
