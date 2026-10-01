"""Settings, and what happens when one arrives unusable.

The failure to guard against is not a crash. It is `CONTRA_MAX_ITEMS=lots`
quietly becoming zero, or a suffix list becoming empty -- every defect this tool
has had degraded into reporting less, and a misread setting is the cheapest way
to do it again.
"""
import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hooks"))
import config as _config  # noqa: E402


@pytest.fixture
def cfg(monkeypatch, tmp_path):
    """A fresh module with no config file and no CONTRA_* in the environment."""
    for key in list(sys.modules):
        pass
    for name in list(_config.DEFAULTS):
        monkeypatch.delenv("CONTRA_" + name.upper(), raising=False)
    monkeypatch.delenv("CONTRA_CONFIG", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    def reload():
        return importlib.reload(_config)

    return reload


def write_config(tmp_path, data) -> Path:
    d = tmp_path / "contra"
    d.mkdir(exist_ok=True)
    p = d / "config.json"
    p.write_text(json.dumps(data) if not isinstance(data, str) else data)
    return p


def test_defaults_apply_with_nothing_set(cfg):
    c = cfg()
    assert c.get("max_items") == 8
    assert c.get("sandbox") == "required"
    assert c.get("probe_model") is None


def test_a_file_overrides_the_default(cfg, tmp_path):
    write_config(tmp_path, {"max_items": 3, "sandbox": "allow_host"})
    c = cfg()
    assert c.get("max_items") == 3
    assert c.get("sandbox") == "allow_host"


def test_the_environment_beats_the_file(cfg, tmp_path, monkeypatch):
    write_config(tmp_path, {"max_items": 3})
    monkeypatch.setenv("CONTRA_MAX_ITEMS", "5")
    assert cfg().get("max_items") == 5


@pytest.mark.parametrize("raw", ["lots", "", "3.5.1", "-"])
def test_an_unusable_number_falls_back_rather_than_becoming_zero(
        cfg, monkeypatch, capsys, raw):
    monkeypatch.setenv("CONTRA_MAX_ITEMS", raw)
    c = cfg()
    assert c.get("max_items") == 8
    assert "max_items" in capsys.readouterr().err


def test_a_broken_config_file_is_ignored_loudly(cfg, tmp_path, capsys):
    write_config(tmp_path, "{not json,}")
    c = cfg()
    assert c.get("max_items") == 8
    assert "ignoring" in capsys.readouterr().err


def test_a_wrongly_typed_file_value_does_not_reach_the_code(cfg, tmp_path, capsys):
    write_config(tmp_path, {"max_items": "eight"})
    c = cfg()
    assert c.get("max_items") == 8
    assert "not a int" in capsys.readouterr().err


def test_an_unknown_setting_is_named(cfg, tmp_path, capsys):
    write_config(tmp_path, {"maxitems": 3})
    cfg()
    assert "maxitems" in capsys.readouterr().err


@pytest.mark.parametrize("raw", ["py,js", ".py,.js", "py .js", " py , js "])
def test_suffixes_accept_the_shapes_people_actually_type(cfg, monkeypatch, raw):
    monkeypatch.setenv("CONTRA_RUNNABLE_SUFFIXES", raw)
    assert cfg().suffixes("runnable_suffixes") == (".py", ".js")


def test_an_empty_suffix_list_keeps_the_default(cfg, monkeypatch):
    """Empty here would silently stop every entry from being verified."""
    monkeypatch.setenv("CONTRA_SOURCE_SUFFIXES", "   ")
    assert cfg().suffixes("source_suffixes")[0] == ".py"


@pytest.mark.parametrize("raw,expected", [
    ("1", True), ("true", True), ("yes", True),
    ("0", False), ("false", False), ("no", False),
])
def test_booleans_read_the_way_people_write_them(cfg, monkeypatch, raw, expected):
    monkeypatch.setenv("CONTRA_SKIP_UNCHANGED_AST", raw)
    assert cfg().get("skip_unchanged_ast") is expected


class TestLanguage:
    def test_an_explicit_setting_wins(self, cfg, monkeypatch):
        monkeypatch.setenv("LANG", "en_US.UTF-8")
        monkeypatch.setenv("CONTRA_LANG", "zh")
        assert cfg().language() == "zh"

    def test_a_chinese_locale_is_followed(self, cfg, monkeypatch):
        monkeypatch.setenv("LANG", "zh_CN.UTF-8")
        assert cfg().language() == "zh"

    def test_anything_else_falls_back_to_english(self, cfg, monkeypatch):
        monkeypatch.setenv("LANG", "C.UTF-8")
        assert cfg().language() == "en"

    def test_an_unsupported_language_says_so(self, cfg, monkeypatch, capsys):
        monkeypatch.setenv("CONTRA_LANG", "fr")
        assert cfg().language() == "en"
        assert "fr" in capsys.readouterr().err


def test_describe_names_where_each_value_came_from(cfg, monkeypatch):
    monkeypatch.setenv("CONTRA_MAX_ITEMS", "4")
    out = cfg().describe()
    assert "max_items" in out and "env" in out
    assert "default" in out
