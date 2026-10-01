"""Which declined entries go to the bottom of the view.

A model was asked to make this call three times and returned 42%, 4% and 0%
against a hand-counted truth near 15%. The wide version buried real problems --
an undeclared dependency, seven failing tests -- which is the expensive
direction to get wrong. So the call is made by text instead: narrow, stable,
free, and wrong in the safe direction when it is wrong.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rank import grade, housekeeping, render  # noqa: E402


@pytest.mark.parametrize("gap", [
    "请求未明确是否应更新 CHANGELOG",
    "请求没有说是否应该更新 CHANGES.rst",
    "Flask 仓库的惯例是为用户可见的行为变更编写 CHANGES.rst 条目",
    "请求没有明确说是否需要在变更日志中记录此次改动",
])
def test_ledger_chores_sink(gap):
    assert housekeeping({"kind": "declined", "gap": gap})


@pytest.mark.parametrize("gap", [
    "新增的行为没有任何断言覆盖",
    "请求未指定是否应在 setup.py 中为 brotli 添加 extras_require",
    "请求只提到改 _is_incomplete_option，没有要求修改那 7 个既有的失败测试",
    "教程 README 第 14 行仍以 master branch 描述默认开发分支，与改名后的事实不符",
    "请求没有明确说是否需要公开的缓存清理接口供外部调用",
])
def test_real_problems_never_sink(gap):
    """The expensive error: a real problem pushed under a changelog reminder."""
    assert not housekeeping({"kind": "declined", "gap": gap})


def test_only_declined_can_sink():
    item = {"kind": "beyond_scope", "gap": "顺手改了 CHANGELOG 的格式"}
    assert not housekeeping(item)


def test_it_sorts_to_the_bottom():
    view = render([
        {"kind": "declined", "gap": "请求未明确是否应更新 CHANGELOG",
         "chosen": "没写", "rival": "补一条"},
        {"kind": "declined", "gap": "新行为没有任何测试固定住",
         "chosen": "没补", "rival": "补回归测试"},
    ])
    assert view.index("新行为没有任何测试固定住") < view.index("CHANGELOG")
    assert grade({"kind": "declined", "gap": "更新 CHANGELOG"}) == "housekeeping"


# --- what survives the cap -------------------------------------------------
import collections  # noqa: E402
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "vd", Path(__file__).resolve().parents[1] / "hooks" / "verify_disclosures.py")
_vd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_vd)


def _items(kind, n):
    return [{"kind": kind, "gap": f"{kind}-{i}"} for i in range(n)]


def test_cap_keeps_every_kind_represented():
    """Ordering by kind is for showing the list, not for deciding what to drop.

    Using it for both meant a change raising many scope questions spent the
    whole budget before reaching one implementation choice -- three of the
    misses in the recall measurement were entries the probe had already found.
    """
    got = collections.Counter(i["kind"] for i in _vd.cap(
        _items("beyond_scope", 5) + _items("declined", 6) + _items("implementation", 4), 8))
    assert got["implementation"] >= 2, got
    assert sum(got.values()) == 8


def test_cap_does_not_waste_places_on_absent_kinds():
    got = collections.Counter(i["kind"] for i in _vd.cap(_items("implementation", 10), 8))
    assert got["implementation"] == 8


def test_cap_is_a_noop_below_the_limit():
    items = _items("declined", 2) + _items("implementation", 3)
    assert len(_vd.cap(items, 8)) == 5


# --- entries with nothing in them ------------------------------------------
def test_render_survives_null_fields():
    """`"gap": null` is valid JSON, and it used to take the whole view down.

    `.get(k, "")` only defends against a missing key. The reader then saw
    "rendering failed" in place of everything the turn had found.
    """
    out = render([{"kind": None, "gap": None, "chosen": None, "rival": None,
                   "verdict": None, "scope": None, "function": None,
                   "caller_hint": None, "source": None, "context": None}])
    assert "None" not in out


def test_a_real_entry_still_renders_beside_a_broken_one():
    out = render([
        {"kind": None, "gap": None, "chosen": None, "rival": None},
        {"kind": "implementation", "gap": "br 放哪", "chosen": "末尾",
         "rival": "开头", "scope": "caller", "verdict": "confirmed",
         "test_path": "t.py"},
    ])
    assert "br 放哪" in out and "t.py" in out
