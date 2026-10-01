#!/usr/bin/env python3
"""View wording, in the reader's language.

Only the frame is translated. What the entries say -- the gap, the reading
chosen, the rival -- comes from the model, which is told to answer in the
language of the request; a request written in Chinese gets Chinese entries
inside an English frame if that is what the locale says, and that is correct.
Pinning both to one language would either mistranslate the request's own words
or force the reader to read their own request back in a language they did not
use.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

TEXT = {
    "en": {
        "header": "decisions I made for you",
        "kind_line": "{title}: {gap}",
        "kind_beyond_scope": "Beyond what you asked",
        "kind_declined": "Decided not to do",
        "kind_housekeeping": "Owed to the project's conventions",
        "from_reviewer": " (found in review; not raised)",
        "from_both": " (raised, and independently found in review)",
        "scope_line": "   I {chosen} · Alternative: {rival}",
        "choice_line": "   I chose {chosen} · Other reading: {rival}",
        "reach_cross_module": "no longer agrees with other modules",
        "reach_caller": "affects callers {hint}",
        "reach_local": "this function only",
        "trigger_common": "ordinary input reaches it",
        "trigger_edge": "only edge input reaches it",
        "irreversible": "expensive to undo",
        "confirmed": "verified → {path} (fails under the other reading)",
        "see_below": "see the test below",
        "pending": ("writing a discriminating test in the background; "
                    "confirmed ones are saved to the repository, the rest to ledger.py"),
        "untested": "not verified: {why}",
        "untested_default": "affects this function only",
        "not_separated": "could not verify: the test passes under both readings",
        "unverifiable": "could not verify: {why}",
        "unverifiable_default": "the tool cannot handle this shape",
        "skipped_local_edge": "local and edge-only; not worth the verification cost",
        "no_runner": "no test runner for {suffix}; reported but not verified",
        "render_failed": "(could not render the view: {exc})",
        "baseline_failed": "CONTRA could not record this turn's baseline: {exc}",
        "save_tests": (
            "The discriminating tests below confirm assumptions disclosed earlier: "
            "each passes on the implemented reading and fails on the other. Save "
            "them into the repository's test directory (name the file "
            "test_contra_<function>.py), then finish.\n\n{tests}\n\n"
            "The ranked view has already been shown to the user directly. Do not "
            "repeat it. Just say where the tests went.\n"),
        "test_for": "# for: {function} — {gap}",
        "ledger_empty": "The ledger is empty: no disclosed decisions recorded yet.",
        "stale_test_gone": "the test that was the evidence is no longer in the repository",
        "stale_file_changed": "the file has been committed to again since",
        "unknown_file": "(unknown file)",
        "ledger_summary": ("{n} decisions the assistant made for you in this "
                           "repository, across {files} files"),
        "ledger_tiers": ("  ⚑ beyond {beyond}   ⚐ declined {declined}"
                         "   ● high {high}   ◐ medium {medium}   ○ low {low}"),
        "ledger_verified": "  confirmed by test {verified}/{testable}",
        "ledger_stale": "   needs review {stale}",
        "html_written": "page written to {path}",
        "html_title": "Decisions the assistant made for you",
        "html_sub": "{n} entries across {files} files",
        "html_meta": "chose {chosen} · other reading: {rival}",
    },
    "zh": {
        "header": "我替你做的",
        "kind_line": "{title}：{gap}",
        "kind_beyond_scope": "超出你说的范围",
        "kind_declined": "我决定不做的",
        "kind_housekeeping": "仓库惯例上还欠着的",
        "from_reviewer": "（审查发现，助手未提）",
        "from_both": "（助手已提，审查独立复现）",
        "scope_line": "   我{chosen} · 另一种做法：{rival}",
        "choice_line": "   我选了{chosen} · 另一读法：{rival}",
        "reach_cross_module": "与其他模块的行为不再一致",
        "reach_caller": "影响调用方 {hint}",
        "reach_local": "只影响本函数",
        "trigger_common": "普通输入即触发",
        "trigger_edge": "只在边界输入触发",
        "irreversible": "选错难以撤销",
        "confirmed": "已验证 → {path}（改成另一读法则失败）",
        "see_below": "见下方测试",
        "pending": "后台正在写区分性测试；证实的会存进仓库，其余见 ledger.py",
        "untested": "未验证：{why}",
        "untested_default": "影响局部",
        "not_separated": "未能验证：写出的测试在两种读法下都通过",
        "unverifiable": "未能验证：{why}",
        "unverifiable_default": "工具无法处理这个形态",
        "skipped_local_edge": "影响局部且只在边界输入触发，未花验证成本",
        "no_runner": "没有 {suffix} 的测试运行器，这条只报不验",
        "render_failed": "(分级视图渲染失败: {exc})",
        "baseline_failed": "CONTRA 未能记录本轮基线：{exc}",
        "save_tests": (
            "以下区分性测试已经证实了先前披露的假设：它们在已实现的读法下通过，"
            "在另一种读法下失败。请把它们保存到仓库的测试目录里"
            "（文件名用 test_contra_<函数名>.py），然后结束。\n\n{tests}\n\n"
            "分级视图此前已直接呈现给用户，不要重复它。"
            "只需说明测试存在了哪里。\n"),
        "test_for": "# 对应：{function} — {gap}",
        "ledger_empty": "账本为空：还没有记录任何被披露的决定。",
        "stale_test_gone": "证据测试已不在仓库中",
        "stale_file_changed": "该文件此后又被提交修改过",
        "unknown_file": "(未知文件)",
        "ledger_summary": "这个仓库里有 {n} 个由助手拿主意的决定，分布在 {files} 个文件",
        "ledger_tiers": ("  ⚑ 越界 {beyond}   ⚐ 未做 {declined}"
                         "   ● 高 {high}   ◐ 中 {medium}   ○ 低 {low}"),
        "ledger_verified": "  已由测试证实 {verified}/{testable}",
        "ledger_stale": "   需要复核 {stale}",
        "html_written": "页面已写入 {path}",
        "html_title": "助手替你做的决定",
        "html_sub": "{n} 条，分布在 {files} 个文件",
        "html_meta": "选了{chosen} · 另读：{rival}",
    },
}


def t(key: str, **kw) -> str:
    lang = config.language()
    table = TEXT.get(lang) or TEXT["en"]
    # An untranslated key must not become an empty line in the view.
    template = table.get(key) or TEXT["en"].get(key) or key
    try:
        return template.format(**kw) if kw else template
    except (KeyError, IndexError, ValueError):
        return template


def headline(n: int) -> str:
    """Word order differs enough that the count cannot be a plain slot."""
    if config.language() == "zh":
        return f"我替你做的 {n} 个决定"
    return f"{n} decision{'' if n == 1 else 's'} I made for you"
