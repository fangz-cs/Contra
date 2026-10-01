"""Run one patched module against the calls its own test suite already made.

Splicing a replacement function back into the real module source keeps every
helper, constant and import the function depends on, so the patch is exercised
in its true context rather than as an isolated snippet.
"""
from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass

BWRAP = "/usr/bin/bwrap"
TIMEOUT_SECONDS = 30

DRIVER = r'''
import importlib, json, sys

sys.setrecursionlimit(10000)


def canon(v, depth=0):
    if depth > 24:
        return ["truncated"]
    if v is None:
        return ["none"]
    if isinstance(v, bool):
        return ["bool", v]
    if isinstance(v, int):
        return ["int", v]
    if isinstance(v, float):
        if v != v:
            return ["float", "nan"]
        if v in (float("inf"), float("-inf")):
            return ["float", repr(v)]
        return ["float", format(round(v, 9) + 0.0, ".9g")]
    if isinstance(v, str):
        return ["str", v]
    if isinstance(v, (list, tuple)):
        return ["seq", [canon(x, depth + 1) for x in v]]
    if isinstance(v, (set, frozenset)):
        return ["set", sorted(json.dumps(canon(x, depth + 1), sort_keys=True) for x in v)]
    if isinstance(v, dict):
        return ["map", sorted(
            [json.dumps(canon(k, depth + 1), sort_keys=True),
             json.dumps(canon(x, depth + 1), sort_keys=True)]
            for k, x in v.items()
        )]
    # A generator carries its behaviour in what it yields, not in the object,
    # whose repr is an address that differs on every run and would otherwise
    # read as a behavioural difference on every single input.
    if hasattr(v, "__iter__") and hasattr(v, "__next__"):
        items = []
        for position, item in enumerate(v):
            if position >= 500:
                items.append(["truncated"])
                break
            items.append(canon(item, depth + 1))
        return ["iter", items]
    text = repr(v)
    if " at 0x" in text or " object at " in text:
        # Default repr: identity, not value. Say it cannot be compared rather
        # than reporting the address as a difference.
        return ["opaque", type(v).__name__]
    return ["repr", text]


def main():
    spec = json.load(sys.stdin)
    module = importlib.import_module(spec["module"])
    fn = getattr(module, spec["function"])
    out = []
    for case in spec["cases"]:
        try:
            value = fn(**case["kwargs"])
        except Exception as exc:
            out.append({"id": case["id"], "status": "raised",
                        "value": ["exc", type(exc).__name__]})
        else:
            out.append({"id": case["id"], "status": "ok", "value": canon(value)})
    print(json.dumps(out, sort_keys=True))


main()
'''


@dataclass(frozen=True)
class RunOutcome:
    status: str
    observations: dict[str, str]
    stderr_tail: str = ""


MAX_EDIT_LINES = 40


def apply_edits(module_source: str, edits: list[dict]) -> str:
    """Apply a set of exact-match replacements to one file.

    The unit of a change is a patch, not a function. Fixing the unit at "one
    function body" quietly excluded everything that happens at module level --
    a default constant, an import, a class attribute -- which is where some of
    the most consequential decisions live. Matching on an exact snippet covers
    those, and requiring the snippet to occur exactly once keeps the same
    guarantee a function swap gave: nothing outside the named region moves.
    """
    if not edits:
        raise ValueError("no edits given")
    out = module_source
    for index, edit in enumerate(edits):
        old = edit.get("old")
        new = edit.get("new")
        if not isinstance(old, str) or not isinstance(new, str) or not old.strip():
            raise ValueError(f"edit {index}: needs a non-empty 'old' and a 'new'")
        if len(old.splitlines()) > MAX_EDIT_LINES:
            raise ValueError(
                f"edit {index}: 'old' spans {len(old.splitlines())} lines, over the "
                f"{MAX_EDIT_LINES}-line limit -- narrow it to the part that changes")
        count = out.count(old)
        if count == 0:
            head = old.strip().splitlines()[0][:70] if old.strip() else ""
            raise ValueError(f"edit {index}: snippet not found: {head!r}")
        if count > 1:
            head = old.strip().splitlines()[0][:70]
            raise ValueError(
                f"edit {index}: snippet appears {count} times, so the target is "
                f"ambiguous -- include surrounding lines: {head!r}")
        out = out.replace(old, new, 1)
    return out


def splice(module_source: str, function_name: str, replacement: str) -> str:
    """Replace one whole function. Kept for callers that work in function units."""
    tree = ast.parse(module_source)
    owner, _, leaf = function_name.rpartition(".")

    def bodies():
        if not owner:
            yield tree.body, 0
            return
        nodes, parts = tree.body, owner.split(".")
        for part in parts:
            found = next((n for n in nodes
                          if isinstance(n, ast.ClassDef) and n.name == part), None)
            if found is None:
                return
            nodes = found.body
        yield nodes, (nodes[0].col_offset if nodes else 4)

    for nodes, indent in bodies():
        for node in nodes:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name != leaf:
                continue
            lines = module_source.splitlines(keepends=True)
            start = node.lineno - 1
            if node.decorator_list:
                start = min(d.lineno for d in node.decorator_list) - 1
            end = node.end_lineno
            body = textwrap.dedent(replacement).rstrip("\n") + "\n"
            if indent:
                pad = " " * indent
                body = "".join(pad + ln if ln.strip() else ln
                               for ln in body.splitlines(keepends=True))
            return "".join(lines[:start]) + body + "".join(lines[end:])
    raise ValueError(f"function not found: {function_name}")


def run(
    *,
    package_dir: str,
    package_name: str,
    module_file: str,
    module_name: str,
    function_name: str,
    patched_source: str,
    cases: list[dict],
    allow_unsandboxed: bool = False,
) -> RunOutcome:
    workdir = tempfile.mkdtemp(prefix="patchbv_")
    try:
        target_pkg = os.path.join(workdir, package_name)
        shutil.copytree(package_dir, target_pkg)
        with open(os.path.join(target_pkg, module_file), "w", encoding="utf-8") as handle:
            handle.write(patched_source)
        driver_path = os.path.join(workdir, "_driver.py")
        with open(driver_path, "w", encoding="utf-8") as handle:
            handle.write(DRIVER)
        payload = json.dumps(
            {"module": module_name, "function": function_name, "cases": cases}
        )
        # The project's dependencies live in whatever interpreter is running us,
        # so the sandbox has to carry that prefix in, not just /usr.
        interpreter = sys.executable
        prefix = sys.prefix
        have_bwrap = os.path.isfile(BWRAP) and os.access(BWRAP, os.X_OK)
        if not have_bwrap and not allow_unsandboxed:
            # This runs code a model just wrote. Refuse rather than quietly
            # execute it on the host with no isolation.
            return RunOutcome(
                "no_sandbox", {},
                f"{BWRAP} not available; install bubblewrap or pass "
                "allow_unsandboxed=True to accept host execution",
            )
        if have_bwrap:
            command = [
                BWRAP, "--unshare-all", "--die-with-parent", "--new-session",
                "--ro-bind", "/usr", "/usr", "--ro-bind", "/lib", "/lib",
                "--ro-bind-try", "/lib64", "/lib64", "--ro-bind-try", "/etc", "/etc",
                "--ro-bind-try", prefix, prefix,
                "--proc", "/proc", "--dev", "/dev",
                "--bind", workdir, "/work", "--chdir", "/work",
                "--setenv", "PYTHONPATH", "/work",
                "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
                "--setenv", "PYTHONHASHSEED", "0",
                interpreter, "/work/_driver.py",
            ]
        else:
            command = [interpreter, driver_path]
        env = {
            "PYTHONPATH": workdir,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PATH": "/usr/bin:/bin",
            "LC_ALL": "C",
            "HOME": "/nonexistent",
        }
        try:
            proc = subprocess.run(
                command, input=payload, capture_output=True, text=True,
                timeout=TIMEOUT_SECONDS, env=env, cwd=workdir,
            )
        except subprocess.TimeoutExpired:
            return RunOutcome("timeout", {})
        if proc.returncode != 0 or not proc.stdout.strip():
            return RunOutcome("crash", {}, (proc.stderr or "")[-600:])
        try:
            rows = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return RunOutcome("unparsable", {}, proc.stdout[:400])
        observations = {
            str(row["id"]): json.dumps(
                [row["status"], row["value"]], sort_keys=True
            )
            for row in rows
        }
        return RunOutcome("ok", observations)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
