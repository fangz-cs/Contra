from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SELECTOR_PATH = ROOT / "code" / "selector" / "run_interactive_question_selector.py"
SPEC = importlib.util.spec_from_file_location("public_selector", SELECTOR_PATH)
assert SPEC is not None and SPEC.loader is not None
selector = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = selector
SPEC.loader.exec_module(selector)


def command(script: str, *args: str, cwd: Path = ROOT):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run(
        [sys.executable, str(ROOT / script), *args],
        cwd=cwd, env=env, capture_output=True, text=True, timeout=30,
    )


@pytest.mark.parametrize(
    "script",
    [
        "code/pipeline/scripts/run_agent1_decision_relevance.py",
        "code/pipeline/scripts/run_agent2_contrastive_resolution.py",
        "code/pipeline/scripts/build_rite_v3_question_plan.py",
        "code/pipeline/scripts/build_stage_question_plans.py",
        "code/pipeline/scripts/run_ccb_v3_pregenerated.py",
        "code/pipeline/scripts/score_ccb_v3_questions.py",
        "code/selector/run_interactive_question_selector.py",
        "code/evaluation/protocol/score_protocol.py",
        "code/evaluation/codegen/run_contextual_pass1_generation.py",
        "code/evaluation/codegen/evaluate_contextual_pass1.py",
        "code/harness/run_harness.py",
    ],
)
def test_public_command_line_entrypoints_import_and_show_help(script) -> None:
    result = command(script, "--help")
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_selector_uses_public_environment_credentials(monkeypatch) -> None:
    monkeypatch.setenv("CONTRA_API_KEY", "synthetic-configured-key")
    client = selector._load_gateway_client(
        transport_path=ROOT / "code" / "selector" / "openai_transport.py",
        base_url="https://api.example.invalid/v1",
    )
    try:
        assert client.api_key == "synthetic-configured-key"
        assert client.max_retries == 0
    finally:
        client.close()


def test_selector_never_reads_a_private_local_key_file(monkeypatch) -> None:
    for name in ("CONTRA_API_KEY", "CLARIFY_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(Path, "home", lambda: pytest.fail("no private-key lookup"))
    with pytest.raises(selector.SelectorError, match="set CONTRA_API_KEY"):
        selector._load_gateway_client(
            transport_path=ROOT / "code" / "selector" / "openai_transport.py",
            base_url="https://api.example.invalid/v1",
        )


def test_selector_defaults_to_the_public_transport_and_configured_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("CONTRA_BASE_URL", "https://api.example.invalid/v1")
    monkeypatch.setattr(
        sys, "argv",
        ["selector", "--data", "tasks.jsonl", "--plan", "plan.json",
         "--release", "runtime", "--out", "output.jsonl", "--model", "test-model",
         "--dry-run"],
    )
    args = selector.parse_args()
    assert args.transport == ROOT / "code" / "selector" / "openai_transport.py"
    assert args.base_url == "https://api.example.invalid/v1"


def test_selector_dry_run_validates_a_synthetic_frozen_plan_without_api(tmp_path) -> None:
    runtime = tmp_path / "runtime" / "clarifycodebench"
    runtime.mkdir(parents=True)
    (runtime / "__init__.py").write_text("", encoding="utf-8")
    (runtime / "prompts.py").write_text(
        'SYSTEM_PROMPT = "synthetic policy"\nJUDGE_PROMPT = "synthetic judge"\n',
        encoding="utf-8",
    )
    (runtime / "interact.py").write_text(
        "import json\nfrom pathlib import Path\n"
        "def load_tasks(path):\n"
        "    return [json.loads(line) for line in Path(path).read_text().splitlines()]\n",
        encoding="utf-8",
    )
    (runtime / "extract.py").write_text(
        "def extract_code(text):\n    return text\n"
        "def extract_question(text):\n    return text\n"
        "def is_code(text):\n    return False\n",
        encoding="utf-8",
    )
    statement = "Print a value."
    task = {"task_id": "synthetic-task", "question_id": "synthetic-source",
            "modified_content": statement, "qa_pairs": []}
    data = tmp_path / "tasks.jsonl"
    data.write_text(json.dumps(task) + "\n", encoding="utf-8")
    plan = {
        "schema_version": selector.PLAN_SCHEMA,
        "benchmark_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "task_count": 1,
        "tasks": {
            task["task_id"]: {
                "question_id": task["question_id"],
                "statement_sha256": hashlib.sha256(statement.encode()).hexdigest(),
                "questions": [{"candidate_id": "synthetic-format",
                               "text": "Use one line or multiple lines?",
                               "queue_position": 0}],
            },
        },
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    output = tmp_path / "result.jsonl"
    result = command(
        "code/selector/run_interactive_question_selector.py",
        "--data", str(data), "--plan", str(plan_path),
        "--release", str(runtime.parent), "--out", str(output),
        "--model", "test-model", "--dry-run",
    )
    assert result.returncode == 0, result.stderr
    manifest = json.loads(result.stdout)
    assert manifest["scope"]["tasks"] == 1
    assert manifest["scope"]["queue_questions"] == 1
    assert not output.exists()


def test_plugin_marketplace_and_hook_settings_are_portable() -> None:
    marketplace = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text())
    assert marketplace["name"] == "contra"
    plugin = marketplace["plugins"][0]
    assert plugin["source"] == "./plugin"
    assert (ROOT / plugin["source"] / ".claude-plugin" / "plugin.json").is_file()
    hooks = json.loads((ROOT / "plugin" / "hooks" / "hooks.json").read_text())
    assert set(hooks["hooks"]) == {"UserPromptSubmit", "Stop"}
    settings = json.loads((ROOT / "plugin" / "hook_settings_eval.json").read_text())
    assert settings["env"]["CONTRA_SYNC_VERIFY"] == "1"
    for event in settings["hooks"].values():
        for entry in event:
            for hook in entry["hooks"]:
                assert "${CLAUDE_PLUGIN_ROOT}" in hook["command"]


def test_real_harness_preflight_requires_explicit_confirmation() -> None:
    result = command("code/harness/run_harness.py", "--preflight")
    assert result.returncode == 2
    assert "without --confirm" in result.stderr


def test_codegen_requires_confirmation_before_reading_any_private_input(tmp_path) -> None:
    result = command(
        "code/evaluation/codegen/run_contextual_pass1_generation.py",
        "--transcript-manifest", str(tmp_path / "missing-manifest.json"),
        "--model", "test-model", "--base-url", "https://api.example.invalid/v1",
        "--api-key-file", str(tmp_path / "missing-key"),
        "--out-dir", str(tmp_path / "outputs"),
    )
    assert result.returncode == 2
    assert "--confirm is required" in result.stderr
    assert not (tmp_path / "outputs").exists()


def test_harness_stub_runs_and_resumes_without_duplicate_rows(tmp_path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        "".join(json.dumps({
            "task_id": f"synthetic-{index}",
            "question_id": f"source-{index}",
            "modified_content": "Read an integer and print it.",
            "qa_pairs": [],
        }) + "\n" for index in range(2)),
        encoding="utf-8",
    )
    args = (
        "--stub", "--tasks", str(tasks), "--run-id", "release-smoke",
        "--arms", "native,ask_permitted", "--workers", "1",
    )
    first = command("code/harness/run_harness.py", *args, cwd=tmp_path)
    assert first.returncode == 0, first.stdout + first.stderr
    files = sorted((tmp_path / "results_stub" / "harness" / "release-smoke" / "rows").glob("*.jsonl"))
    before = {path.name: path.read_bytes() for path in files}
    rows = [json.loads(line) for path in files for line in path.read_text().splitlines()]
    assert len(rows) == 4
    assert len({row["cell_key"] for row in rows}) == 4
    assert all(row["stub"] is True for row in rows)
    second = command("code/harness/run_harness.py", *args, cwd=tmp_path)
    assert second.returncode == 0, second.stdout + second.stderr
    assert {path.name: path.read_bytes() for path in files} == before
