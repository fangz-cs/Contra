from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from forkcast.ccb_v3_harness import (
    HarnessError,
    PLAN_SCHEMA,
    index_tasks,
    load_plan,
    read_jsonl,
    run_question_task,
    sha256_bytes,
    sha256_text,
)
from forkcast.ccb_v3_scoring import per_task, summarize


REPO = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = REPO / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_synthetic_interactive_dataset_census_and_projection(tmp_path, monkeypatch) -> None:
    builder = load_script("build_ccb_v3_interactive")
    source_path = tmp_path / "source.jsonl"
    data_path = tmp_path / "interactive.jsonl"
    manifest_path = tmp_path / "manifest.json"
    write_jsonl(source_path, [
        {
            "task_id": "synthetic-1",
            "question_id": "source-1",
            "modified_content": "Print a value.",
            "qa_pairs_v3": [
                {"question": "Use one line?", "answer": "Yes",
                 "status": "open", "behaviour": "observable"},
                {"question": "Already specified?", "answer": "Yes",
                 "status": "resolved_elsewhere", "behaviour": "observable"},
                {"question": "Choose a variable name?", "answer": "value",
                 "status": "open", "behaviour": "non_observable"},
            ],
        },
        {
            "task_id": "synthetic-2",
            "question_id": "source-2",
            "modified_content": "Print two values on one line.",
            "qa_pairs_v3": [],
        },
    ])
    for name, value in (
        ("EXPECTED_TASKS", 2),
        ("EXPECTED_GOLD", 1),
        ("EXPECTED_K0_TASKS", 1),
        ("EXPECTED_SOURCE_PROBLEMS", 2),
    ):
        monkeypatch.setattr(builder, name, value)
    manifest = builder.build(source_path, data_path, manifest_path)
    rows = read_jsonl(data_path)
    source = index_tasks(read_jsonl(source_path))

    assert len(rows) == 2
    assert len({row["question_id"] for row in rows}) == 2
    assert sum(row["k"] for row in rows) == 1
    assert sum(row["k"] == 0 for row in rows) == 1
    assert manifest["output_sha256"] == sha256_bytes(data_path.read_bytes())
    assert json.loads(manifest_path.read_text()) == manifest
    for row in rows:
        original = source[row["task_id"]]
        assert row["modified_content"] == original["modified_content"]
        expected = [
            gold["question"]
            for gold in original["qa_pairs_v3"]
            if gold["status"] == "open"
            and gold["behaviour"] == "observable"
        ]
        assert [gold["question"] for gold in row["qa_pairs"]] == expected


def test_projection_retains_zero_gold_tasks() -> None:
    builder = load_script("build_ccb_v3_interactive")
    rows = builder.project(
        [
            {
                "task_id": "task_1",
                "question_id": "p1",
                "modified_content": "S1",
                "keywords": [],
                "qa_pairs_v3": [
                    {
                        "question": "Q1?",
                        "answer": "A1",
                        "status": "open",
                        "behaviour": "observable",
                    }
                ],
            },
            {
                "task_id": "task_2",
                "question_id": "p2",
                "modified_content": "S2",
                "keywords": [],
                "qa_pairs_v3": [
                    {
                        "question": "Settled?",
                        "answer": "Yes",
                        "status": "resolved_elsewhere",
                        "behaviour": "observable",
                    }
                ],
            },
        ]
    )
    assert [row["k"] for row in rows] == [1, 0]
    assert rows[1]["qa_pairs"] == []


def make_plan(data_path: Path, tasks: dict[str, dict]) -> dict:
    entries = {}
    for task_id, task in tasks.items():
        entries[task_id] = {
            "question_id": task["question_id"],
            "statement_sha256": sha256_text(task["modified_content"]),
            "questions": [],
        }
    entries["task_1"]["questions"] = [
        {
            "candidate_id": "v_task_1::candidate_a",
            "text": "Which interpretation?",
            "queue_position": 0,
        }
    ]
    return {
        "schema_version": PLAN_SCHEMA,
        "method": "RITE",
        "benchmark_sha256": sha256_bytes(data_path.read_bytes()),
        "task_count": len(entries),
        "question_count": 1,
        "tasks": entries,
    }


def test_plan_is_gold_blind_and_bound_to_dataset(tmp_path: Path) -> None:
    data_path = tmp_path / "data.jsonl"
    rows = [
        {
            "task_id": "task_1",
            "question_id": "p1",
            "modified_content": "S1",
            "qa_pairs": [{"question": "Gold?", "answer": "A"}],
        },
        {
            "task_id": "task_2",
            "question_id": "p2",
            "modified_content": "S2",
            "qa_pairs": [],
        },
    ]
    write_jsonl(data_path, rows)
    tasks = index_tasks(rows)
    plan = make_plan(data_path, tasks)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    loaded = load_plan(plan_path, dataset_path=data_path, tasks=tasks)
    assert loaded["question_count"] == 1

    plan["tasks"]["task_1"]["questions"][0]["gold"] = "leak"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(HarnessError, match="Gold-bearing"):
        load_plan(plan_path, dataset_path=data_path, tasks=tasks)


class FakeInteract:
    SYSTEM_PROMPT = "official system"
    FOLLOWUP = "\nfollow up"

    @staticmethod
    def _judge_answer(
        _client,
        question,
        _qa_pairs,
        hit_questions,
        _judge_model,
        _consistency,
        _max_retries,
    ):
        if question == "first?":
            key = "gold one"
            if key in hit_questions:
                return 0, "already", None
            return 1, "answer one", key
        return 0, "not relevant", None


def test_question_runner_preserves_order_and_never_generates_code() -> None:
    task = {
        "task_id": "task_1",
        "question_id": "p1",
        "modified_content": "statement",
        "qa_pairs": [
            {"question": "gold one", "answer": "answer one"},
            {"question": "gold two", "answer": "answer two"},
        ],
    }
    questions = [
        {
            "candidate_id": f"c{index}",
            "text": text,
            "queue_position": index,
        }
        for index, text in enumerate(("first?", "second?", "third?"))
    ]
    record = run_question_task(
        interact=FakeInteract,
        client=object(),
        task=task,
        plan_entry={"questions": questions},
        judge_model="judge",
        max_turns=3,
        consistency=3,
    )

    assert record["hit_dict"] == {"0": 1, "1": 0}
    assert [row["candidate_id"] for row in record["asked_questions"]] == [
        "c0",
        "c1",
    ]
    assert record["n_questions_omitted_by_turn_budget"] == 1
    assert record["turns"] == 3
    assert record["logical_terminal_turn"] is True
    assert record["code_generation_evaluated"] is False
    assert record["final_code"] == ""


class FakeOfficialMetrics:
    @staticmethod
    def score_record(record):
        questions = record["turns"] - 1
        hits = sum(record["hit_dict"].values())
        return {
            "tkqr": hits / max(1, record["key_question_num"]),
            "ora": 1.0 if questions == record["key_question_num"] else 0.5,
        }


def test_scoring_uses_question_budget_and_dataset_gold() -> None:
    task = {
        "task_id": "task_1",
        "question_id": "p1",
        "modified_content": "statement",
        "qa_pairs": [{"question": "g1"}, {"question": "g2"}],
    }
    record = {
        "task_id": "task_1",
        "prompt": "statement",
        "hit_dict": {"0": 1, "1": 0, "2": 1},
        "finished": True,
        "key_question_num": 999,
    }
    row = per_task(
        record=record,
        task=task,
        official_metrics=FakeOfficialMetrics,
        question_budget=2,
    )
    assert row["K"] == 2
    assert row["hits"] == 1
    assert row["question_turns"] == 2
    summary = summarize([row])
    assert summary["KQR_micro"] == 0.5
