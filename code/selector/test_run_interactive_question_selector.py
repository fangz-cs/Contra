from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = Path(__file__).with_name(
    "run_interactive_question_selector.py"
)
SPEC = importlib.util.spec_from_file_location(
    "interactive_question_selector", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def tool_response(candidate_id: str):
    function = SimpleNamespace(
        name=MODULE.SELECT_TOOL_NAME,
        arguments=json.dumps({"candidate_id": candidate_id}),
    )
    call = SimpleNamespace(function=function)
    message = SimpleNamespace(content=None, tool_calls=[call])
    choice = SimpleNamespace(message=message, finish_reason="tool_calls")
    return SimpleNamespace(
        id="response-id",
        model="test-model",
        choices=[choice],
        model_dump=lambda: {"usage": {"output_tokens": 1}},
    )


def finish_response():
    function = SimpleNamespace(
        name=MODULE.FINISH_TOOL_NAME,
        arguments="{}",
    )
    call = SimpleNamespace(function=function)
    message = SimpleNamespace(content=None, tool_calls=[call])
    choice = SimpleNamespace(message=message, finish_reason="tool_calls")
    return SimpleNamespace(
        id="response-id",
        model="test-model",
        choices=[choice],
        model_dump=lambda: {"usage": {"output_tokens": 1}},
    )


def response_trace(tool_name: str, arguments: dict) -> dict:
    return {
        "response": {
            "content": None,
            "tool_calls": [
                {
                    "name": tool_name,
                    "arguments": json.dumps(arguments, sort_keys=True),
                }
            ],
        }
    }


class InteractiveSelectorTest(unittest.TestCase):
    def test_prompt_is_concise_task_focused_and_benchmark_agnostic(self) -> None:
        prompt = MODULE.SELECTOR_SYSTEM_PROMPT
        normalized = " ".join(prompt.split())

        self.assertIn("receive a system prompt, a requirement", normalized)
        self.assertIn("conversation history", normalized)
        self.assertIn(
            "potentially relevant clarification questions",
            normalized,
        )
        self.assertIn(
            "Update the question set using the requirement and "
            "conversation history",
            normalized,
        )
        self.assertIn(
            "question is resolved when its answer is already determined",
            normalized,
        )
        self.assertIn(
            "prior answers make it inapplicable",
            normalized,
        )
        self.assertIn(
            "Select the most useful unresolved question",
            normalized,
        )
        self.assertIn(
            "no question remains unresolved",
            normalized,
        )
        self.assertNotIn("different implementations", normalized)
        self.assertNotIn("policy", prompt.lower())
        self.assertNotIn("value of information", prompt.lower())
        self.assertNotIn("marginal utility", prompt.lower())
        self.assertNotIn("unresolved task requirements", prompt)
        self.assertNotIn("ClarifyCodeBench", prompt)
        self.assertNotIn("benchmark", prompt.lower())
        self.assertNotIn("gold", prompt.lower())
        self.assertNotIn("do not", prompt.lower())
        self.assertNotIn("must", prompt.lower())
        self.assertNotIn("five", prompt.lower())

    def test_request_contains_only_dialogue_and_selection_context(self) -> None:
        conversation = [
            {"role": "system", "content": "Official CCB prompt."},
            {"role": "user", "content": "Write a program."},
            {
                "role": "assistant",
                "content": "[QUESTION]\nUse A or B?\n[/QUESTION]",
            },
            {
                "role": "user",
                "content": "Use A.\nPlease continue.",
            },
        ]
        messages = MODULE.build_selector_messages(
            task_system_prompt="Official CCB prompt.",
            conversation=conversation,
            remaining_questions=[
                {
                    "candidate_id": "c3",
                    "text": "Print spaces or newlines?",
                    "original_queue_position": 3,
                },
                {
                    "candidate_id": "c7",
                    "text": "Use ascending order?",
                    "original_queue_position": 7,
                }
            ],
            round_index=2,
        )
        payload = json.loads(messages[1]["content"])

        self.assertEqual(payload["system_prompt"], "Official CCB prompt.")
        self.assertEqual(payload["requirement"], "Write a program.")
        self.assertEqual(
            payload["conversation_history"], conversation[2:]
        )
        self.assertEqual(
            set(payload),
            {
                "system_prompt",
                "requirement",
                "conversation_history",
                "remaining_questions",
            },
        )
        self.assertEqual(
            payload["remaining_questions"],
            [
                {
                    "candidate_id": "c3",
                    "question": "Print spaces or newlines?",
                },
                {
                    "candidate_id": "c7",
                    "question": "Use ascending order?",
                },
            ],
        )
        self.assertNotIn("qa_pairs", messages[1]["content"])
        self.assertNotIn("matched", messages[1]["content"])
        self.assertNotIn("original_queue_position", messages[1]["content"])

    def test_request_accepts_a_frozen_selector_prompt(self) -> None:
        messages = MODULE.build_selector_messages(
            task_system_prompt="Official CCB prompt.",
            conversation=[
                {"role": "system", "content": "Official CCB prompt."},
                {"role": "user", "content": "Write a program."},
            ],
            remaining_questions=[
                {
                    "candidate_id": "c1",
                    "text": "Use standard input?",
                    "original_queue_position": 0,
                }
            ],
            round_index=1,
            selector_system_prompt="Frozen prompt variant.",
        )

        self.assertEqual(
            messages[0],
            {"role": "system", "content": "Frozen prompt variant."},
        )

    def test_tools_expose_queue_selection_and_dynamic_finish(self) -> None:
        tool = MODULE.build_selector_tool(
            [
                {"candidate_id": "c1"},
                {"candidate_id": "c2"},
            ]
        )
        parameters = tool["function"]["parameters"]

        self.assertEqual(parameters["required"], ["candidate_id"])
        self.assertEqual(
            parameters["properties"]["candidate_id"]["enum"],
            ["c1", "c2"],
        )
        self.assertNotIn("question_text", json.dumps(tool))
        finish = MODULE.build_finish_tool()
        self.assertEqual(
            finish["function"]["name"], MODULE.FINISH_TOOL_NAME
        )
        self.assertEqual(
            finish["function"]["parameters"]["properties"], {}
        )

    def test_parser_accepts_question_or_finish_action(self) -> None:
        self.assertEqual(
            MODULE.parse_selection_response(
                tool_response("c2"),
                allowed_candidate_ids=["c1", "c2"],
            ),
            {"action": "ask", "candidate_id": "c2"},
        )
        self.assertEqual(
            MODULE.parse_selection_response(
                finish_response(),
                allowed_candidate_ids=["c1", "c2"],
            ),
            {"action": "finish", "candidate_id": None},
        )
        with self.assertRaises(MODULE.SelectorResponseError):
            MODULE.parse_selection_response(
                tool_response("STOP"),
                allowed_candidate_ids=["c1", "c2"],
            )
        extra = tool_response("c1")
        extra.choices[0].message.tool_calls[0].function.arguments = (
            '{"candidate_id":"c1","question":"new text"}'
        )
        with self.assertRaises(MODULE.SelectorResponseError):
            MODULE.parse_selection_response(
                extra,
                allowed_candidate_ids=["c1", "c2"],
            )
        free_text = tool_response("c1")
        free_text.choices[0].message.content = "I selected c1."
        with self.assertRaises(MODULE.SelectorResponseError):
            MODULE.parse_selection_response(
                free_text,
                allowed_candidate_ids=["c1", "c2"],
            )

    def test_selector_forces_dynamic_candidate_enum(self) -> None:
        class Completions:
            def __init__(self):
                self.kwargs = None

            def create(self, **kwargs):
                self.kwargs = kwargs
                return tool_response("c2")

        completions = Completions()
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )
        selector = MODULE.DialogueQuestionSelector(
            client,
            model="test-model",
            max_tokens=128,
            reasoning_effort="low",
            max_retries=1,
        )
        result = selector.select(
            task_system_prompt="CCB",
            conversation=[
                {"role": "system", "content": "CCB"},
                {"role": "user", "content": "Task"},
            ],
            remaining_questions=[
                {
                    "candidate_id": "c1",
                    "text": "Question 1?",
                    "original_queue_position": 0,
                },
                {
                    "candidate_id": "c2",
                    "text": "Question 2?",
                    "original_queue_position": 1,
                },
            ],
            round_index=1,
        )

        self.assertEqual(result["candidate_id"], "c2")
        self.assertEqual(result["action"], "ask")
        self.assertEqual(completions.kwargs["tool_choice"], "required")
        enum = completions.kwargs["tools"][0]["function"]["parameters"][
            "properties"
        ]["candidate_id"]["enum"]
        self.assertEqual(enum, ["c1", "c2"])
        self.assertEqual(
            completions.kwargs["tools"][1]["function"]["name"],
            MODULE.FINISH_TOOL_NAME,
        )

    def test_selector_can_emit_responses_tool_schema(self) -> None:
        class Completions:
            def __init__(self):
                self.kwargs = None

            def create(self, **kwargs):
                self.kwargs = kwargs
                return tool_response("c2")

        completions = Completions()
        selector = MODULE.DialogueQuestionSelector(
            SimpleNamespace(
                chat=SimpleNamespace(completions=completions)
            ),
            model="test-model",
            max_tokens=128,
            reasoning_effort="low",
            max_retries=1,
            tool_schema="responses",
            temperature=None,
        )
        selector.select(
            task_system_prompt="CCB",
            conversation=[
                {"role": "system", "content": "CCB"},
                {"role": "user", "content": "Task"},
            ],
            remaining_questions=[
                {
                    "candidate_id": "c1",
                    "text": "Question 1?",
                    "original_queue_position": 0,
                },
                {
                    "candidate_id": "c2",
                    "text": "Question 2?",
                    "original_queue_position": 1,
                },
            ],
            round_index=1,
        )

        tools = completions.kwargs["tools"]
        self.assertEqual(tools[0]["name"], MODULE.SELECT_TOOL_NAME)
        self.assertEqual(tools[1]["name"], MODULE.FINISH_TOOL_NAME)
        self.assertNotIn("function", tools[0])
        self.assertNotIn("temperature", completions.kwargs)
        self.assertEqual(
            tools[0]["parameters"]["properties"]["candidate_id"]["enum"],
            ["c1", "c2"],
        )

    def test_selector_can_use_a_followup_prompt_after_an_answer(self) -> None:
        class Completions:
            def __init__(self):
                self.calls = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                return tool_response("c1")

        completions = Completions()
        selector = MODULE.DialogueQuestionSelector(
            SimpleNamespace(
                chat=SimpleNamespace(completions=completions)
            ),
            model="test-model",
            max_tokens=128,
            reasoning_effort="low",
            max_retries=1,
            system_prompt="Initial prompt.",
            followup_system_prompt="Followup prompt.",
        )
        queue = [
            {
                "candidate_id": "c1",
                "text": "Question 1?",
                "original_queue_position": 0,
            }
        ]
        selector.select(
            task_system_prompt="CCB",
            conversation=[
                {"role": "system", "content": "CCB"},
                {"role": "user", "content": "Task"},
            ],
            remaining_questions=queue,
            round_index=1,
        )
        selector.select(
            task_system_prompt="CCB",
            conversation=[
                {"role": "system", "content": "CCB"},
                {"role": "user", "content": "Task"},
                {"role": "assistant", "content": "Question 0?"},
                {"role": "user", "content": "Answer 0."},
            ],
            remaining_questions=queue,
            round_index=2,
        )

        self.assertEqual(
            completions.calls[0]["messages"][0]["content"],
            "Initial prompt.",
        )
        self.assertEqual(
            completions.calls[1]["messages"][0]["content"],
            "Followup prompt.",
        )

    def test_plan_validation_preserves_full_queue_and_lineage(self) -> None:
        statement = "Print one integer."
        digest = hashlib.sha256(statement.encode()).hexdigest()
        task = {
            "task_id": "task_1",
            "question_id": "q1",
            "modified_content": statement,
        }
        plan = {
            "schema_version": MODULE.PLAN_SCHEMA,
            "benchmark_sha256": "benchmark",
            "task_count": 1,
            "tasks": {
                "task_1": {
                    "question_id": "q1",
                    "statement_sha256": digest,
                    "questions": [
                        {
                            "candidate_id": f"c{index}",
                            "queue_position": index,
                            "text": f"Question {index}?",
                        }
                        for index in range(7)
                    ],
                }
            },
        }

        queues = MODULE.validate_frozen_plan(
            plan=plan,
            tasks=[task],
            benchmark_sha256="benchmark",
        )

        self.assertEqual(len(queues["task_1"]), 7)
        self.assertEqual(
            queues["task_1"][6]["candidate_id"], "c6"
        )
        changed = json.loads(json.dumps(plan))
        changed["tasks"]["task_1"]["questions"][0]["text"] = (
            "[QUESTION] rewritten [/QUESTION]"
        )
        with self.assertRaises(MODULE.SelectorError):
            MODULE.validate_frozen_plan(
                plan=changed,
                tasks=[task],
                benchmark_sha256="benchmark",
            )

    def test_completion_checks_replay_question_and_tool_identity(self) -> None:
        queues = {
            "task_1": (
                {
                    "candidate_id": "c1",
                    "text": "Frozen question?",
                    "original_queue_position": 0,
                },
                {
                    "candidate_id": "c2",
                    "text": "Second frozen question?",
                    "original_queue_position": 1,
                },
            )
        }
        row = {
            "task_id": "task_1",
            "selector_stop_action_available": True,
            "selector_question_generation_available": False,
            "selector_stop_reason": "selector_finish",
            "harness_question_round_limit": (
                MODULE.HARNESS_QUESTION_ROUND_LIMIT
            ),
            "initial_queue_size": 2,
            "remaining_candidate_ids": ["c2"],
            "asked_questions": [
                {
                    "candidate_id": "c1",
                    "question": "Frozen question?",
                    "original_queue_position": 0,
                    "render": "frozen_queue_verbatim",
                }
            ],
            "selector_decisions": [
                {
                    "round": 1,
                    "action": "ask",
                    "candidate_id": "c1",
                    "trace": response_trace(
                        MODULE.SELECT_TOOL_NAME,
                        {"candidate_id": "c1"},
                    ),
                },
                {
                    "round": 2,
                    "action": "finish",
                    "candidate_id": None,
                    "trace": response_trace(
                        MODULE.FINISH_TOOL_NAME,
                        {},
                    ),
                },
            ],
        }
        self.assertTrue(
            MODULE.questions_are_frozen_verbatim(
                row=row, frozen_queues=queues
            )
        )
        self.assertTrue(
            MODULE.selector_decisions_are_valid(
                row=row,
                frozen_queues=queues,
            )
        )

        rewritten = json.loads(json.dumps(row))
        rewritten["asked_questions"][0]["question"] = "Rewritten?"
        self.assertFalse(
            MODULE.questions_are_frozen_verbatim(
                row=rewritten, frozen_queues=queues
            )
        )
        mismatched_tool = json.loads(json.dumps(row))
        mismatched_tool["selector_decisions"][0]["trace"]["response"][
            "tool_calls"
        ][0]["arguments"] = '{"candidate_id":"STOP"}'
        self.assertFalse(
            MODULE.selector_decisions_are_valid(
                row=mismatched_tool,
                frozen_queues=queues,
            )
        )
        impossible_finish = json.loads(json.dumps(row))
        impossible_finish["initial_queue_size"] = 1
        impossible_finish["remaining_candidate_ids"] = []
        self.assertFalse(
            MODULE.selector_decisions_are_valid(
                row=impossible_finish,
                frozen_queues={"task_1": queues["task_1"][:1]},
            )
        )

    def test_final_code_parser_rejects_surrounding_question(self) -> None:
        def extract_code(content: str) -> str:
            if "[CODE]" not in content or "[/CODE]" not in content:
                return ""
            return (
                content.split("[CODE]", 1)[1]
                .split("[/CODE]", 1)[0]
                .strip()
            )

        self.assertEqual(
            MODULE.exact_code_block(
                "[CODE]\nprint(0)\n[/CODE]",
                extract_code=extract_code,
            ),
            "print(0)",
        )
        self.assertIsNone(
            MODULE.exact_code_block(
                "Should I use zero?\n[CODE]\nprint(0)\n[/CODE]",
                extract_code=extract_code,
            )
        )
        self.assertIsNone(
            MODULE.exact_code_block(
                "print(0)",
                extract_code=extract_code,
            )
        )

    def test_non_code_final_response_is_terminal_model_outcome(self) -> None:
        class FakeSelector:
            model = "test-model"

            @staticmethod
            def select(**_kwargs):
                return {
                    "action": "finish",
                    "candidate_id": None,
                    "trace": response_trace(MODULE.FINISH_TOOL_NAME, {}),
                }

        class FakeInteract:
            FOLLOWUP = "\nContinue."

        class FakePrompts:
            SYSTEM_PROMPT = "Official CCB prompt."

        def extract_code(content: str) -> str:
            if content.startswith("[CODE]") and content.endswith("[/CODE]"):
                return content[6:-7].strip()
            return ""

        raw_response = (
            "[QUESTION]\nShould I read from standard input?\n[/QUESTION]"
        )
        record = MODULE.run_task(
            harness=(
                FakeInteract,
                FakePrompts,
                extract_code,
                lambda content: content,
                lambda content: bool(extract_code(content)),
            ),
            task={
                "task_id": "task_1",
                "question_id": "q1",
                "modified_content": "Print zero.",
                "qa_pairs": [],
            },
            frozen_queue=[
                {
                    "candidate_id": "c1",
                    "text": "Use standard input?",
                    "original_queue_position": 0,
                }
            ],
            selector=FakeSelector(),
            generate_final=lambda _messages: (
                raw_response,
                {"ok": True},
            ),
            judge_client=None,
            judge_model="gpt-4o",
            consistency=3,
            max_retries=1,
        )

        self.assertTrue(record["finished"])
        self.assertIsNone(record["failure_reason"])
        self.assertEqual(record["final_code"], "")
        self.assertEqual(
            record["final_response_status"], "no_exact_code_block"
        )
        self.assertEqual(record["response_contents"], [raw_response])
        self.assertEqual(
            record["selector_stop_reason"], "selector_finish"
        )

    def test_resume_rows_repairs_valid_tail_without_newline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.jsonl"
            path.write_bytes(b'{"sequence": 0}')

            self.assertEqual(
                MODULE._resume_rows(path),
                [{"sequence": 0}],
            )
            self.assertTrue(path.read_bytes().endswith(b"\n"))

            writer = MODULE.DurableWriter(path)
            writer.append({"sequence": 1})
            self.assertEqual(
                MODULE._resume_rows(path),
                [{"sequence": 0}, {"sequence": 1}],
            )

            torn = Path(temporary) / "torn.jsonl"
            torn.write_bytes(
                b'{"sequence": 0}\n{"content":"\xe4\xb8'
            )
            self.assertEqual(
                MODULE._resume_rows(torn),
                [{"sequence": 0}],
            )
            self.assertEqual(
                torn.read_bytes(),
                b'{"sequence": 0}\n',
            )

    def test_completion_canonicalizes_result_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "results.jsonl"
            manifest = root / "manifest.json"
            manifest.write_text("{}\n", encoding="utf-8")

            def row(task_id: str) -> dict:
                return {
                    "schema_version": MODULE.SCHEMA_RESULT,
                    "manifest_identity": "identity",
                    "task_id": task_id,
                    "finished": True,
                    "asked_questions": [],
                    "selector_decisions": [],
                    "selector_stop_action_available": True,
                    "selector_question_generation_available": False,
                    "selector_stop_reason": "queue_exhausted",
                    "harness_question_round_limit": (
                        MODULE.HARNESS_QUESTION_ROUND_LIMIT
                    ),
                    "initial_queue_size": 0,
                    "remaining_candidate_ids": [],
                }

            output.write_text(
                json.dumps(row("task_2"), sort_keys=True)
                + "\n"
                + json.dumps(row("task_1"), sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            self.assertTrue(
                MODULE.write_completion_if_full(
                    output_path=output,
                    manifest_path=manifest,
                    manifest_identity="identity",
                    expected_task_ids=["task_1", "task_2"],
                    frozen_queues={"task_1": (), "task_2": ()},
                )
            )
            task_ids = [
                json.loads(line)["task_id"]
                for line in output.read_text().splitlines()
            ]
            self.assertEqual(task_ids, ["task_1", "task_2"])

    def test_dialogue_selects_verbatim_questions_then_finishes(self) -> None:
        class FakeSelector:
            model = "test-model"

            def __init__(self):
                self.actions = [
                    ("ask", "c5"),
                    ("ask", "c4"),
                    ("finish", None),
                ]
                self.calls = []

            def select(self, **kwargs):
                self.calls.append(
                    json.loads(json.dumps(kwargs, ensure_ascii=False))
                )
                action, candidate_id = self.actions[len(self.calls) - 1]
                if action == "ask":
                    trace = response_trace(
                        MODULE.SELECT_TOOL_NAME,
                        {"candidate_id": candidate_id},
                    )
                else:
                    trace = response_trace(MODULE.FINISH_TOOL_NAME, {})
                return {
                    "action": action,
                    "candidate_id": candidate_id,
                    "trace": trace,
                }

        class FakeInteract:
            FOLLOWUP = "\nContinue."

            @staticmethod
            def _judge_answer(
                _client,
                model_question,
                _qa_pairs,
                _hit_questions,
                _judge_model,
                _consistency,
                _max_retries,
            ):
                return 0, f"Answer to {model_question}", None

        class FakePrompts:
            SYSTEM_PROMPT = "Official CCB prompt."

        def extract_question(content: str) -> str:
            return (
                content.replace("[QUESTION]", "")
                .replace("[/QUESTION]", "")
                .strip()
            )

        def extract_code(content: str) -> str:
            if "[CODE]" not in content or "[/CODE]" not in content:
                return ""
            return (
                content.split("[CODE]", 1)[1]
                .split("[/CODE]", 1)[0]
                .strip()
            )

        selector = FakeSelector()
        final_calls = []
        events = []

        def generate_final(messages):
            final_calls.append(json.loads(json.dumps(messages)))
            return "[CODE]\nprint(0)\n[/CODE]", {"ok": True}

        queue = [
            {
                "candidate_id": f"c{index}",
                "text": f"Frozen question {index}?",
                "original_queue_position": index,
            }
            for index in range(6)
        ]
        record = MODULE.run_task(
            harness=(
                FakeInteract,
                FakePrompts,
                extract_code,
                extract_question,
                lambda content: bool(extract_code(content)),
            ),
            task={
                "task_id": "task_1",
                "question_id": "q1",
                "modified_content": "Print zero.",
                "qa_pairs": [
                    {"question": "Hidden Gold", "answer": "Hidden answer"}
                ],
            },
            frozen_queue=queue,
            selector=selector,
            generate_final=generate_final,
            judge_client=object(),
            judge_model="gpt-4o",
            consistency=3,
            max_retries=1,
            event_sink=events.append,
        )

        self.assertTrue(record["finished"])
        self.assertEqual(record["selector_stop_reason"], "selector_finish")
        self.assertTrue(record["selector_stop_action_available"])
        self.assertFalse(record["selector_question_generation_available"])
        self.assertEqual(
            [row["candidate_id"] for row in record["asked_questions"]],
            ["c5", "c4"],
        )
        self.assertEqual(
            [row["question"] for row in record["asked_questions"]],
            [
                "Frozen question 5?",
                "Frozen question 4?",
            ],
        )
        self.assertEqual(
            record["remaining_candidate_ids"], ["c0", "c1", "c2", "c3"]
        )
        self.assertEqual(
            [decision["action"] for decision in record["selector_decisions"]],
            ["ask", "ask", "finish"],
        )
        self.assertTrue(
            MODULE.selector_decisions_are_valid(
                row=record,
                frozen_queues={"task_1": tuple(queue)},
            )
        )
        self.assertEqual(record["turns"], 3)
        self.assertEqual(len(selector.calls), 3)
        self.assertEqual(len(final_calls), 1)
        self.assertIn(
            "Answer to Frozen question 5?",
            selector.calls[1]["conversation"][-1]["content"],
        )
        self.assertEqual(
            [
                row["candidate_id"]
                for row in selector.calls[1]["remaining_questions"]
            ],
            ["c0", "c1", "c2", "c3", "c4"],
        )
        self.assertEqual(
            final_calls[0][-1]["content"], MODULE.CODE_NUDGE
        )
        self.assertEqual(
            [event["event"] for event in events],
            [
                "selection",
                "answer",
                "selection",
                "answer",
                "selection",
                "final_generation",
            ],
        )

        class NeverSelector:
            model = "test-model"

            def select(self, **_kwargs):
                raise AssertionError("saved selector calls must be replayed")

        class NeverJudge(FakeInteract):
            @staticmethod
            def _judge_answer(*_args):
                raise AssertionError("saved judge calls must be replayed")

        replay = MODULE.run_task(
            harness=(
                NeverJudge,
                FakePrompts,
                extract_code,
                extract_question,
                lambda content: bool(extract_code(content)),
            ),
            task={
                "task_id": "task_1",
                "question_id": "q1",
                "modified_content": "Print zero.",
                "qa_pairs": [
                    {"question": "Hidden Gold", "answer": "Hidden answer"}
                ],
            },
            frozen_queue=queue,
            selector=NeverSelector(),
            generate_final=lambda _messages: (_ for _ in ()).throw(
                AssertionError("saved final generation must be replayed")
            ),
            judge_client=object(),
            judge_model="gpt-4o",
            consistency=3,
            max_retries=1,
            resume_events=events,
            event_sink=lambda _event: (_ for _ in ()).throw(
                AssertionError("replay must not append duplicate events")
            ),
        )
        self.assertTrue(replay["finished"])
        self.assertEqual(replay["messages"], record["messages"])
        self.assertEqual(
            replay["asked_questions"], record["asked_questions"]
        )
        self.assertEqual(
            replay["selector_decisions"], record["selector_decisions"]
        )
        self.assertEqual(replay["hit_dict"], record["hit_dict"])

        class AlwaysAskSelector:
            model = "test-model"

            def __init__(self):
                self.calls = []

            def select(self, **kwargs):
                self.calls.append(kwargs)
                candidate_id = f"c{len(self.calls) - 1}"
                return {
                    "action": "ask",
                    "candidate_id": candidate_id,
                    "trace": response_trace(
                        MODULE.SELECT_TOOL_NAME,
                        {"candidate_id": candidate_id},
                    ),
                }

        capped_selector = AlwaysAskSelector()
        capped = MODULE.run_task(
            harness=(
                FakeInteract,
                FakePrompts,
                extract_code,
                extract_question,
                lambda content: bool(extract_code(content)),
            ),
            task={
                "task_id": "task_1",
                "question_id": "q1",
                "modified_content": "Print zero.",
                "qa_pairs": [],
            },
            frozen_queue=queue,
            selector=capped_selector,
            generate_final=lambda _messages: (
                "[CODE]\nprint(0)\n[/CODE]",
                {},
            ),
            judge_client=object(),
            judge_model="gpt-4o",
            consistency=3,
            max_retries=1,
        )
        self.assertEqual(
            capped["selector_stop_reason"], "harness_round_limit"
        )
        self.assertEqual(len(capped_selector.calls), 5)
        self.assertEqual(len(capped["asked_questions"]), 5)
        self.assertEqual(capped["remaining_candidate_ids"], ["c5"])
        self.assertTrue(
            MODULE.selector_decisions_are_valid(
                row=capped,
                frozen_queues={"task_1": tuple(queue)},
            )
        )

    def test_empty_queue_goes_directly_to_code_without_selector_call(self) -> None:
        class NeverSelector:
            model = "test-model"

            def select(self, **_kwargs):
                raise AssertionError("selector must not run")

        class FakeInteract:
            FOLLOWUP = ""

            @staticmethod
            def _judge_answer(*_args):
                raise AssertionError("judge must not run")

        class FakePrompts:
            SYSTEM_PROMPT = "CCB"

        def extract_code(content: str) -> str:
            if "[CODE]" not in content:
                return ""
            return content.split("[CODE]", 1)[1].split("[/CODE]", 1)[0].strip()

        record = MODULE.run_task(
            harness=(
                FakeInteract,
                FakePrompts,
                extract_code,
                lambda _content: "",
                lambda content: bool(extract_code(content)),
            ),
            task={
                "task_id": "task_1",
                "question_id": "q1",
                "modified_content": "Print zero.",
                "qa_pairs": [],
            },
            frozen_queue=[],
            selector=NeverSelector(),
            generate_final=lambda _messages: (
                "[CODE]\nprint(0)\n[/CODE]",
                {},
            ),
            judge_client=object(),
            judge_model="gpt-4o",
            consistency=3,
            max_retries=1,
        )

        self.assertTrue(record["finished"])
        self.assertEqual(record["asked_questions"], [])
        self.assertEqual(record["selector_stop_reason"], "queue_exhausted")
        self.assertEqual(record["turns"], 1)


if __name__ == "__main__":
    unittest.main()
