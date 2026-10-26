from copy import deepcopy
import unittest

from src.evaluation.gpqa import parse_choice, prepare_records, score_result
from src.reasoning.engine import ReasoningEngine
from src.tools.registry import ToolRegistry


class GPQATests(unittest.TestCase):
    def test_choice_parser_and_failed_sample_scoring(self):
        for text in ("A", "(A)", "a", r"\text{A}", r"\mathrm{A}", "$A$", "Option A"):
            self.assertEqual(parse_choice(text), "A")
        for text in (None, "", "A or B", "AB", "1", "The answer might be A"):
            self.assertIsNone(parse_choice(text))
        self.assertTrue(score_result({"status": "completed", "answer": "A"}, "A")["is_correct"])
        self.assertFalse(score_result({"status": "max_tokens", "answer": "A"}, "A")["is_correct"])

    def test_row_order_option_mapping_and_no_reference_explanation_in_prompt(self):
        rows = [{
            "Question": f"Question {index}", "Correct Answer": f"Correct option {index}",
            "Incorrect Answer 1": "Wrong one", "Incorrect Answer 2": "Wrong two",
            "Incorrect Answer 3": "Wrong three", "Record ID": f"record-{index}",
            "Explanation": "SECRET_SOLUTION_MARKER",
        } for index in range(3)]
        records = prepare_records(rows, 2, 42)
        self.assertEqual(records, prepare_records(rows, 2, 42))
        all_records = prepare_records(rows, None, 42)
        self.assertEqual(len(all_records), len(rows))
        self.assertEqual(all_records, prepare_records(rows, len(rows), 42))
        self.assertEqual(all_records[:2], records)
        for invalid_limit in (0, -1, len(rows) + 1):
            with self.assertRaises(ValueError):
                prepare_records(rows, invalid_limit, 42)
        with self.assertRaises(ValueError):
            prepare_records([], None, 42)
        self.assertEqual([row["source_record_id"] for row in records], ["record-0", "record-1"])
        for index, row in enumerate(records):
            self.assertEqual(row["options"][row["correct_label"]], rows[index]["Correct Answer"])
            self.assertNotIn("SECRET_SOLUTION_MARKER", row["question"])


class ProgressTests(unittest.IsolatedAsyncioTestCase):
    async def test_partial_history_survives_a_later_model_failure(self):
        snapshots = []

        class Model:
            async def complete(self, messages, tools):
                if messages[-1]["role"] == "tool":
                    raise RuntimeError("model connection failed")
                return {"choices": [{"message": {
                    "role": "assistant", "content": "", "reasoning": "Need a calculation.",
                    "tool_calls": [{"id": "call-1", "type": "function", "function": {
                        "name": "code", "arguments": '{"code": "print(2)"}',
                    }}],
                }, "finish_reason": "tool_calls"}], "usage": {"total_tokens": 10}}

        async def code(code):
            return {"stdout": "2\n", "stderr": "", "returncode": 0, "timed_out": False}

        engine = ReasoningEngine(Model(), ToolRegistry(None, code), on_progress=lambda state: snapshots.append(deepcopy(state)))
        with self.assertRaisesRegex(RuntimeError, "model connection failed"):
            await engine.run("Compute a value")
        self.assertEqual(snapshots[0]["rounds"], 0)
        last = snapshots[-1]
        self.assertEqual(last["rounds"], 1)
        self.assertEqual(last["messages"][-1]["role"], "tool")
        self.assertIn('"stdout": "2\\n"', last["messages"][-1]["content"])
        self.assertEqual(last["trace"][0]["message"]["reasoning"], "Need a calculation.")
