import argparse
import contextlib
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from src.evaluation.gpqa import parse_choice
from src.evaluation.supergpqa import evaluate, prepare_records


PROJECT = Path(__file__).resolve().parents[1]


def source_rows():
    return [{"uuid": f"source-{i}", "question": f"Question {i}",
             "options": [f"Option text {j}" for j in range(10)],
             "answer_letter": "J", "answer": "REFERENCE_ONLY_SENTINEL", "discipline": "Science"}
            for i in range(3)]


class PreparationTests(unittest.TestCase):
    def test_zero_based_source_range_options_and_no_reference_leakage(self):
        rows = source_rows()
        self.assertEqual([r["index"] for r in prepare_records(rows)], [0, 1, 2])
        selected = prepare_records(rows, 1, end_index=3)
        self.assertEqual([r["index"] for r in selected], [1, 2])
        self.assertEqual([r["id"] for r in selected], ["source-1", "source-2"])
        self.assertEqual(prepare_records(rows, 1, limit=1), selected[:1])
        self.assertEqual(prepare_records(rows, 2, limit=100)[0]["index"], 2)
        for row in selected:
            self.assertIn("A. Option text 0", row["question"])
            self.assertIn("J. Option text 9", row["question"])
            self.assertNotIn("REFERENCE_ONLY_SENTINEL", row["question"])
            self.assertEqual(row["correct_label"], "J")
            self.assertEqual(row["input"]["answer"], "REFERENCE_ONLY_SENTINEL")

    def test_invalid_indices_limits_and_labels_are_rejected(self):
        rows = source_rows()
        for kwargs in ({"start_index": -1}, {"start_index": 3}, {"end_index": 0},
                       {"end_index": 4}, {"limit": 0}, {"limit": -1}, {"end_index": 2, "limit": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                prepare_records(rows, **kwargs)
        with self.assertRaises(ValueError):
            prepare_records([])
        rows[0]["answer_letter"] = "K"
        with self.assertRaisesRegex(ValueError, "answer_letter"):
            prepare_records(rows)

    def test_answer_parser_supports_all_present_options_and_preserves_gpqa_rules(self):
        for answer in ("J", "j", "Option J", r"\text{J}", "(J)"):
            self.assertEqual(parse_choice(answer, "ABCDEFGHIJ"), "J")
        for answer in ("K", "J or A", "JJ", None):
            self.assertIsNone(parse_choice(answer, "ABCDEFGHIJ"))
        self.assertIsNone(parse_choice("J"))
        self.assertEqual(parse_choice("A"), "A")


class GenerationTests(unittest.IsolatedAsyncioTestCase):
    def args(self, temporary):
        path = Path(temporary) / "data.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in source_rows()))
        return argparse.Namespace(input=path, config=PROJECT / "configs/deepseek.yaml",
                                  env_file=Path(temporary) / ".env", start_index=1, end_index=3,
                                  limit=None, concurrency=1, task_name="supergpqa-test",
                                  output_dir=None, resume=False, dry_run=False)

    async def test_trajectory_checkpoints_failure_and_resume_keep_absolute_indices(self):
        calls = []
        failing = True

        async def fake_solve(question, config, http, *, on_progress):
            self.assertNotIn("REFERENCE_ONLY_SENTINEL", question)
            self.assertEqual(config.model.provider, "deepseek")
            calls.append(question)
            state = {"question": question, "status": "running", "answer": None, "rounds": 1,
                     "messages": [{"role": "assistant", "content": "", "reasoning_content": "Need code.",
                                   "tool_calls": [{"id": "call-1", "type": "function", "function": {
                                       "name": "code", "arguments": '{"code":"1+1"}'}}]},
                                  {"role": "tool", "tool_call_id": "call-1", "content": '{"stdout":"2\\n"}'}],
                     "trace": [{"round": 1, "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}]}
            on_progress(state)
            saved = json.loads((output_root / "supergpqa-test-20261025-120000" / "traces" /
                                ("sample-000002.json" if "Question 2" in question else "sample-000001.json")).read_text())
            self.assertEqual(saved["status"], "running")
            if failing and "Question 2" in question:
                raise RuntimeError("later model failure")
            return {**state, "status": "completed", "answer": "J"}

        real_client = httpx.AsyncClient

        def handler(request):
            self.assertEqual(str(request.url), "https://api.deepseek.com/v1/models")
            self.assertEqual(request.headers["Authorization"], "Bearer test-key")
            return httpx.Response(200, json={"data": [{"id": "deepseek-flash"}]})

        with tempfile.TemporaryDirectory() as temporary:
            args = self.args(temporary)
            output_root = Path(temporary) / "outputs"
            args.output_dir = output_root / "supergpqa-test-20261025-120000"
            with patch("src.output.OUTPUT_ROOT", output_root), \
                    patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}), \
                    patch("src.evaluation.supergpqa.httpx.AsyncClient", lambda: real_client(transport=httpx.MockTransport(handler))), \
                    patch("src.evaluation.supergpqa.solve", fake_solve), contextlib.redirect_stdout(io.StringIO()):
                with self.assertLogs(level="ERROR"):
                    output = await evaluate(args)
                summary = json.loads((output / "summary.json").read_text())
                self.assertEqual(summary["finished"], 2)
                self.assertEqual(summary["status_counts"], {"completed": 1, "error": 1})
                self.assertEqual(summary["usage"]["total_tokens"], 30)
                self.assertEqual(summary["active_indices"], [])
                results = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
                self.assertEqual([row["index"] for row in results], [1, 2])
                for row in results:
                    self.assertNotIn("messages", row)
                    self.assertNotIn("trace", row)
                    self.assertNotIn("input", row)
                failed_path = output / "traces/sample-000002.json"
                failed = json.loads(failed_path.read_text())
                self.assertEqual(failed["index"], 2)
                self.assertEqual(failed["status"], "error")
                self.assertEqual(failed["messages"][-1]["role"], "tool")
                self.assertEqual(failed["messages"][0]["reasoning_content"], "Need code.")
                self.assertEqual(failed["input"]["uuid"], "source-2")
                args.resume = True
                await evaluate(args)
                self.assertEqual(len(calls), 2)
                # An unfinished checkpoint is restarted, while final results are skipped.
                failed["status"] = "running"
                failed.pop("evaluation")
                failed_path.write_text(json.dumps(failed))
                failing = False
                await evaluate(args)
                self.assertEqual(len(calls), 3)
                summary = json.loads((output / "summary.json").read_text())
                self.assertEqual(summary["correct"], 2)
                self.assertEqual(summary["status"], "completed")
                self.assertEqual(summary["status_counts"], {"completed": 2})
                self.assertEqual(len((output / "results.jsonl").read_text().splitlines()), 2)
                args.start_index = 0
                with self.assertRaisesRegex(ValueError, "Cannot resume"):
                    await evaluate(args)

    async def test_dry_run_validates_without_network_or_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.args(temporary)
            args.dry_run = True
            captured = io.StringIO()
            with patch("src.evaluation.supergpqa.httpx.AsyncClient", side_effect=AssertionError("No network")), \
                    patch("src.evaluation.supergpqa.create_run_directory", side_effect=AssertionError("No output")), \
                    contextlib.redirect_stdout(captured):
                self.assertIsNone(await evaluate(args))
            summary = json.loads(captured.getvalue())
            self.assertEqual(summary["start_index"], 1)
            self.assertEqual(summary["end_index_exclusive"], 3)
            self.assertEqual(summary["total"], 2)
            self.assertEqual(summary["max_tokens"], 393216)


if __name__ == "__main__":
    unittest.main()
