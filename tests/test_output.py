import argparse
import contextlib
from copy import deepcopy
import csv
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from src.cli import main
from src.config import Config
from src.evaluation.gpqa import evaluate, prepare_records
from src.output import create_run_directory
from src.runner import run_dataset


class DirectoryTests(unittest.TestCase):
    def test_timestamped_directory_and_output_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "outputs"
            with patch("src.output.OUTPUT_ROOT", root):
                directory = create_run_directory("test-task")
                self.assertEqual(directory.parent, root)
                self.assertRegex(directory.name, r"^test-task-\d{8}-\d{6}$")
                self.assertEqual(create_run_directory("unused", directory, resume=True), directory)
                with self.assertRaises(ValueError):
                    create_run_directory("../outside")
                with self.assertRaises(ValueError):
                    create_run_directory("test", Path(temporary) / "outside")
                with self.assertRaises(ValueError):
                    create_run_directory("test", resume=True)

    def test_single_question_cli_uses_compact_run_directory(self):
        async def fake_solve(question, config, http, *, on_progress):
            result = {"question": question, "status": "completed", "answer": "2", "rounds": 1,
                      "messages": [{"role": "assistant", "content": r"\boxed{2}"}], "trace": []}
            on_progress(result)
            return result

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "outputs"
            with patch("src.output.OUTPUT_ROOT", root), patch("src.runner.solve", fake_solve), \
                    patch("sys.argv", ["scientific-reasoning", "--question", "1+1", "--task-name", "addition"]), \
                    contextlib.redirect_stdout(io.StringIO()):
                main()
            directory, = root.iterdir()
            self.assertEqual({p.name for p in directory.iterdir()}, {"traces", "results.jsonl", "summary.json"})
            result = json.loads((directory / "results.jsonl").read_text())
            self.assertNotIn("messages", result)
            self.assertEqual(result["answer"], "2")
            self.assertEqual(json.loads((directory / result["trace_file"]).read_text())["question"], "1+1")


class OutputTests(unittest.IsolatedAsyncioTestCase):
    async def test_dataset_failure_keeps_partial_trace_without_duplicate_history(self):
        config = Config()
        config.dataset.concurrency = 1
        snapshots = []

        async def fake_solve(question, config, http, *, on_progress):
            state = {"question": question, "status": "running", "answer": None, "rounds": 1,
                     "messages": [{"role": "tool", "content": '{"stdout":"2"}'}],
                     "trace": [{"round": 1, "message": {"role": "assistant"}}]}
            on_progress(state)
            snapshots.append(json.loads((traces / "sample-001.json").read_text()))
            raise RuntimeError("later model failure")

        with tempfile.TemporaryDirectory() as temporary:
            traces = Path(temporary) / "traces"
            output = io.StringIO()
            with patch("src.runner.solve", fake_solve), self.assertLogs("src.runner", level="ERROR"):
                await run_dataset([{"id": "../unsafe-id", "question": "compute"}], config, output, traces_dir=traces)
            result = json.loads(output.getvalue())
            saved = json.loads((traces / "sample-001.json").read_text())
            self.assertEqual(snapshots[0]["status"], "running")
            self.assertEqual(saved["status"], "error")
            self.assertEqual(len(saved["messages"]), 1)
            self.assertEqual(saved["rounds"], 1)
            self.assertNotIn("messages", result)
            self.assertNotIn("trace", result)
            self.assertEqual(result["trace_file"], "traces/sample-001.json")

    async def test_gpqa_outputs_and_resume_preserve_scores_and_partial_failure(self):
        rows = [{"Question": f"Question {i}", "Correct Answer": f"Correct {i}",
                 "Incorrect Answer 1": "Wrong one", "Incorrect Answer 2": "Wrong two",
                 "Incorrect Answer 3": "Wrong three"} for i in range(2)]
        expected = prepare_records(rows, 2, 42)
        calls = []

        async def fake_solve(question, config, http, *, on_progress):
            calls.append(question)
            index = next(i for i, row in enumerate(expected) if row["question"] == question)
            state = {"question": question, "status": "running", "answer": None, "rounds": 1,
                     "messages": [{"role": "assistant", "content": "Partial reasoning"}],
                     "trace": [{"round": 1, "usage": {"prompt_tokens": 10, "completion_tokens": 5}}]}
            on_progress(deepcopy(state))
            if index == 1:
                raise RuntimeError("later model failure")
            return {**state, "status": "completed", "answer": expected[index]["correct_label"]}

        real_client = httpx.AsyncClient

        def client():
            return real_client(transport=httpx.MockTransport(lambda request: httpx.Response(
                200, json={"data": [{"id": "Qwen/Qwen3-8B"}]},
            )))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "outputs"
            dataset = Path(temporary) / "dataset.csv"
            with dataset.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            args = argparse.Namespace(csv=dataset, config=None, env_file=Path(temporary) / ".env",
                                      task_name=None, output_dir=None, resume=False, limit=None, seed=42)
            with patch("src.output.OUTPUT_ROOT", root), patch("src.evaluation.gpqa.solve", fake_solve), \
                    patch("src.evaluation.gpqa.httpx.AsyncClient", client), contextlib.redirect_stdout(io.StringIO()):
                await evaluate(args)
                directory, = root.iterdir()
                self.assertTrue(directory.name.startswith("gpqa-diamond-all-"))
                self.assertEqual({p.name for p in directory.iterdir()}, {"traces", "results.jsonl", "summary.json"})
                self.assertEqual(len(list((directory / "traces").iterdir())), 2)
                summary = json.loads((directory / "summary.json").read_text())
                self.assertEqual(summary["accuracy"], 0.5)
                self.assertEqual(summary["total"], len(rows))
                self.assertEqual(summary["run"]["identity"]["limit"], len(rows))
                results = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines()]
                for row in results:
                    self.assertNotIn("messages", row)
                    self.assertNotIn("trace", row)
                failed = json.loads((directory / results[1]["trace_file"]).read_text())
                self.assertEqual(failed["status"], "error")
                self.assertEqual(len(failed["messages"]), 1)
                self.assertFalse(failed["evaluation"]["is_correct"])
                # Simulate the saved configuration format from before API providers.
                legacy_summary = json.loads((directory / "summary.json").read_text())
                legacy_model = legacy_summary["run"]["identity"]["config"]["model"]
                legacy_model.pop("provider")
                legacy_model.pop("reasoning_effort")
                (directory / "summary.json").write_text(json.dumps(legacy_summary))
                args.output_dir, args.resume = directory, True
                args.limit = len(rows)
                await evaluate(args)
                self.assertEqual(len(calls), 2)
                self.assertEqual(json.loads((directory / "summary.json").read_text())["correct"], 1)
                args.seed = 43
                with self.assertRaisesRegex(ValueError, "settings differ"):
                    await evaluate(args)

    async def test_deepseek_gpqa_uses_remote_model_and_existing_scoring(self):
        row = {"Question": "Compute one plus one.", "Correct Answer": "2",
               "Incorrect Answer 1": "1", "Incorrect Answer 2": "3", "Incorrect Answer 3": "4"}
        expected = prepare_records([row], None, 42)[0]
        seen = []

        def handler(request):
            self.assertEqual(request.url.host, "api.deepseek.com")
            self.assertEqual(request.headers["Authorization"], "Bearer test-key")
            seen.append(request.url.path)
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": "deepseek-flash"}]})
            self.assertEqual(request.url.path, "/v1/chat/completions")
            payload = json.loads(request.content)
            self.assertEqual(payload["thinking"], {"type": "enabled"})
            self.assertEqual(payload["reasoning_effort"], "high")
            self.assertEqual(payload["messages"][1]["content"], expected["question"])
            self.assertEqual({t["function"]["name"] for t in payload["tools"]}, {"search", "code"})
            self.assertNotIn("chat_template_kwargs", payload)
            return httpx.Response(200, json={"choices": [{"message": {
                "role": "assistant", "content": r"\boxed{" + expected["correct_label"] + "}",
                "reasoning_content": "One plus one is two.",
            }, "finish_reason": "stop"}], "usage": {"prompt_tokens": 20, "completion_tokens": 10}})

        real_client = httpx.AsyncClient
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "outputs"
            dataset = Path(temporary) / "dataset.csv"
            with dataset.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            args = argparse.Namespace(csv=dataset, config=Path(__file__).resolve().parents[1] / "configs/deepseek.yaml",
                                      env_file=Path(temporary) / ".env", task_name="deepseek-gpqa",
                                      output_dir=None, resume=False, limit=None, seed=42)
            with patch("src.output.OUTPUT_ROOT", root), patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}), \
                    patch("src.evaluation.gpqa.httpx.AsyncClient", lambda: real_client(transport=httpx.MockTransport(handler))), \
                    contextlib.redirect_stdout(io.StringIO()):
                await evaluate(args)
            directory, = root.iterdir()
            summary = json.loads((directory / "summary.json").read_text())
            self.assertEqual(summary["accuracy"], 1.0)
            self.assertEqual(summary["run"]["identity"]["config"]["model"]["provider"], "deepseek")
            result = json.loads((directory / "results.jsonl").read_text())
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["answer"], expected["correct_label"])
            trace = json.loads((directory / result["trace_file"]).read_text())
            self.assertEqual(trace["messages"][-1]["reasoning_content"], "One plus one is two.")
            self.assertEqual(seen, ["/v1/models", "/v1/chat/completions"])


if __name__ == "__main__":
    unittest.main()
