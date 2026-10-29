import asyncio
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from no_tools_eval.datasets import messages, numeric_target, prepare_dataset
from no_tools_eval.runner import ROOT, evaluate, metrics, parser, request_payload, solve_record
from no_tools_eval.scoring import grade, number, parse_answer


def frames(*packets, done=True):
    body = "".join("data: " + json.dumps(packet) + "\n\n" for packet in packets)
    return body + ("data: [DONE]\n\n" if done else "")


def completion(content=r"\boxed{B}", finish="stop"):
    return frames(
        {"id": "local-test", "choices": [{"delta": {"reasoning": r"An intermediate \boxed{A}."}}]},
        {"choices": [{"delta": {"content": content}, "finish_reason": finish}]},
        {"choices": [], "usage": {"prompt_tokens": 25, "completion_tokens": 12, "total_tokens": 37}},
    )


class DatasetTests(unittest.TestCase):
    def test_real_dataset_selection_preserves_order_and_caps(self):
        expected = {"gpqa-diamond": 198, "rbench": 500, "scieval": 500, "scibench": 500}
        for name, count in expected.items():
            with self.subTest(name=name):
                records, metadata = prepare_dataset(ROOT, name)
                self.assertEqual(len(records), count)
                self.assertEqual(metadata["selected"], count)
                self.assertEqual([r["index"] for r in records], list(range(count)))
                self.assertTrue(all(r["reference"] is not None for r in records))
                if name == "scibench":
                    self.assertTrue(all(not r["source_file"].endswith("_sol.json") for r in records))
                    self.assertEqual(records[0]["source_file"], "datasets/scibench/dataset/original/atkins.json")
                if name == "scieval":
                    self.assertEqual({r["task_name"] for r in records}, {"MedQA"})

    def test_gpqa_shuffles_options_only_and_never_sends_explanations(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "datasets/gpqa").mkdir(parents=True)
            rows = [{"Question": f"Question {i}", "Correct Answer": f"correct {i}",
                     "Incorrect Answer 1": "wrong 1", "Incorrect Answer 2": "wrong 2", "Incorrect Answer 3": "wrong 3",
                     "Explanation": "SECRET_EXPLANATION", "Canary String": "SECRET_METADATA"} for i in range(3)]
            with (root / "datasets/gpqa/gpqa_diamond.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            records, metadata = prepare_dataset(root, "gpqa-diamond", limit=2)
            self.assertEqual(metadata["available"], 3)
            self.assertEqual(records, prepare_dataset(root, "gpqa-diamond", limit=2)[0])
            for i, record in enumerate(records):
                self.assertIn(f"Question {i}", record["question"])
                self.assertEqual(record["options"][record["reference"]], f"correct {i}")
                self.assertNotIn("SECRET", json.dumps(messages(record)))

    def test_scaled_units_and_limits(self):
        for unit in [r"$10^{-19} \mathrm{J}$", r"$\mathrm{10^{-19}} \mathrm{J}$"]:
            value, clean = numeric_target("3.52", unit)
            self.assertAlmostEqual(value / 1e-19, 3.52)
            self.assertEqual(clean, r"\mathrm{J}")
        self.assertEqual(numeric_target("7", r"$\mathrm{10^7} \mathrm{km}$"), (7e7, r"\mathrm{km}"))
        for limit in (0, 501):
            with self.assertRaises(ValueError):
                prepare_dataset(ROOT, "gpqa-diamond", limit=limit)


class GradingTests(unittest.TestCase):
    def test_only_final_answer_is_used(self):
        self.assertEqual(parse_answer(r"<think>\boxed{A}</think>Finally \boxed{\text{B}}", "choice"), "B")
        self.assertIsNone(parse_answer(r"<think>Still thinking \boxed{A}", "choice"))
        self.assertEqual(parse_answer('{"answer":"F"}', "choice"), "F")
        self.assertIsNone(parse_answer(r"\boxed{A or B}", "choice"))

    def test_numeric_formats_and_safe_arithmetic(self):
        samples = {r"3.52\times10^{-19}": 3.52e-19, r"\frac{1}{2}": 0.5,
                   r"\sqrt{4}": 2, "89,034.79": 89034.79, "−1.642876": -1.642876, "1.2e+4": 12000}
        for text, expected in samples.items():
            self.assertTrue(abs(number(text) / expected - 1) < 1e-10)
        for text in ["__import__('os').system('false')", "10**1000000", "nan", "inf", "1/0"]:
            self.assertIsNone(number(text))

    def test_tolerance_and_truncation_are_scored_correctly(self):
        numeric = {"kind": "numeric", "reference": 1e-23}
        self.assertTrue(grade(numeric, r"\boxed{1.04e-23}", "completed")["is_correct"])
        self.assertFalse(grade(numeric, r"\boxed{0}", "completed")["is_correct"])
        self.assertFalse(grade(numeric, r"\boxed{1e-23}", "max_tokens")["is_correct"])
        choice = {"kind": "choice", "options": {"A": "one", "B": "two"}, "reference": "B"}
        self.assertFalse(grade(choice, r"\boxed{B}", "error")["is_correct"])

    def test_invalid_source_reference_is_retained_but_not_counted_as_wrong(self):
        record = {"kind": "choice", "reference": "E\n\nF", "options": dict.fromkeys("ABCDEF", "option")}
        evaluation = grade(record, r"\boxed{E}", "completed")
        self.assertIsNone(evaluation["is_correct"])
        self.assertEqual(evaluation["reference"], "E\n\nF")
        rows = [{"status": "completed", "evaluation": evaluation},
                {"status": "completed", "evaluation": {"is_correct": True, "parsed": True}}]
        summary = metrics(rows, 2, scorable_total=1)
        self.assertEqual(summary["data_errors"], 1)
        self.assertEqual(summary["finished"], 2)
        self.assertEqual(summary["scored"], 1)
        self.assertEqual(summary["accuracy_on_finished"], 1)


class StreamingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.output = Path(self.folder.name)
        for name in ("traces", "streams"):
            (self.output / "gpqa-diamond" / name).mkdir(parents=True)
        self.args = parser().parse_args([])
        self.args.tokenizer_path = None
        self.args.retries = 0
        self.args.max_tokens = 100
        self.record = {"id": "gpqa-diamond-000000", "index": 0, "dataset": "gpqa-diamond", "kind": "choice",
                       "question": "What is 1 + 1? A. 1 B. 2", "reference": "B", "options": {"A": "1", "B": "2"}}

    async def test_tool_free_request_stream_and_reasoning_are_preserved(self):
        def handler(request):
            data = json.loads(request.content)
            self.assertFalse({"tools", "tool_choice", "functions", "function_call"} & data.keys())
            self.assertEqual(data["n"], 1)
            return httpx.Response(200, text=completion())

        payload = request_payload({**self.record, "reference": "PRIVATE_REFERENCE"}, self.args)
        self.assertNotIn("PRIVATE_REFERENCE", json.dumps(payload))
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            result = await solve_record(http, self.args, self.record, self.output)
        self.assertTrue(result["evaluation"]["is_correct"])
        trace = json.loads((self.output / result["trace_file"]).read_text())
        self.assertIn(r"\boxed{A}", trace["response"]["reasoning"])
        self.assertEqual(trace["response"]["content"], r"\boxed{B}")
        self.assertEqual(trace["response"]["usage"]["completion_tokens"], 12)
        stream = self.output / trace["attempts"][0]["stream_file"]
        self.assertIn("[DONE]", stream.read_text())

    async def test_disconnect_preserves_partial_output_and_new_attempt_does_not_overwrite(self):
        class BrokenStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield frames({"choices": [{"delta": {"reasoning_content": "partial reasoning"}}]}, done=False).encode()
                raise httpx.ReadError("simulated disconnect")

        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=BrokenStream()))) as http:
            first = await solve_record(http, self.args, self.record, self.output)
        trace_path = self.output / first["trace_file"]
        trace = json.loads(trace_path.read_text())
        original_stream = (self.output / trace["attempts"][0]["stream_file"]).read_bytes()
        self.assertEqual(trace["response"]["reasoning"], "partial reasoning")
        self.assertEqual(first["status"], "error")
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, text=completion()))) as http:
            second = await solve_record(http, self.args, self.record, self.output)
        trace = json.loads(trace_path.read_text())
        self.assertEqual(len(trace["attempts"]), 2)
        self.assertEqual((self.output / trace["attempts"][0]["stream_file"]).read_bytes(), original_stream)
        self.assertTrue(second["evaluation"]["is_correct"])

    async def test_length_finish_and_tool_calls_cannot_be_successes(self):
        for body, expected in [(completion(finish="length"), "max_tokens"),
                               (frames({"choices": [{"delta": {"tool_calls": [{"function": {"name": "code"}}]}}]}), "error")]:
            async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, text=body))) as http:
                result = await solve_record(http, self.args, self.record, self.output)
            self.assertEqual(result["status"], expected)
            self.assertFalse(result["evaluation"]["is_correct"])

    async def test_end_to_end_resume_skips_finished_samples_and_checks_identity(self):
        calls = []

        def handler(request):
            if request.url.path.endswith("/models"):
                return httpx.Response(200, json={"data": [{"id": self.args.model}]})
            calls.append(json.loads(request.content))
            return httpx.Response(200, text=completion())

        # Use real GPQA source rows and a small cap, with only the HTTP transport mocked.
        self.args.datasets = ["gpqa-diamond"]
        self.args.limit = 2
        self.args.output_dir = self.output
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        with patch("no_tools_eval.runner.httpx.AsyncClient", return_value=client):
            await evaluate(self.args)
        self.assertEqual(len(calls), 2)
        self.args.resume = True
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        with patch("no_tools_eval.runner.httpx.AsyncClient", return_value=client):
            await evaluate(self.args)
        self.assertEqual(len(calls), 2)
        summary = json.loads((self.output / "summary.json").read_text())
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["finished"], 2)
        self.assertEqual(len((self.output / "gpqa-diamond/results.jsonl").read_text().splitlines()), 2)
        self.args.max_tokens += 1
        with self.assertRaisesRegex(ValueError, "identical"):
            await evaluate(self.args)


if __name__ == "__main__":
    unittest.main()
