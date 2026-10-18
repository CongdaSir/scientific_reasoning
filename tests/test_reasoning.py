import asyncio
import io
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import httpx

from src.config import CodeConfig, Config
from src.reasoning.answer import extract_boxed
from src.reasoning.engine import ReasoningEngine
from src.runner import run_dataset, solve
from src.tools.code import CodeTool
from src.tools.registry import ToolRegistry


def completion(content="", calls=None, finish="stop", **extra):
    message = {"role": "assistant", "content": content, **extra}
    if calls:
        message["tool_calls"] = calls
    return {"choices": [{"message": message, "finish_reason": finish}], "usage": {"total_tokens": 100}}


def tool_call(name, arguments, call_id="call-1"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


class AnswerTests(unittest.TestCase):
    def test_balanced_latex_and_last_box(self):
        self.assertEqual(extract_boxed(r"\boxed{1} then \boxed{\frac{1}{2}\ \mathrm{J}}"), r"\frac{1}{2}\ \mathrm{J}")
        self.assertEqual(extract_boxed(r"\boxed{\{a,b\}}"), r"\{a,b\}")

    def test_incomplete_empty_and_reasoning_boxes(self):
        for text in (r"\boxed{\frac{1}{2}", r"\boxed{}", r"<think>\boxed{wrong}</think>", r"<think>\boxed{wrong}"):
            self.assertIsNone(extract_boxed(text))
        self.assertEqual(extract_boxed(r"\boxed{wrong}</think>\boxed{right}"), "right")


class WorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_code_boxed_over_http_protocol(self):
        requests = []

        def handler(request):
            payload = json.loads(request.content)
            if request.url.path == "/v1/tools/search":
                self.assertEqual(request.headers["Authorization"], "Bearer test-search-key")
                self.assertEqual(payload, {"query": "test mass", "count": 5, "need_content": False})
                return httpx.Response(200, json={"results": [{
                    "title": "Reference", "url": "https://example.org/reference",
                    "summary": "The mass is 2 kg.", "content": "", "snippet": "mass",
                }]})
            self.assertEqual(request.url.path, "/v1/chat/completions")
            self.assertEqual({t["function"]["name"] for t in payload["tools"]}, {"search", "code"})
            requests.append(payload)
            if len(requests) == 1:
                return httpx.Response(200, json=completion(calls=[tool_call("search", {"query": "test mass"})], reasoning_content="Need evidence."))
            if len(requests) == 2:
                result = json.loads(payload["messages"][-1]["content"])
                self.assertEqual(result["results"][0]["text"], "The mass is 2 kg.")
                self.assertEqual(payload["messages"][-1]["tool_call_id"], "call-1")
                return httpx.Response(200, json=completion(calls=[tool_call("code", {"code": "print(0.5 * 2 * 12**2)"}, "call-2")]))
            result = json.loads(payload["messages"][-1]["content"])
            self.assertEqual(result["stdout"].strip(), "144.0")
            self.assertEqual(payload["messages"][1]["content"], "Find the kinetic energy.")
            return httpx.Response(200, json=completion(r"The kinetic energy is \boxed{144\ \mathrm{J}}."))

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-search-key"}):
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
                result = await solve("Find the kinetic energy.", Config(), http)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["answer"], r"144\ \mathrm{J}")
        self.assertEqual(result["rounds"], 3)
        self.assertEqual(result["trace"][0]["message"]["reasoning_content"], "Need evidence.")
        self.assertEqual([m["role"] for m in result["messages"]], ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"])

    async def test_multiple_tools_execute_before_final_detection(self):
        class Model:
            async def complete(self, messages, tools):
                if messages[-1]["role"] == "user":
                    return completion(r"Tentative \boxed{0}", [tool_call("search", {"query": "a"}), tool_call("search", {"query": "b"}, "call-2")])
                return completion(r"\boxed{2}")

        seen = []

        async def search(query):
            seen.append(query)
            return {"value": query}

        result = await ReasoningEngine(Model(), ToolRegistry(search, None)).run("question")
        self.assertEqual(seen, ["a", "b"])
        self.assertEqual(result["answer"], "2")

    async def test_continuation_round_limit_and_length(self):
        class Model:
            def __init__(self, finish="stop"):
                self.finish = finish

            async def complete(self, messages, tools):
                return completion("Need more thought.", finish=self.finish)

        tools = ToolRegistry(None, None)
        result = await ReasoningEngine(Model(), tools, 2).run("question")
        self.assertEqual(result["status"], "max_rounds")
        self.assertEqual(result["rounds"], 2)
        self.assertEqual(result["messages"][-2]["role"], "user")
        result = await ReasoningEngine(Model("length"), tools, 2).run("question")
        self.assertEqual(result["status"], "max_tokens")
        self.assertEqual(result["rounds"], 1)

    async def test_tool_errors_return_to_model(self):
        tools = ToolRegistry(None, None)
        for call in (tool_call("missing", {}), {"function": {"name": "code", "arguments": "bad json"}}):
            self.assertIn("error", json.loads(await tools.execute(call)))

    async def test_dataset_concurrency_and_sample_error(self):
        config = Config()
        config.dataset.question_field = "problem"
        config.dataset.concurrency = 2
        active = peak = 0

        async def fake_solve(question, config, http):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return {"status": "completed", "answer": question}

        output = io.StringIO()
        records = [{"id": "a", "problem": "1"}, {"id": "bad"}, {"id": "b", "problem": "2"}]
        with patch("src.runner.solve", fake_solve):
            with self.assertLogs("src.runner", level="ERROR"):
                await run_dataset(records, config, output)
        results = {r["id"]: r for r in map(json.loads, output.getvalue().splitlines())}
        self.assertEqual(peak, 2)
        self.assertEqual(results["bad"]["status"], "error")
        self.assertEqual(results["b"]["answer"], "2")
        self.assertEqual(results["b"]["input"], records[2])


class CodeTests(unittest.IsolatedAsyncioTestCase):
    async def test_files_survive_but_variables_do_not(self):
        with tempfile.TemporaryDirectory() as workdir:
            tool = CodeTool(CodeConfig(), workdir)
            first = await tool("x = 7\nopen('value.txt', 'w').write(str(x))")
            second = await tool("print(open('value.txt').read())\nprint('x' in globals())")
        self.assertEqual(first["returncode"], 0)
        self.assertEqual(second["stdout"], "7\nFalse\n")

    async def test_error_timeout_truncation_and_credentials(self):
        with tempfile.TemporaryDirectory() as workdir:
            tool = CodeTool(CodeConfig(timeout=0.5, max_output_chars=20), workdir)
            result = await tool("raise ValueError('bad')")
            self.assertNotEqual(result["returncode"], 0)
            self.assertTrue(result["stderr"])
            result = await tool("while True: pass")
            self.assertTrue(result["timed_out"])
            result = await tool("print('x' * 100)")
            self.assertTrue(result["truncated"])
            self.assertEqual(len(result["stdout"]), 20)
            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "secret"}):
                result = await tool("import os; print(os.getenv('DEEPSEEK_API_KEY'))")
            self.assertEqual(result["stdout"], "None\n")


if __name__ == "__main__":
    unittest.main()
