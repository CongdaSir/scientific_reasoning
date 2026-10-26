import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from src.clients import create_model_client
from src.clients.deepseek import DeepSeekClient
from src.clients.vllm import VLLMClient
from src.config import Config, ModelConfig
from src.tools.registry import TOOL_SCHEMAS


PROJECT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_legacy_local_and_deepseek_configs(self):
        default = Config()
        self.assertEqual(default.model.provider, "vllm")
        self.assertEqual(default.model.base_url, "http://localhost:8000/v1")
        self.assertEqual(default.model.name, "Qwen/Qwen3-8B")
        self.assertEqual(default.model.api_key_env, "VLLM_API_KEY")
        self.assertEqual(default.model.max_tokens, 8192)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.yaml"
            path.write_text("model:\n  base_url: http://localhost:8123/v1\n  name: local-qwen\n")
            legacy = Config.load(path)
            self.assertEqual(legacy.model.provider, "vllm")
            self.assertEqual(legacy.model.name, "local-qwen")
            self.assertEqual(legacy.model.base_url, "http://localhost:8123/v1")
            path.write_text("model:\n  provider: deepseek\n")
            remote = Config.load(path)
            self.assertEqual(remote.model.base_url, "https://api.deepseek.com/v1")
            self.assertEqual(remote.model.name, "deepseek-flash")
            self.assertEqual(remote.model.api_key_env, "DEEPSEEK_API_KEY")
            self.assertEqual(remote.model.max_tokens, 393216)
        local = Config.load(PROJECT / "configs/default.yaml")
        remote = Config.load(PROJECT / "configs/deepseek.yaml")
        self.assertEqual(local.model.provider, "vllm")
        self.assertEqual(remote.model.provider, "deepseek")
        self.assertTrue(remote.model.enable_thinking)
        self.assertEqual(remote.model.reasoning_effort, "high")
        self.assertEqual(remote.model.max_tokens, 393216)
        for section in ("search", "code", "reasoning", "dataset"):
            self.assertEqual(getattr(local, section), getattr(remote, section))

    def test_direct_configuration_overrides_and_invalid_provider(self):
        config = ModelConfig(provider="deepseek", base_url="https://example.org/v1", name="custom", api_key_env="MODEL_KEY", max_tokens=2048)
        self.assertEqual((config.base_url, config.name, config.api_key_env), ("https://example.org/v1", "custom", "MODEL_KEY"))
        self.assertEqual(config.max_tokens, 2048)
        with self.assertRaisesRegex(ValueError, "model.provider"):
            ModelConfig(provider="typo")


class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_factory_and_local_request_compatibility(self):
        def handler(request):
            payload = json.loads(request.content)
            self.assertEqual(str(request.url), "http://localhost:8000/v1/chat/completions")
            self.assertEqual(request.headers["Authorization"], "Bearer EMPTY")
            self.assertEqual(payload, {
                "model": "Qwen/Qwen3-8B", "messages": [], "tools": TOOL_SCHEMAS,
                "tool_choice": "auto", "max_tokens": 8192, "temperature": 0.6,
                "top_p": 0.95, "top_k": 20, "chat_template_kwargs": {"enable_thinking": True},
                "stream": False,
            })
            return httpx.Response(200, json={"choices": []})

        with patch.dict(os.environ, {}, clear=True):
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
                local = create_model_client(ModelConfig(), http)
                self.assertIsInstance(local, VLLMClient)
                self.assertIsInstance(create_model_client(ModelConfig(provider="deepseek"), http), DeepSeekClient)
                await local.complete([], TOOL_SCHEMAS)

    async def test_thinking_request_preserves_history_and_uses_api_parameters(self):
        messages = [
            {"role": "user", "content": "Compute a value"},
            {"role": "assistant", "content": "", "reasoning_content": "Use code.", "tool_calls": [
                {"id": "call-1", "type": "function", "function": {"name": "code", "arguments": '{"code":"1+1"}'}}
            ]},
            {"role": "tool", "tool_call_id": "call-1", "content": '{"stdout":"2\\n"}'},
        ]
        response = {"choices": [{"message": {"role": "assistant", "content": r"\boxed{2}", "reasoning_content": "Done."}, "finish_reason": "stop"}]}

        def handler(request):
            self.assertEqual(str(request.url), "https://api.deepseek.com/v1/chat/completions")
            self.assertEqual(request.headers["Authorization"], "Bearer test-model-key")
            self.assertEqual(request.extensions["timeout"]["read"], 42)
            self.assertEqual(json.loads(request.content), {
                "model": "deepseek-flash", "messages": messages, "max_tokens": 4096,
                "tools": TOOL_SCHEMAS, "tool_choice": "auto", "thinking": {"type": "enabled"},
                "reasoning_effort": "high", "top_p": 0.95, "stream": False,
            })
            return httpx.Response(200, json=response)

        config = ModelConfig(provider="deepseek", base_url="https://api.deepseek.com/v1/", api_key_env="MODEL_KEY", max_tokens=4096, timeout=42, reasoning_effort="high")
        with patch.dict(os.environ, {"MODEL_KEY": "test-model-key"}):
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
                result = await create_model_client(config, http).complete(messages, TOOL_SCHEMAS)
        self.assertEqual(result, response)

    async def test_non_thinking_and_no_tools(self):
        def handler(request):
            payload = json.loads(request.content)
            self.assertEqual(payload["model"], "custom-model")
            self.assertEqual(payload["thinking"], {"type": "disabled"})
            self.assertEqual(payload["temperature"], 0.2)
            self.assertNotIn("reasoning_effort", payload)
            self.assertNotIn("tools", payload)
            self.assertNotIn("tool_choice", payload)
            self.assertNotIn("top_k", payload)
            self.assertNotIn("chat_template_kwargs", payload)
            return httpx.Response(200, json={"choices": []})

        config = ModelConfig(provider="deepseek", name="custom-model", enable_thinking=False, temperature=0.2, reasoning_effort="high")
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}):
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
                await create_model_client(config, http).complete([], [])

    async def test_missing_key_fails_before_any_request(self):
        def handler(request):
            self.fail("No request should be sent without an API key")

        with patch.dict(os.environ, {}, clear=True):
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
                with self.assertRaisesRegex(ValueError, "DEEPSEEK_API_KEY"):
                    await create_model_client(ModelConfig(provider="deepseek"), http).complete([], TOOL_SCHEMAS)

    async def test_http_errors_propagate(self):
        for status in (401, 429, 500):
            with self.subTest(status=status), patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}):
                async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(status))) as http:
                    with self.assertRaises(httpx.HTTPStatusError) as raised:
                        await create_model_client(ModelConfig(provider="deepseek"), http).complete([], TOOL_SCHEMAS)
                    self.assertEqual(raised.exception.response.status_code, status)


if __name__ == "__main__":
    unittest.main()
