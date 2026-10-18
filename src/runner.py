import asyncio
import logging
import tempfile

import httpx

from .clients.vllm import VLLMClient
from .config import Config
from .data.io import write_record
from .reasoning.engine import ReasoningEngine
from .tools.code import CodeTool
from .tools.registry import ToolRegistry
from .tools.search import SearchTool

logger = logging.getLogger(__name__)


async def solve(question: str, config: Config, http: httpx.AsyncClient) -> dict:
    with tempfile.TemporaryDirectory(prefix="scientific-reasoning-") as workdir:
        tools = ToolRegistry(SearchTool(config.search, http), CodeTool(config.code, workdir))
        engine = ReasoningEngine(VLLMClient(config.model, http), tools, config.reasoning.max_rounds)
        return await engine.run(question)


async def run_dataset(records: list[dict], config: Config, output) -> None:
    concurrency = config.dataset.concurrency
    if concurrency < 1:
        raise ValueError("dataset.concurrency must be positive")
    pending = iter(enumerate(records))
    async with httpx.AsyncClient() as http:
        async def worker():
            for index, record in pending:
                sample_id = record.get(config.dataset.id_field, index)
                try:
                    result = await solve(record[config.dataset.question_field], config, http)
                except Exception as exc:
                    # One failed sample should not discard a long dataset run.
                    logger.exception("Sample %s failed", sample_id)
                    result = {"status": "error", "answer": None, "error": str(exc)}
                write_record(output, {"id": sample_id, "index": index, "input": record, **result})
                logger.info("Sample %s: %s", sample_id, result["status"])

        await asyncio.gather(*(worker() for _ in range(concurrency)))
