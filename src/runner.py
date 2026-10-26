import asyncio
import logging
from pathlib import Path
import tempfile

import httpx

from .clients import create_model_client
from .config import Config
from .data.io import write_record
from .output import compact_result, write_json
from .reasoning.engine import ReasoningEngine
from .tools.code import CodeTool
from .tools.registry import ToolRegistry
from .tools.search import SearchTool

logger = logging.getLogger(__name__)


async def solve(question: str, config: Config, http: httpx.AsyncClient, *, on_progress=None) -> dict:
    with tempfile.TemporaryDirectory(prefix="scientific-reasoning-") as workdir:
        tools = ToolRegistry(SearchTool(config.search, http), CodeTool(config.code, workdir))
        engine = ReasoningEngine(
            create_model_client(config.model, http), tools, config.reasoning.max_rounds,
            on_progress=on_progress,
        )
        return await engine.run(question)


async def run_dataset(records: list[dict], config: Config, output, *, traces_dir: Path | None = None) -> None:
    concurrency = config.dataset.concurrency
    if concurrency < 1:
        raise ValueError("dataset.concurrency must be positive")
    if traces_dir is not None:
        traces_dir.mkdir(parents=True, exist_ok=True)
    pending = iter(enumerate(records))
    async with httpx.AsyncClient() as http:
        async def worker():
            for index, record in pending:
                sample_id = record.get(config.dataset.id_field, index)
                metadata = {"id": sample_id, "index": index, "input": record}
                state = {"rounds": 0, "messages": [], "trace": []}
                path = traces_dir / f"sample-{index + 1:03d}.json" if traces_dir is not None else None

                def checkpoint(value):
                    nonlocal state
                    state = {**metadata, **value}
                    write_json(path, state)

                try:
                    kwargs = {"on_progress": checkpoint} if path is not None else {}
                    result = await solve(record[config.dataset.question_field], config, http, **kwargs)
                except Exception as exc:
                    # One failed sample should not discard a long dataset run.
                    logger.exception("Sample %s failed", sample_id)
                    result = {**state, "status": "error", "answer": None, "error": str(exc)}
                result = {**metadata, **result}
                if path is not None:
                    write_json(path, result)
                    write_record(output, {**compact_result(result), "trace_file": f"traces/{path.name}"})
                else:
                    write_record(output, result)
                logger.info("Sample %s: %s", sample_id, result["status"])

        await asyncio.gather(*(worker() for _ in range(concurrency)))
