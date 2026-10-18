import argparse
import asyncio
import json
import logging
from pathlib import Path

import httpx
from dotenv import load_dotenv

from .config import Config
from .data.io import load_records
from .runner import run_dataset, solve


def main():
    parser = argparse.ArgumentParser(description="Qwen3 scientific reasoning with search/code tools")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--question", help="Run a single question")
    source.add_argument("--input", type=Path, help="JSONL file or JSON array")
    parser.add_argument("--output", type=Path, default=Path("outputs/results.jsonl"))
    parser.add_argument("--config", type=Path, help="YAML config; omitted fields use defaults")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()

    load_dotenv(args.env_file)
    config = Config.load(args.config)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    async def run():
        if args.question is not None:
            async with httpx.AsyncClient() as http:
                result = await solve(args.question, config, http)
            args.output.write_text(json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8")
            print(json.dumps({key: result[key] for key in ("status", "answer", "rounds")}, ensure_ascii=False))
        else:
            with args.output.open("w", encoding="utf-8") as output:
                await run_dataset(load_records(args.input), config, output)

    asyncio.run(run())
