import argparse
import asyncio
from collections import Counter
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from .config import Config
from .data.io import load_records
from .output import create_run_directory, write_json
from .runner import run_dataset


def main():
    parser = argparse.ArgumentParser(description="Scientific reasoning with vLLM or DeepSeek API and search/code tools")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--question", help="Run a single question")
    source.add_argument("--input", type=Path, help="JSONL file or JSON array")
    parser.add_argument("--task-name", help="Run name; defaults to 'single' or the input filename stem")
    parser.add_argument("--output-dir", type=Path, help="Directory name under project outputs; timestamp added automatically")
    parser.add_argument("--config", type=Path, help="YAML config; omitted fields use defaults")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()

    load_dotenv(args.env_file)
    config = Config.load(args.config)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    records = [{config.dataset.id_field: "sample-001", config.dataset.question_field: args.question}] if args.question is not None else load_records(args.input)
    task = args.task_name or ("single" if args.question is not None else args.input.stem)
    output_dir = create_run_directory(task, args.output_dir)
    summary = {"task": task, "total": len(records), "finished": 0, "status_counts": {}}
    write_json(output_dir / "summary.json", summary)
    print(f"Output: {output_dir}", flush=True)

    async def run():
        results_path = output_dir / "results.jsonl"
        with results_path.open("w", encoding="utf-8") as output:
            await run_dataset(records, config, output, traces_dir=output_dir / "traces")
        results = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]
        summary.update({"finished": len(results), "status_counts": dict(Counter(row["status"] for row in results))})
        write_json(output_dir / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False))

    asyncio.run(run())
