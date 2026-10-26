"""Generate checkpointed DeepSeek trajectories in SuperGPQA source order."""

import argparse
import asyncio
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import logging
import os
from pathlib import Path
import string
import time

import httpx
from dotenv import load_dotenv

from .gpqa import now, parse_choice
from ..config import Config
from ..data.io import load_records, write_record
from ..output import compact_result, create_run_directory, write_json
from ..runner import solve


PROJECT = Path(__file__).resolve().parents[2]


def prepare_records(rows: list[dict], start_index: int = 0, *, end_index: int | None = None,
                    limit: int | None = None) -> list[dict]:
    if not 0 <= start_index < len(rows):
        raise ValueError(f"start_index must be between 0 and {len(rows) - 1}")
    if end_index is not None and limit is not None:
        raise ValueError("Use either end_index or limit")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    end = end_index if end_index is not None else len(rows)
    if limit is not None:
        end = min(len(rows), start_index + limit)
    if not start_index < end <= len(rows):
        raise ValueError("end_index must be greater than start_index and no greater than the dataset size")
    records = []
    for index in range(start_index, end):
        row = rows[index]
        options = row["options"]
        if not row["question"].strip() or not isinstance(options, list) or not 2 <= len(options) <= 26:
            raise ValueError(f"Invalid question or options at source index {index}")
        if any(not isinstance(option, str) or not option.strip() for option in options):
            raise ValueError(f"Empty or invalid option at source index {index}")
        labels = string.ascii_uppercase[:len(options)]
        correct_label = row["answer_letter"].strip().upper()
        if correct_label not in labels:
            raise ValueError(f"Invalid answer_letter at source index {index}")
        prompt = (
            "Solve the following multiple-choice question. Use search or code when helpful.\n\n"
            + row["question"].strip() + "\n\nOptions:\n"
            + "\n".join(f"{label}. {option}" for label, option in zip(labels, options))
            + "\n\nGive your final answer as exactly one letter from "
            + ", ".join(labels) + r" inside \boxed{...}."
        )
        records.append({"id": row["uuid"], "index": index, "question": prompt,
                        "labels": labels, "correct_label": correct_label, "input": row})
    return records


async def evaluate(args) -> Path | None:
    load_dotenv(args.env_file)
    cfg = Config.load(args.config)
    if cfg.model.provider != "deepseek":
        raise ValueError("This launcher requires model.provider: deepseek")
    if args.concurrency is not None:
        cfg.dataset.concurrency = args.concurrency
    if cfg.dataset.concurrency < 1:
        raise ValueError("concurrency must be positive")
    rows = load_records(args.input)
    records = prepare_records(rows, args.start_index, end_index=args.end_index, limit=args.limit)
    end_index = records[-1]["index"] + 1
    selected = {
        "dataset_rows": len(rows), "start_index": args.start_index,
        "end_index_exclusive": end_index, "total": len(records),
        "provider": cfg.model.provider, "model": cfg.model.name,
        "max_tokens": cfg.model.max_tokens, "concurrency": cfg.dataset.concurrency,
    }
    if args.dry_run:
        print(json.dumps(selected, ensure_ascii=False, indent=2), flush=True)
        return None

    key = os.getenv(cfg.model.api_key_env, "").strip()
    if not key:
        raise ValueError(f"DeepSeek API key is missing; set {cfg.model.api_key_env}")
    async with httpx.AsyncClient() as http:
        health = await http.get(cfg.model.base_url.rstrip("/") + "/models",
                                headers={"Authorization": "Bearer " + key}, timeout=30)
        health.raise_for_status()
        if cfg.model.name not in {model["id"] for model in health.json()["data"]}:
            raise ValueError("Configured model is not available from the endpoint")

        task = args.task_name or f"supergpqa-deepseek-from{args.start_index}"
        output = create_run_directory(task, args.output_dir, resume=args.resume)
        print(f"Output: {output}", flush=True)
        print(f"Source indices: {args.start_index}..{end_index - 1} (inclusive); "
              f"total={len(records)}; model={cfg.model.name}; concurrency={cfg.dataset.concurrency}", flush=True)
        traces_dir = output / "traces"
        traces_dir.mkdir(exist_ok=True)
        identity = {
            "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
            "start_index": args.start_index, "end_index_exclusive": end_index,
            "config": asdict(cfg), "prompt_version": 1,
        }
        summary_path = output / "summary.json"
        if args.resume:
            run = json.loads(summary_path.read_text())["run"]
            if run["identity"] != identity:
                raise ValueError("Cannot resume: dataset, selected range or configuration differs")
        else:
            run = {"task": task, "started_at": now(), "identity": identity}

        results_path = output / "results.jsonl"
        results = {}
        if args.resume:
            # Compact records avoid loading all historical reasoning into memory.
            for record in records:
                path = traces_dir / f'sample-{record["index"]:06d}.json'
                if path.exists():
                    saved = json.loads(path.read_text())
                    if saved.get("status") != "running" and "evaluation" in saved:
                        results[record["index"]] = {**compact_result(saved), "trace_file": f"traces/{path.name}"}
        with results_path.open("w", encoding="utf-8") as stream:
            for index in sorted(results):
                write_record(stream, results[index])

        status_counts = Counter(row["status"] for row in results.values())
        correct = sum(row["evaluation"]["is_correct"] for row in results.values())
        token_counts = Counter()
        for row in results.values():
            token_counts.update(row.get("usage", {}))
        active = set()
        run_status = "running"

        def save_summary():
            write_json(summary_path, {
                **selected, "status": run_status, "finished": len(results),
                "remaining": len(records) - len(results), "active_indices": sorted(active),
                "status_counts": dict(status_counts), "correct": correct,
                "accuracy_on_finished": correct / len(results) if results else None,
                "usage": dict(token_counts), "updated_at": now(), "run": run,
            })

        save_summary()
        pending = iter(record for record in records if record["index"] not in results)
        with results_path.open("a", encoding="utf-8") as stream:
            async def worker():
                nonlocal correct
                for record in pending:
                    index = record["index"]
                    path = traces_dir / f"sample-{index:06d}.json"
                    started = time.monotonic()
                    metadata = {"id": record["id"], "index": index, "input": record["input"], "started_at": now()}
                    state = {"question": record["question"], "status": "running", "answer": None,
                             "rounds": 0, "messages": [], "trace": []}

                    def checkpoint(value):
                        nonlocal state
                        state = {**metadata, **value}
                        write_json(path, state)

                    active.add(index)
                    checkpoint(state)
                    save_summary()
                    try:
                        result = await solve(record["question"], cfg, http, on_progress=checkpoint)
                    except Exception as exc:
                        logging.exception("Source index %s failed", index)
                        result = {**state, "status": "error", "answer": None,
                                  "error": f"{type(exc).__name__}: {exc}"}
                    predicted = parse_choice(result.get("answer"), record["labels"]) if result["status"] == "completed" else None
                    usage = {name: sum(step.get("usage", {}).get(name, 0) for step in result.get("trace", []))
                             for name in ("prompt_tokens", "completion_tokens", "total_tokens")}
                    result = {
                        **metadata, **result, "finished_at": now(),
                        "elapsed_seconds": round(time.monotonic() - started, 3), "usage": usage,
                        "evaluation": {"predicted_label": predicted, "correct_label": record["correct_label"],
                                       "is_correct": predicted == record["correct_label"]},
                    }
                    write_json(path, result)
                    compact = {**compact_result(result), "trace_file": f"traces/{path.name}"}
                    write_record(stream, compact)
                    results[index] = compact
                    active.remove(index)
                    status_counts[result["status"]] += 1
                    correct += result["evaluation"]["is_correct"]
                    token_counts.update(usage)
                    save_summary()
                    print(f'{len(results)}/{len(records)} index={index} id={record["id"]} '
                          f'status={result["status"]} predicted={predicted} '
                          f'correct={result["evaluation"]["is_correct"]} rounds={result["rounds"]}', flush=True)

            try:
                await asyncio.gather(*(worker() for _ in range(cfg.dataset.concurrency)))
                run_status = "completed"
            finally:
                if run_status != "completed":
                    run_status = "interrupted"
                save_summary()
        print(f"Finished: {len(results)}/{len(records)}; results={results_path}", flush=True)
        return output


def main():
    parser = argparse.ArgumentParser(description="Generate DeepSeek trajectories for zero-indexed SuperGPQA questions")
    parser.add_argument("--input", type=Path, default=PROJECT / "datasets/supergpqa/SuperGPQA-all.jsonl")
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/deepseek.yaml")
    parser.add_argument("--env-file", type=Path, default=PROJECT / ".env")
    parser.add_argument("--start-index", type=int, default=0, help="Inclusive source question index; minimum 0")
    stop = parser.add_mutually_exclusive_group()
    stop.add_argument("--end-index", type=int, help="Exclusive source question index; omitted means dataset end")
    stop.add_argument("--limit", type=int, help="Maximum number of questions starting at --start-index")
    parser.add_argument("--concurrency", type=int, help="Override dataset.concurrency")
    parser.add_argument("--task-name")
    parser.add_argument("--output-dir", type=Path, help="Directory under project outputs; required with --resume")
    parser.add_argument("--resume", action="store_true", help="Skip saved final results; restart interrupted questions")
    parser.add_argument("--dry-run", action="store_true", help="Validate the range and show settings without API calls")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        asyncio.run(evaluate(args))
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
