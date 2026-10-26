import argparse
import asyncio
from collections import Counter
from dataclasses import asdict
from datetime import datetime
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import re
import time
from zoneinfo import ZoneInfo

import httpx
from dotenv import load_dotenv

from ..config import Config
from ..output import compact_result, create_run_directory
from ..runner import solve


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def prepare_records(rows: list[dict], limit: int | None, seed: int) -> list[dict]:
    if limit is None:
        limit = len(rows)
    if not 0 < limit <= len(rows):
        raise ValueError("limit must be positive and no greater than the dataset size")
    rng = random.Random(seed)
    records = []
    for index, row in enumerate(rows[:limit]):
        choices = [
            (row["Correct Answer"].strip(), True),
            *((row[f"Incorrect Answer {i}"].strip(), False) for i in range(1, 4)),
        ]
        if not row["Question"].strip() or any(not text for text, _ in choices):
            raise ValueError(f"Missing question or option at source row {index}")
        rng.shuffle(choices)
        options = dict(zip("ABCD", (text for text, _ in choices)))
        correct_label = next(label for label, (_, correct) in zip("ABCD", choices) if correct)
        prompt = (
            "Solve the following multiple-choice science question. "
            "Use search or code when helpful.\n\n"
            + row["Question"].strip()
            + "\n\nOptions:\n"
            + "\n".join(f"{label}. {text}" for label, text in options.items())
            + "\n\nGive your final answer as exactly one letter A, B, C, or D "
            r"inside \boxed{...}."
        )
        records.append({
            "id": f"gpqa-diamond-{index + 1:03d}", "index": index,
            "source_record_id": row.get("Record ID"),
            "domain": row.get("High-level domain"), "subdomain": row.get("Subdomain"),
            "question": prompt, "options": options,
            "correct_label": correct_label, "correct_answer": row["Correct Answer"].strip(),
        })
    return records


def parse_choice(answer: str | None, labels: str = "ABCD") -> str | None:
    if not isinstance(answer, str):
        return None
    value = answer.strip()
    for _ in range(4):
        value = re.sub(r"\\(?:text|mathrm|mathbf|operatorname)\s*\{([^{}]*)\}", r"\1", value)
    value = value.strip().strip("$ ")
    match = re.fullmatch(r"(?:option\s+)?[\(\[]?([A-Z])[\)\]]?\.?", value, re.IGNORECASE)
    return match.group(1).upper() if match and match.group(1).upper() in labels else None


def score_result(result: dict, correct_label: str) -> dict:
    predicted = parse_choice(result.get("answer")) if result.get("status") == "completed" else None
    return {"predicted_label": predicted, "correct_label": correct_label,
            "is_correct": predicted == correct_label}


def summarize(results: list[dict], total: int) -> dict:
    correct = sum(row["evaluation"]["is_correct"] for row in results)
    tool_counts = Counter()
    tool_errors = Counter()
    for row in results:
        names = {call["id"]: call["function"]["name"]
                 for msg in row.get("messages", []) for call in msg.get("tool_calls", [])}
        tool_counts.update(names.values())
        for msg in row.get("messages", []):
            if msg["role"] != "tool":
                continue
            data = json.loads(msg["content"])
            name = names.get(msg["tool_call_id"], "unknown")
            if data.get("error") or data.get("timed_out") or (name == "code" and data.get("returncode") != 0):
                tool_errors[name] += 1
    return {
        "total": total, "finished": len(results), "correct": correct,
        "accuracy": correct / total, "accuracy_percent": round(100 * correct / total, 4),
        "status_counts": dict(Counter(row["status"] for row in results)),
        "unparsed_completed_answers": sum(row["status"] == "completed" and row["evaluation"]["predicted_label"] is None for row in results),
        "tool_calls": dict(tool_counts), "tool_errors": dict(tool_errors),
        "model_rounds": sum(row.get("rounds", 0) for row in results),
        "prompt_tokens": sum(step.get("usage", {}).get("prompt_tokens", 0) for row in results for step in row.get("trace", [])),
        "completion_tokens": sum(step.get("usage", {}).get("completion_tokens", 0) for row in results for step in row.get("trace", [])),
        "updated_at": now(),
    }


async def evaluate(args) -> None:
    load_dotenv(args.env_file)
    cfg = Config.load(args.config)
    if cfg.dataset.concurrency < 1:
        raise ValueError("dataset.concurrency must be positive")
    source_bytes = args.csv.read_bytes()
    with args.csv.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    records = prepare_records(rows, args.limit, args.seed)
    task = args.task_name or ("gpqa-diamond-all" if args.limit is None else f"gpqa-diamond-first{args.limit}")
    output = create_run_directory(task, args.output_dir, resume=args.resume)
    print(f"Output: {output}", flush=True)
    traces_dir = output / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    secrets = [os.environ.get(name) for name in (cfg.model.api_key_env, cfg.search.api_key_env) if os.environ.get(name)]

    def serialize(value, *, indent=None):
        encoded = json.dumps(value, ensure_ascii=False, indent=indent)
        for secret in secrets:
            encoded = encoded.replace(json.dumps(secret, ensure_ascii=False)[1:-1], "[REDACTED]")
        return encoded

    def write_json(path: Path, value):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(serialize(value, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    identity = {
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(), "limit": len(records),
        "option_shuffle_seed": args.seed, "config": asdict(cfg),
        "protocol": "original CSV row order; independently shuffled options; one attempt per question; tools enabled; failures count as incorrect",
        "answer_parser": "single boxed A-D, optionally wrapped in LaTeX, parentheses or 'option'",
    }
    summary_path = output / "summary.json"
    if args.resume:
        saved = json.loads(summary_path.read_text())
        run = saved.get("run", {})
        saved_identity = run.get("identity", {})
        # Runs created before provider selection used vLLM and no effort override.
        saved_model = saved_identity.get("config", {}).get("model", {})
        saved_model.setdefault("provider", "vllm")
        saved_model.setdefault("reasoning_effort", None)
        if saved_identity != identity:
            raise ValueError("Cannot resume: dataset or evaluation settings differ from the saved run")
    else:
        run = {"task": task, "started_at": now(), "identity": identity, "dataset_rows": len(rows), "model_sampling_seed": None}

    def write_summary():
        summary = {**summarize(results, len(records)), "run": run}
        write_json(summary_path, summary)
        return summary

    def result_record(row):
        return {**compact_result(row), "trace_file": f'traces/{row["id"]}.json'}

    results = []
    done_ids = set()
    if args.resume:
        for row in records:
            path = traces_dir / f'{row["id"]}.json'
            if path.exists():
                saved = json.loads(path.read_text())
                if "evaluation" in saved and saved.get("status") != "running":
                    results.append(saved)
                    done_ids.add(row["id"])
    write_summary()
    results_path = output / "results.jsonl"
    results_path.write_text("".join(serialize(result_record(row)) + "\n" for row in sorted(results, key=lambda row: row["index"])), encoding="utf-8")
    headers = {"Authorization": "Bearer " + os.getenv(cfg.model.api_key_env, "EMPTY")}
    pending = iter(row for row in records if row["id"] not in done_ids)

    async with httpx.AsyncClient() as http:
        health = await http.get(cfg.model.base_url.rstrip("/") + "/models", headers=headers, timeout=30)
        health.raise_for_status()
        models = health.json()
        if cfg.model.name not in [model["id"] for model in models["data"]]:
            raise ValueError("Configured model is not served by the endpoint")

        async def worker():
            for record in pending:
                path = traces_dir / f'{record["id"]}.json'
                started_at = now()
                started = time.monotonic()
                metadata = {key: record[key] for key in ("id", "index", "source_record_id", "domain", "subdomain")}

                def checkpoint(state):
                    write_json(path, {**metadata, "started_at": started_at, **state})

                try:
                    # Only question text and shuffled choices enter the model conversation.
                    result = await solve(record["question"], cfg, http, on_progress=checkpoint)
                except Exception as exc:
                    result = json.loads(path.read_text()) if path.exists() else {
                        "question": record["question"], "rounds": 0, "messages": [], "trace": [],
                    }
                    result.update({"status": "error", "answer": None, "error": f"{type(exc).__name__}: {exc}"})
                    if isinstance(exc, httpx.HTTPStatusError):
                        result["http_error"] = {"status": exc.response.status_code, "body": exc.response.text}
                result = {
                    **result, **metadata, "started_at": started_at, "finished_at": now(),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "evaluation": score_result(result, record["correct_label"]),
                }
                write_json(path, result)
                results.append(result)
                with results_path.open("a", encoding="utf-8") as stream:
                    stream.write(serialize(result_record(result)) + "\n")
                summary = write_summary()
                print(f'{len(results):02d}/{len(records)} {record["id"]}: {result["status"]}, '
                      f'predicted={result["evaluation"]["predicted_label"]}, '
                      f'correct={result["evaluation"]["is_correct"]}, '
                      f'rounds={result.get("rounds", 0)}, total_correct={summary["correct"]}', flush=True)

        await asyncio.gather(*(worker() for _ in range(cfg.dataset.concurrency)))

    results.sort(key=lambda row: row["index"])
    results_path.write_text("".join(serialize(result_record(row)) + "\n" for row in results), encoding="utf-8")
    summary = write_summary()
    print(serialize(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Evaluate GPQA Diamond with complete per-question checkpoints")
    parser.add_argument("--csv", type=Path, default=Path("datasets/gpqa/gpqa_diamond.csv"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--limit", type=int, help="Evaluate the first N questions; omitted means the entire dataset")
    parser.add_argument("--seed", type=int, default=42, help="Option shuffle seed; question order is unchanged")
    parser.add_argument("--task-name", help="Run name; defaults to gpqa-diamond-all or gpqa-diamond-first<N>")
    parser.add_argument("--output-dir", type=Path, help="Directory under project outputs; required with --resume")
    parser.add_argument("--resume", action="store_true")
    asyncio.run(evaluate(parser.parse_args()))


if __name__ == "__main__":
    main()
