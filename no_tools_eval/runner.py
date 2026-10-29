"""Streaming, resumable pass@1 evaluation against a local vLLM endpoint."""

import argparse
import asyncio
from collections import Counter
from datetime import datetime
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import signal
import time
from zoneinfo import ZoneInfo

import httpx

from .datasets import DATASETS, messages, prepare_dataset, sha256
from .scoring import grade, valid_reference


ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_jsonl(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(encode(row) + "\n" for row in rows))
    temporary.replace(path)


def request_payload(record, args):
    return {"model": args.model, "messages": messages(record), "n": 1,
            "max_tokens": args.max_tokens, "temperature": args.temperature,
            "top_p": args.top_p, "top_k": args.top_k, "min_p": 0.0,
            "seed": args.seed + record["index"],
            "chat_template_kwargs": {"enable_thinking": args.thinking},
            "stream": True, "stream_options": {"include_usage": True}}


def alive(pid_file):
    if pid_file is None:
        return True
    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, 0)
        status = Path(f"/proc/{pid}/stat")
        return not status.exists() or status.read_text().rsplit(")", 1)[-1].split()[0] != "Z"
    except (OSError, ValueError, IndexError):
        return False


async def wait_for_server(http, args):
    deadline = time.monotonic() + args.wait_seconds
    while True:
        if not alive(args.server_pid_file):
            raise RuntimeError("Model server exited; inspect server.log")
        try:
            response = await http.get(args.base_url.rstrip("/") + "/models", timeout=5)
            response.raise_for_status()
            if args.model not in {x["id"] for x in response.json()["data"]}:
                raise ValueError(f"Endpoint does not serve {args.model}")
            return
        except (httpx.HTTPError, KeyError) as exc:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Model endpoint was not ready after {args.wait_seconds}s") from exc
            await asyncio.sleep(3)


class IncompleteStream(RuntimeError):
    pass


async def read_stream(http, args, payload, stream_path, state, checkpoint):
    last_save = time.monotonic()
    saw_done = False
    # Every received data frame is flushed, including before a checkpoint exists.
    with stream_path.open("x", encoding="utf-8", buffering=1) as output:
        output.write(encode({"time": now(), "type": "request", "payload": payload}) + "\n")
        async with http.stream("POST", args.base_url.rstrip("/") + "/chat/completions", json=payload) as response:
            if response.is_error:
                await response.aread()
                response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                output.write(encode({"time": now(), "type": "data", "data": data}) + "\n")
                if data == "[DONE]":
                    saw_done = True
                    break
                packet = json.loads(data)
                if packet.get("error"):
                    raise RuntimeError(f"Model stream error: {packet['error']}")
                state["id"] = packet.get("id", state.get("id"))
                if packet.get("usage"):
                    state["usage"] = packet["usage"]
                for choice in packet.get("choices", []):
                    delta = choice.get("delta", {})
                    if delta.get("tool_calls") or delta.get("function_call"):
                        raise RuntimeError("Unexpected tool call in a tool-free evaluation")
                    state["content"] += delta.get("content") or ""
                    state["reasoning"] += delta.get("reasoning") or delta.get("reasoning_content") or ""
                    if choice.get("finish_reason") is not None:
                        state["finish_reason"] = choice["finish_reason"]
                if time.monotonic() - last_save >= 2:
                    checkpoint()
                    last_save = time.monotonic()
    if not saw_done or state["finish_reason"] is None:
        raise IncompleteStream("Response ended without a finish reason and [DONE]")


async def solve_record(http, args, record, output):
    path = output / record["dataset"] / "traces" / (record["id"] + ".json")
    stream_dir = path.parent.parent / "streams"
    previous = json.loads(path.read_text()) if path.exists() else {}
    trace = {"record": record, "started_at": now(), "status": "running",
             "request": request_payload(record, args), "attempts": previous.get("attempts", [])}
    started = time.monotonic()
    for retry in range(args.retries + 1):
        attempt_number = len(trace["attempts"]) + 1
        stream_path = stream_dir / f'{record["id"]}.attempt-{attempt_number:03d}.jsonl'
        state = {"content": "", "reasoning": "", "usage": {}, "finish_reason": None}
        attempt = {"number": attempt_number, "stream_file": str(stream_path.relative_to(output)), "started_at": now()}
        trace["attempts"].append(attempt)
        trace["response"] = state

        def checkpoint():
            trace["updated_at"] = now()
            write_json(path, trace)

        checkpoint()
        try:
            await asyncio.wait_for(read_stream(http, args, trace["request"], stream_path, state, checkpoint), timeout=args.request_timeout)
            trace["status"] = {"stop": "completed", "length": "max_tokens"}.get(state["finish_reason"], "error")
            if trace["status"] == "error":
                trace["error"] = f"Unexpected finish reason: {state['finish_reason']}"
            attempt.update({"status": trace["status"], "finished_at": now(), "usage": state["usage"]})
            break
        except asyncio.CancelledError:
            trace["status"] = "interrupted"
            attempt.update({"status": "interrupted", "finished_at": now()})
            checkpoint()
            raise
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, httpx.HTTPStatusError):
                error += ": " + exc.response.text[:4000]
            attempt.update({"status": "error", "error": error, "finished_at": now(), "usage": state["usage"]})
            trace.update({"status": "error", "error": error})
            checkpoint()
            transient = isinstance(exc, (httpx.TransportError, IncompleteStream, asyncio.TimeoutError)) or (
                isinstance(exc, httpx.HTTPStatusError) and (exc.response.status_code == 429 or exc.response.status_code >= 500))
            if transient and retry < args.retries and alive(args.server_pid_file):
                trace["status"] = "running"
                trace.pop("error", None)
                await asyncio.sleep(2 ** retry)
                continue
            break
    trace.update({"finished_at": now(), "elapsed_seconds": round(time.monotonic() - started, 3),
                  "evaluation": grade(record, trace["response"]["content"], trace["status"])})
    checkpoint()
    return {"id": record["id"], "dataset": record["dataset"], "index": record["index"],
            "status": trace["status"], "evaluation": trace["evaluation"],
            "usage": trace["response"]["usage"], "category": record.get("category"),
            "task_name": record.get("task_name"), "elapsed_seconds": trace["elapsed_seconds"],
            "trace_file": str(path.relative_to(output)), "error": trace.get("error")}


def metrics(rows, total, scorable_total=None):
    scorable_total = total if scorable_total is None else scorable_total
    scored = [x for x in rows if x["evaluation"].get("reference_valid", True)]
    correct = sum(x["evaluation"]["is_correct"] for x in scored)
    return {"total": total, "finished": len(rows), "pending": total - len(rows), "correct": correct,
            "scorable_total": scorable_total, "scored": len(scored), "data_errors": total - scorable_total,
            "unscorable_finished": len(rows) - len(scored),
            "accuracy_on_finished": correct / len(scored) if scored else None,
            "accuracy_over_selected": correct / scorable_total if scorable_total else None,
            "status_counts": dict(Counter(x["status"] for x in rows)),
            "unparsed": sum(not x["evaluation"]["parsed"] for x in rows),
            "prompt_tokens": sum(x.get("usage", {}).get("prompt_tokens", 0) for x in rows),
            "completion_tokens": sum(x.get("usage", {}).get("completion_tokens", 0) for x in rows)}


async def evaluate(args):
    if args.concurrency < 1 or args.max_tokens < 1 or args.retries < 0:
        raise ValueError("concurrency/max-tokens must be positive and retries nonnegative")
    suites, records = {}, {}
    for name in args.datasets:
        records[name], suites[name] = prepare_dataset(ROOT, name, args.limit, args.seed, args.rbench_subset)
    if args.tokenizer_path:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
        for record in itertools.chain.from_iterable(records.values()):
            tokens = tokenizer.apply_chat_template(messages(record), tokenize=True, add_generation_prompt=True,
                                                   enable_thinking=args.thinking, return_dict=False)
            record["prompt_tokens"] = len(tokens)
            if len(tokens) + args.max_tokens > args.max_model_len:
                raise ValueError(f"{record['id']} would exceed the model context window")
    generation = {key: getattr(args, key) for key in ("model", "max_tokens", "temperature", "top_p", "top_k", "seed", "thinking", "max_model_len")}
    identity = {"protocol_version": 1, "mode": "no tools; one response per question; pass@1; incomplete/error responses are incorrect",
                "generation": generation, "datasets": suites,
                "prepared_sha256": hashlib.sha256(encode(records).encode()).hexdigest()}
    if args.dry_run:
        print(json.dumps(identity, ensure_ascii=False, indent=2), flush=True)
        return

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # The descriptor lives for the entire coroutine, preventing two evaluators
    # from overwriting the same run even if one was launched by nohup.
    with (output / "evaluator.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another evaluator already owns this output directory") from exc
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if not args.resume or manifest["identity"] != identity:
                raise ValueError("Existing run requires --resume and identical dataset/generation settings")
        else:
            if args.resume:
                raise ValueError("Cannot resume: manifest.json does not exist")
            manifest = {"started_at": now(), "identity": identity, "base_url": args.base_url,
                        "concurrency": args.concurrency, "request_timeout": args.request_timeout,
                        "transport_retries": args.retries,
                        "implementation": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}}
            write_json(manifest_path, manifest)
            write_jsonl(output / "prepared.jsonl", itertools.chain.from_iterable(records.values()))
        implementation = {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
        history = manifest.setdefault("implementation_history", [{"updated_at": manifest["started_at"], "sha256": manifest["implementation"]}])
        if implementation != manifest["implementation"]:
            history.append({"updated_at": now(), "sha256": implementation})
            manifest["implementation"] = implementation
        write_json(manifest_path, manifest)
        reference_issues = [{"id": record["id"], "dataset": record["dataset"], "source_file": record["source_file"],
                             "source_index": record["source_index"], "reference": record["reference"],
                             "reason": "not a single available choice label; response retained but excluded from accuracy"}
                            for record in itertools.chain.from_iterable(records.values()) if not valid_reference(record)]
        write_json(output / "data_quality.json", {"reference_issues": reference_issues, "count": len(reference_issues)})
        results = {name: {} for name in args.datasets}
        for name, examples in records.items():
            for folder in ("traces", "streams"):
                (output / name / folder).mkdir(parents=True, exist_ok=True)
            if args.resume:
                for record in examples:
                    path = output / name / "traces" / (record["id"] + ".json")
                    if not path.exists():
                        continue
                    trace = json.loads(path.read_text())
                    status = trace.get("status")
                    if "evaluation" not in trace or status in ("running", "interrupted") or (args.retry_errors and status == "error"):
                        continue
                    results[name][record["id"]] = {"id": record["id"], "dataset": name, "index": record["index"],
                        "status": status, "evaluation": trace["evaluation"], "usage": trace["response"]["usage"],
                        "category": record.get("category"), "task_name": record.get("task_name"),
                        "elapsed_seconds": trace["elapsed_seconds"], "trace_file": str(path.relative_to(output)), "error": trace.get("error")}
        run_status = "waiting_for_model"

        def save():
            dataset_metrics = {}
            combined = []
            for name, items in results.items():
                rows = sorted(items.values(), key=lambda x: x["index"])
                combined.extend(rows)
                summary = {**metrics(rows, len(records[name]), sum(valid_reference(x) for x in records[name])), "dataset": name, "updated_at": now()}
                summary["by_category"] = {category: metrics([x for x in rows if x.get("category") == category],
                    sum(x.get("category") == category for x in records[name]),
                    sum(x.get("category") == category and valid_reference(x) for x in records[name]))
                    for category in sorted({x.get("category") for x in records[name] if x.get("category")})}
                dataset_metrics[name] = summary
                write_json(output / name / "summary.json", summary)
                write_jsonl(output / name / "results.jsonl", rows)
            summary = {"status": run_status, "started_at": manifest["started_at"], "updated_at": now(),
                       **metrics(combined, sum(map(len, records.values())),
                                 sum(valid_reference(x) for x in itertools.chain.from_iterable(records.values()))), "datasets": dataset_metrics}
            write_json(output / "summary.json", summary)
            return summary

        save()
        loop = asyncio.get_running_loop()
        task = asyncio.current_task()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, task.cancel)
        timeout = httpx.Timeout(connect=10, read=120, write=30, pool=30)
        try:
            async with httpx.AsyncClient(timeout=timeout, trust_env=False,
                                         limits=httpx.Limits(max_connections=args.concurrency + 2)) as http:
                await wait_for_server(http, args)
                run_status = "running"
                save()
                # Interleave the selected datasets so all four start immediately.
                work = iter(record for group in itertools.zip_longest(*records.values()) for record in group
                            if record is not None and record["id"] not in results[record["dataset"]])

                async def worker():
                    for record in work:
                        result = await solve_record(http, args, record, output)
                        results[record["dataset"]][record["id"]] = result
                        summary = save()
                        print(f'{now()} {summary["finished"]}/{summary["total"]} {record["id"]} '
                              f'{result["status"]} prediction={result["evaluation"]["prediction"]} '
                              f'correct={result["evaluation"]["is_correct"]}', flush=True)
                        if not alive(args.server_pid_file):
                            raise RuntimeError("Model server exited during evaluation; resume after restarting it")

                workers = [asyncio.create_task(worker()) for _ in range(args.concurrency)]
                try:
                    await asyncio.gather(*workers)
                finally:
                    for worker_task in workers:
                        worker_task.cancel()
                    await asyncio.gather(*workers, return_exceptions=True)
            run_status = "completed_with_errors" if any(x["status"] == "error" for items in results.values() for x in items.values()) else "completed"
        except asyncio.CancelledError:
            run_status = "interrupted"
            raise
        except Exception:
            run_status = "failed"
            raise
        finally:
            save()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.remove_signal_handler(sig)
        print(json.dumps(save(), ensure_ascii=False, indent=2), flush=True)


def parser():
    parser = argparse.ArgumentParser(description="Tool-free Qwen3 evaluation with streamed checkpoints")
    parser.add_argument("--datasets", choices=DATASETS, nargs="+", default=list(DATASETS))
    parser.add_argument("--limit", type=int, default=500, help="First N per dataset; maximum 500")
    parser.add_argument("--rbench-subset", choices=("rbench-t_en", "rbench-t_zh"), default="rbench-t_en")
    parser.add_argument("--base-url", default="http://127.0.0.1:18041/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--tokenizer-path", type=Path, default=Path("/data/shared/model/Qwen3-4B"))
    parser.add_argument("--max-model-len", type=int, default=40960)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--request-timeout", type=float, default=1800)
    parser.add_argument("--wait-seconds", type=float, default=1200)
    parser.add_argument("--retries", type=int, default=2, help="Retries only for failed transports/HTTP 429/5xx, not wrong answers")
    parser.add_argument("--server-pid-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate data and context budgets without model requests or output files")
    return parser


def main():
    arguments = parser().parse_args()
    if not arguments.dry_run and arguments.output_dir is None:
        raise SystemExit("--output-dir is required unless --dry-run is used")
    try:
        asyncio.run(evaluate(arguments))
    except (KeyboardInterrupt, asyncio.CancelledError):
        raise SystemExit(130)
