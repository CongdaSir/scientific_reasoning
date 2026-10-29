"""Deterministic, answer-isolated adapters for the downloaded benchmarks."""

import csv
import hashlib
import json
from pathlib import Path
import random
import re


DATASETS = ("gpqa-diamond", "rbench", "scieval", "scibench")
SYSTEM_PROMPT = "You are a careful scientific reasoner. Solve the problem step by step and follow the requested final-answer format."


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric_target(answer: str, unit: str) -> tuple[float, str]:
    """Move powers of ten in the dataset's unit into the numeric target."""
    value = float(str(answer).strip().replace(",", "").replace("−", "-"))
    exponent = r"(?:\{\s*([+-]?\d+)\s*\}|([+-]?\d+))"
    # Some source units put the complete scale inside \mathrm{...}.
    scale = re.search(r"\\(?:mathrm|text)\s*\{\s*10\s*\^\s*" + exponent + r"\s*\}", unit)
    if scale is None:
        scale = re.search(r"(?:\\times|\\cdot|[×*])?\s*10\s*\^\s*" + exponent, unit)
    if scale:
        value *= 10 ** int(scale.group(1) or scale.group(2))
        unit = unit[:scale.start()] + unit[scale.end():]
    return value, unit.replace("$", "").strip()


def choice_prompt(question: str, options: dict | None = None) -> str:
    if options:
        question += "\n\nOptions:\n" + "\n".join(f"{k}. {v}" for k, v in options.items())
        labels = ", ".join(options)
    else:
        labels = ", ".join(dict.fromkeys(re.findall(r"(?:^|\n)\s*([A-Z])[.)、]", question))) or "A, B, C, D"
    return question.strip() + f"\n\nReason step by step. Put only the final choice letter ({labels}) inside \\boxed{{...}}."


def prepare_dataset(root: Path, name: str, limit: int = 500, seed: int = 42,
                    rbench_subset: str = "rbench-t_en") -> tuple[list[dict], dict]:
    if not 1 <= limit <= 500:
        raise ValueError("limit must be between 1 and 500")
    records, sources = [], []
    base = root / "datasets"
    metadata = {"dataset": name, "selection": "first N in source order; questions are not shuffled", "limit": limit}

    def source(path):
        sources.append({"path": str(path.relative_to(root)), "sha256": sha256(path), "bytes": path.stat().st_size})
        return path

    if name == "gpqa-diamond":
        path = source(base / "gpqa/gpqa_diamond.csv")
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        rng = random.Random(seed)
        for index, row in enumerate(rows[:limit]):
            choices = [(row["Correct Answer"].strip(), True)] + [
                (row[f"Incorrect Answer {n}"].strip(), False) for n in range(1, 4)
            ]
            rng.shuffle(choices)
            options = dict(zip("ABCD", [value for value, _ in choices]))
            records.append({"question": choice_prompt(row["Question"].strip(), options),
                            "options": options, "reference": next(k for k, (_, ok) in zip(options, choices) if ok),
                            "kind": "choice", "source_record_id": row.get("Record ID"),
                            "category": row.get("High-level domain"), "source_file": str(path.relative_to(root)),
                            "source_index": index})
        metadata["option_shuffle_seed"] = seed
    elif name == "rbench":
        if rbench_subset not in ("rbench-t_en", "rbench-t_zh"):
            raise ValueError("Qwen3-4B supports the text subsets rbench-t_en and rbench-t_zh")
        import pyarrow.parquet as pq

        path = source(base / "rbench" / rbench_subset / "test-00000-of-00001.parquet")
        table = pq.read_table(path)
        rows = table.slice(0, min(limit, table.num_rows)).to_pylist()
        for index, row in enumerate(rows):
            options = {label: row[label].strip().replace("\\n", "\n") for label in "ABCDEF" if row.get(label)}
            records.append({"question": choice_prompt(row["question"].replace("\\n", "\n"), options),
                            "options": options, "reference": row["answer"].strip().upper(), "kind": "choice",
                            "source_record_id": row["index"], "source_file": str(path.relative_to(root)),
                            "source_index": index, "category": rbench_subset})
        metadata.update({"subset": rbench_subset, "available": table.num_rows,
                         "excluded_subsets": {"rbench-m_en": "requires image input", "rbench-m_zh": "requires image input"},
                         "languages": "one language per run; translations are not counted as different questions"})
    elif name == "scieval":
        path = source(base / "scieval/scieval-test-local.json")
        rows = json.loads(path.read_text())
        for index, row in enumerate(rows[:limit]):
            kind = row["type"]
            if kind == "multiple-choice":
                prompt = choice_prompt(row["question"])
                kind = "choice"
            else:
                prompt = row["question"].strip() + "\n\nReason step by step. Put the final answer inside \\boxed{...}."
            if not row.get("answer"):
                raise ValueError(f"SciEval row {index} has no reference answer")
            records.append({"question": prompt, "reference": row["answer"][0], "kind": kind,
                            "source_record_id": row.get("id"), "category": row.get("category"),
                            "task_name": row.get("task_name"), "ability": row.get("ability"),
                            "source_file": str(path.relative_to(root)), "source_index": index})
        metadata["subset"] = "test-local (labelled counterpart of test; dev/valid/dynamic are excluded)"
        metadata["source_order_note"] = "The first 500 test rows are all MedQA; no stratified resampling."
    elif name == "scibench":
        paths = sorted(p for p in (base / "scibench/dataset/original").glob("*.json") if not p.stem.endswith("_sol"))
        if not paths:
            raise FileNotFoundError("SciBench original JSON files were not found")
        rows = []
        for path in paths:
            source(path)
            rows.extend((path, i, row) for i, row in enumerate(json.loads(path.read_text())))
        for path, index, row in rows[:limit]:
            reference, unit = numeric_target(row["answer_number"], row.get("unit", ""))
            prompt = row["problem_text"].strip()
            if unit:
                prompt += f"\n\nExpress the answer in this unit: {unit}."
            prompt += "\n\nReason step by step. Put only the final numeric value, without units, inside \\boxed{...}."
            records.append({"question": prompt, "reference": reference, "kind": "numeric", "unit": unit,
                            "original_answer_number": row["answer_number"], "original_unit": row.get("unit", ""),
                            "source_record_id": row.get("problemid"), "category": path.stem,
                            "source_file": str(path.relative_to(root)), "source_index": index})
        metadata.update({"file_order": [p.name for p in paths], "selection": "lexicographic textbook filename order, then original row order; first N overall",
                         "excluded": {"*_sol.json": "worked examples, not test problems", "img/": "requires image input"},
                         "numeric_relative_tolerance": 0.05, "numeric_absolute_tolerance": 0.0})
    else:
        raise ValueError(f"Unknown dataset: {name}")

    for index, record in enumerate(records):
        record.update({"id": f"{name}-{index:06d}", "index": index, "dataset": name})
        if not record["question"].strip():
            raise ValueError(f"Empty question: {record['id']}")
    metadata.setdefault("available", len(rows))
    metadata.update({"selected": len(records), "sources": sources})
    return records, metadata


def messages(record: dict) -> list[dict]:
    # Reference answers, explanations, and solutions never enter the conversation.
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": record["question"]}]
