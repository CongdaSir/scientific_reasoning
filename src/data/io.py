import json
from pathlib import Path


def load_records(path: str | Path) -> list[dict]:
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in stream if line.strip()]
        return json.load(stream)


def write_record(stream, record: dict) -> None:
    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    stream.flush()
