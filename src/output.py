from datetime import datetime
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo


OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "outputs"


def create_run_directory(task: str, output_dir: Path | None = None, *, resume=False) -> Path:
    """Keep each run directly under the project's outputs directory."""
    root = OUTPUT_ROOT.resolve()
    if output_dir is None:
        if resume:
            raise ValueError("--resume requires --output-dir pointing to an existing run")
        if not task or task in {".", ".."} or Path(task).name != task:
            raise ValueError("Task name must be a nonempty directory name")
        output_dir = root / task
    else:
        output_dir = output_dir.resolve()
    if output_dir.parent != root:
        raise ValueError(f"Output directory must be directly inside {root}")
    if resume:
        if not output_dir.is_dir():
            raise ValueError("Cannot resume a run directory that does not exist")
        return output_dir
    if not re.search(r"-\d{8}-\d{6}$", output_dir.name):
        stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
        output_dir = output_dir.with_name(f"{output_dir.name}-{stamp}")
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def compact_result(result: dict) -> dict:
    """Store the conversation once, in its per-question trace."""
    return {key: value for key, value in result.items() if key not in {"messages", "trace", "question", "input"}}
