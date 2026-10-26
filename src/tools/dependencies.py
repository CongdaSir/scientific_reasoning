import ast
import asyncio
from contextlib import asynccontextmanager
import fcntl
import json
from pathlib import Path
import re
import sys
import tempfile

from ..config import CodeConfig
from .python_process import run_python


REQUIREMENTS_PATH = Path(__file__).resolve().parents[2] / "requirements.txt"
INSTALL_LOCK_PATH = Path(sys.prefix) / ".code-tool-install.lock"
PACKAGE_NAMES = {
    "PIL": "Pillow", "cv2": "opencv-python", "sklearn": "scikit-learn",
    "skimage": "scikit-image", "yaml": "PyYAML", "bs4": "beautifulsoup4",
    "Bio": "biopython", "Crypto": "pycryptodome", "dateutil": "python-dateutil",
    "rdkit": "rdkit",
}
_CHECK_MODULES = """
import importlib.util
import json
import sys
print(json.dumps([name for name in json.loads(sys.argv[1])
                  if importlib.util.find_spec(name) is None]))
"""


def required_modules(code: str) -> list[str]:
    try:
        tree = ast.parse(code, filename="<string>")
        compile(tree, "<string>", "exec", dont_inherit=True)
    except SyntaxError:
        # Let the normal interpreter return the original syntax error; install nothing.
        return []
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call) and node.args:
            function = node.func
            dynamic_import = (
                isinstance(function, ast.Name) and function.id == "__import__"
                or isinstance(function, ast.Attribute) and function.attr == "import_module"
                and isinstance(function.value, ast.Name) and function.value.id == "importlib"
            )
            argument = node.args[0]
            if dynamic_import and isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                if argument.value and not argument.value.startswith("."):
                    names.add(argument.value.split(".")[0])
    return sorted(names - sys.stdlib_module_names - set(sys.builtin_module_names) - {"__main__"})


class DependencyError(Exception):
    def __init__(self, message: str, result: dict | None = None):
        result = result or {"stdout": "", "stderr": "", "returncode": 1, "timed_out": False}
        self.result = {
            "stdout": "",
            "stderr": message + "\n" + result["stdout"] + result["stderr"],
            "returncode": result["returncode"] or 1,
            "timed_out": result["timed_out"],
        }
        super().__init__(message)


@asynccontextmanager
async def installation_lock(timeout: float):
    with INSTALL_LOCK_PATH.open("a") as stream:
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if asyncio.get_running_loop().time() >= deadline:
                    raise DependencyError("Timed out waiting for another dependency installation.", {
                        "stdout": "", "stderr": "", "returncode": -9, "timed_out": True,
                    })
                await asyncio.sleep(0.1)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def record_requirements(report: dict) -> list[str]:
    installed = {item["metadata"]["name"]: item["metadata"]["version"] for item in report.get("install", [])}
    if not installed:
        return []
    normalize = lambda name: re.sub(r"[-_.]+", "-", name).lower()
    remaining = {normalize(name): (name, version) for name, version in installed.items()}
    original = REQUIREMENTS_PATH.read_text(encoding="utf-8") if REQUIREMENTS_PATH.exists() else ""
    lines = []
    for line in original.splitlines():
        match = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==", line)
        replacement = remaining.pop(normalize(match[1]), None) if match else None
        lines.append(f"{replacement[0]}=={replacement[1]}" if replacement else line)
    lines.extend(f"{name}=={version}" for name, version in sorted(remaining.values()))
    temporary = REQUIREMENTS_PATH.with_suffix(".txt.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(REQUIREMENTS_PATH)
    return [f"{name}=={version}" for name, version in installed.items()]


class DependencyManager:
    def __init__(self, config: CodeConfig, workdir: str | Path):
        self.config = config
        self.workdir = workdir

    async def missing(self, names: list[str]) -> list[str]:
        result = await run_python(["-c", _CHECK_MODULES, json.dumps(names)], self.workdir, self.config.timeout)
        if result["returncode"] != 0:
            raise DependencyError("Python dependency check failed; user code was not executed.", result)
        return json.loads(result["stdout"])

    async def prepare(self, code: str) -> str:
        if not self.config.auto_install:
            return ""
        names = required_modules(code)
        if not names:
            return ""
        messages = []
        async with installation_lock(self.config.install_timeout):
            # Wait for other installers before checking partially written package files.
            missing = await self.missing(names)
            for name in missing:
                if not await self.missing([name]):
                    continue
                package = self.config.package_map.get(name, PACKAGE_NAMES.get(name, name))
                if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", package):
                    raise DependencyError(f"Invalid pip package mapping for module {name!r}: {package!r}.")
                with tempfile.TemporaryDirectory(prefix="code-dependencies-") as temporary:
                    report_path = Path(temporary) / "pip-report.json"
                    result = await run_python([
                        "-m", "pip", "install", "--disable-pip-version-check", "--no-input",
                        "--progress-bar", "off", "--report", str(report_path), package,
                    ], self.workdir, self.config.install_timeout, network=True)
                    if result["returncode"] != 0:
                        reason = "timed out" if result["timed_out"] else "failed"
                        raise DependencyError(
                            f"Dependency installation {reason}: module {name!r}, pip package {package!r}. "
                            "User code was not executed.", result,
                        )
                    installed = record_requirements(json.loads(report_path.read_text(encoding="utf-8")))
                    messages.append("[dependencies] Installed: " + (", ".join(installed) or package) + "\n")
                if await self.missing([name]):
                    raise DependencyError(
                        f"Installed pip package {package!r}, but module {name!r} is still unavailable. "
                        "Check code.package_map or correct the import. User code was not executed."
                    )
        return "".join(messages)
