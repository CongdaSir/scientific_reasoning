from pathlib import Path

from ..config import CodeConfig
from .dependencies import DependencyError, DependencyManager
from .python_process import run_python


_PYTHON_RUNNER = """
import ast
import __future__
import sys
from types import ModuleType

source = sys.argv[1]
sys.argv = ["-c"]
tree = ast.parse(source, filename="<string>", mode="exec")
# Validate the whole program before running any statements.
compiled = compile(tree, "<string>", "exec", dont_inherit=True)
future_flags = compiled.co_flags & sum(
    getattr(__future__, name).compiler_flag for name in __future__.all_feature_names
)
original_main = sys.modules["__main__"]
module = ModuleType("__main__")
sys.modules["__main__"] = module
namespace = module.__dict__
if tree.body and isinstance(tree.body[-1], ast.Expr):
    expression = tree.body.pop()
    exec(compile(tree, "<string>", "exec", flags=future_flags, dont_inherit=True), namespace)
    value = eval(compile(ast.Expression(expression.value), "<string>", "eval",
                         flags=future_flags, dont_inherit=True), namespace)
    sys.displayhook(value)
else:
    exec(compiled, namespace)
"""


class CodeTool:
    """Fresh Python process per call; files survive within one sample's workdir."""

    def __init__(self, config: CodeConfig, workdir: str | Path):
        self.config = config
        self.workdir = str(workdir)
        self.dependencies = DependencyManager(config, workdir)

    async def __call__(self, code: str) -> dict:
        try:
            installation_output = await self.dependencies.prepare(code)
        except DependencyError as exc:
            return exc.result
        result = await run_python(["-c", _PYTHON_RUNNER, code], self.workdir, self.config.timeout)
        result["stderr"] = installation_output + result["stderr"]
        return result
