import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from src.config import CodeConfig
from src.tools.code import CodeTool
from src.tools.dependencies import DependencyError, DependencyManager, required_modules


def process_result(stdout="", stderr="", returncode=0, timed_out=False):
    return {"stdout": stdout, "stderr": stderr, "returncode": returncode, "timed_out": timed_out}


class ImportTests(unittest.TestCase):
    def test_import_detection_and_syntax_validation(self):
        source = (
            "import os, numpy as np\nfrom rdkit.Chem import MolFromSmiles\n"
            "from PIL import Image\nfrom .local import helper\n"
            "import importlib\nimportlib.import_module('sklearn.metrics')\n"
            "__import__('scipy')\nfrom sympy import PointGroup\n"
        )
        self.assertEqual(required_modules(source), ["PIL", "numpy", "rdkit", "scipy", "sklearn", "sympy"])
        self.assertEqual(required_modules("import os, sys, math\nimport __main__"), [])
        self.assertEqual(required_modules("import rdkit\nif :"), [])
        self.assertEqual(required_modules("import rdkit\nreturn"), [])


class InstallationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workdir = Path(temporary.name)
        self.requirements = self.workdir / "requirements.txt"
        self.requirements.write_text("# Existing dependencies\n-e .[science]\nnumpy==2.0.0\nPillow==11.0.0\n")
        for name, value in (
            ("INSTALL_LOCK_PATH", self.workdir / "install.lock"),
            ("REQUIREMENTS_PATH", self.requirements),
        ):
            patcher = patch("src.tools.dependencies." + name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    async def test_installed_packages_and_disabled_installation_do_not_run_pip(self):
        for config in (CodeConfig(), CodeConfig(auto_install=False)):
            manager = DependencyManager(config, self.workdir)
            manager.missing = AsyncMock(return_value=[])
            with patch("src.tools.dependencies.run_python", new_callable=AsyncMock) as execute:
                self.assertEqual(await manager.prepare("import numpy"), "")
                execute.assert_not_called()
                if not config.auto_install:
                    manager.missing.assert_not_called()

    async def test_concurrent_questions_install_once_and_record_versions(self):
        installed = set()
        calls = []

        async def missing(names):
            return [name for name in names if name not in installed]

        async def install(arguments, workdir, timeout, *, network):
            self.assertTrue(network)
            self.assertEqual(arguments[-1], "Pillow")
            calls.append(arguments)
            await asyncio.sleep(0.02)
            report = Path(arguments[arguments.index("--report") + 1])
            report.write_text(json.dumps({"install": [
                {"metadata": {"name": "Pillow", "version": "12.0.0"}},
                {"metadata": {"name": "extra-dependency", "version": "1.2.3"}},
            ]}))
            installed.add("PIL")
            return process_result(stdout="pip download progress\n")

        managers = [DependencyManager(CodeConfig(), self.workdir) for _ in range(2)]
        for manager in managers:
            manager.missing = missing
        with patch("src.tools.dependencies.run_python", side_effect=install):
            results = await asyncio.gather(*(manager.prepare("from PIL import Image") for manager in managers))
        self.assertEqual(len(calls), 1)
        self.assertEqual(sum("Installed: Pillow==12.0.0" in text for text in results), 1)
        contents = self.requirements.read_text()
        self.assertIn("numpy==2.0.0\n", contents)
        self.assertIn("-e .[science]\n", contents)
        self.assertIn("Pillow==12.0.0\n", contents)
        self.assertNotIn("Pillow==11.0.0", contents)
        self.assertEqual(contents.count("Pillow=="), 1)
        self.assertIn("extra-dependency==1.2.3\n", contents)

    async def test_install_failure_and_timeout_prevent_user_code_execution(self):
        for failure in (
            process_result(stdout="resolver output\n" * 5000, stderr="No matching distribution\n", returncode=1),
            process_result(stderr="interrupted download\n", returncode=-9, timed_out=True),
        ):
            tool = CodeTool(CodeConfig(), self.workdir)
            tool.dependencies.missing = AsyncMock(return_value=["rdkit"])
            with patch("src.tools.dependencies.run_python", new_callable=AsyncMock, return_value=failure):
                result = await tool("open('executed.txt', 'w').write('ran')\nimport rdkit")
            self.assertFalse((self.workdir / "executed.txt").exists())
            self.assertEqual(set(result), {"stdout", "stderr", "returncode", "timed_out"})
            self.assertEqual(result["returncode"], failure["returncode"])
            self.assertEqual(result["timed_out"], failure["timed_out"])
            self.assertIn("User code was not executed", result["stderr"])
            self.assertIn(failure["stdout"] + failure["stderr"], result["stderr"])

    async def test_custom_package_mapping_and_verification_failure(self):
        manager = DependencyManager(CodeConfig(package_map={"example_import": "example-package"}), self.workdir)
        manager.missing = AsyncMock(return_value=["example_import"])

        async def install(arguments, *args, **kwargs):
            self.assertEqual(arguments[-1], "example-package")
            Path(arguments[arguments.index("--report") + 1]).write_text(json.dumps({"install": [
                {"metadata": {"name": "example-package", "version": "1.0.0"}},
            ]}))
            return process_result()

        with patch("src.tools.dependencies.run_python", side_effect=install):
            with self.assertRaisesRegex(DependencyError, "still unavailable"):
                await manager.prepare("import example_import")
        self.assertIn("example-package==1.0.0", self.requirements.read_text())

    async def test_invalid_mapping_does_not_invoke_installer(self):
        manager = DependencyManager(CodeConfig(package_map={"missing": "--upgrade"}), self.workdir)
        manager.missing = AsyncMock(return_value=["missing"])
        with patch("src.tools.dependencies.run_python", new_callable=AsyncMock) as execute:
            with self.assertRaisesRegex(DependencyError, "Invalid pip package mapping"):
                await manager.prepare("import missing")
            execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
