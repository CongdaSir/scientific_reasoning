import asyncio
import os
import signal
import sys
import tempfile
from pathlib import Path

from ..config import CodeConfig


class CodeTool:
    """Fresh Python process per call; files survive within one sample's workdir."""

    def __init__(self, config: CodeConfig, workdir: str | Path):
        self.config = config
        self.workdir = str(workdir)

    async def __call__(self, code: str) -> dict:
        cfg = self.config
        # Disk-backed streams avoid retaining unbounded child output in RAM.
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-I", "-u", "-c", code,
                cwd=self.workdir,
                env={"PATH": os.defpath, "HOME": self.workdir, "LANG": "C.UTF-8"},
                stdin=asyncio.subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            timed_out = False
            try:
                await asyncio.wait_for(process.wait(), timeout=cfg.timeout)
            except asyncio.TimeoutError:
                timed_out = True
            finally:
                # Also clean up child processes spawned by the generated program.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await process.wait()

            def read_output(stream):
                stream.seek(0)
                text = stream.read(cfg.max_output_chars * 4 + 1).decode("utf-8", errors="replace")
                return text[:cfg.max_output_chars], len(text) > cfg.max_output_chars

            out, out_cut = read_output(stdout)
            err, err_cut = read_output(stderr)
        return {
            "stdout": out,
            "stderr": err,
            "returncode": process.returncode,
            "timed_out": timed_out,
            "truncated": out_cut or err_cut,
        }
