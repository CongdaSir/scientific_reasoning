import asyncio
import os
from pathlib import Path
import signal
import sys
import tempfile


async def run_python(arguments: list[str], workdir: str | Path, timeout: float, *, network=False) -> dict:
    env = {"PATH": os.defpath, "HOME": str(workdir), "LANG": "C.UTF-8"}
    if network:
        # pip may need the configured package mirror or proxy, but never model API keys.
        for name in (
            "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
            "http_proxy", "https_proxy", "all_proxy", "no_proxy",
            "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE",
            "PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL", "PIP_TRUSTED_HOST",
            "PIP_CERT", "PIP_CLIENT_CERT", "PIP_CONFIG_FILE",
        ):
            if name in os.environ:
                env[name] = os.environ[name]
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-I", "-u", *arguments,
            cwd=workdir, env=env, stdin=asyncio.subprocess.DEVNULL,
            stdout=stdout, stderr=stderr, start_new_session=True,
        )
        timed_out = False
        try:
            await asyncio.wait_for(process.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            timed_out = True
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()
        stdout.seek(0)
        stderr.seek(0)
        return {
            "stdout": stdout.read().decode("utf-8", errors="replace"),
            "stderr": stderr.read().decode("utf-8", errors="replace"),
            "returncode": process.returncode,
            "timed_out": timed_out,
        }
