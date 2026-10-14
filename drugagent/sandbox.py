"""Run model-written analysis code where it cannot reach anything.

An agent that writes research code has to run it, and the code is untrusted by
definition. The Docker backend runs it with no network, a read-only root
filesystem, no capabilities, an unprivileged user, and caps on memory, CPU,
process count and wall time; the working directory is the only writable mount, and
whatever the code leaves in `out/` comes back to the caller.

The subprocess backend exists so tests and CI run without Docker. It enforces the
timeout and an isolated interpreter (`python -I`) and nothing else - it is a
convenience, not a security boundary, and `RunResult.isolated` says so.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

IMAGE = "drugagent-sandbox:latest"


@dataclass
class RunResult:
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool
    backend: str
    isolated: bool
    seconds: float
    outputs: dict[str, bytes] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def docker_available(image: str = IMAGE) -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        probe = subprocess.run(["docker", "image", "inspect", image], capture_output=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0


def _docker_command(workdir: Path, name: str, image: str, memory: str, cpus: str) -> list[str]:
    return [
        "docker", "run", "--rm", "--name", name,
        "--network", "none",                      # no exfiltration, no downloads
        "--read-only", "--tmpfs", "/tmp:rw,size=64m",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--user", "65534:65534",                  # nobody
        "--memory", memory, "--cpus", cpus, "--pids-limit", "64",
        "-e", "MPLCONFIGDIR=/tmp", "-e", "HOME=/tmp",
        "-v", f"{workdir}:/work", "-w", "/work",
        image, "python", "main.py",
    ]


def run_python(code: str, timeout: float = 60.0, backend: str = "auto", files: dict[str, bytes] | None = None,
               image: str = IMAGE, memory: str = "512m", cpus: str = "1") -> RunResult:
    """Run `code` as main.py in a fresh directory holding `files`; collect everything written to out/."""
    if backend == "auto":
        backend = "docker" if docker_available(image) else "subprocess"
    workdir = Path(tempfile.mkdtemp(prefix="drugagent-sandbox-"))
    try:
        (workdir / "main.py").write_text(code, encoding="utf-8")
        (workdir / "out").mkdir()
        for name, data in (files or {}).items():
            (workdir / name).write_bytes(data)
        workdir.chmod(0o777)
        (workdir / "out").chmod(0o777)

        name = f"drugagent-sandbox-{uuid.uuid4().hex[:12]}"
        command = (_docker_command(workdir, name, image, memory, cpus) if backend == "docker"
                   else [sys.executable, "-I", "main.py"])
        start, timed_out = time.perf_counter(), False
        try:
            proc = subprocess.run(command, cwd=workdir, capture_output=True, text=True, timeout=timeout)
            returncode, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as exc:
            if backend == "docker":
                subprocess.run(["docker", "kill", name], capture_output=True, timeout=30)
            timed_out, returncode = True, None
            stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode(errors="replace")
            stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode(errors="replace")
        seconds = time.perf_counter() - start
        outputs = {p.relative_to(workdir / "out").as_posix(): p.read_bytes()
                   for p in (workdir / "out").rglob("*") if p.is_file()}
        return RunResult(returncode, stdout, stderr, timed_out, backend, backend == "docker", seconds, outputs)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
