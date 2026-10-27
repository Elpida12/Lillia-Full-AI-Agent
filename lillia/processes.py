from __future__ import annotations

import asyncio
import codecs
import hashlib
import json
import os
import signal
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path


def _supervise(argv: list[str]):
    """Keep the managed group alive until its ordinary descendants have exited.

    Linux reparents orphaned descendants to this subreaper. Unlike waiting only
    for Bash or its output pipe, waiting for those children also covers nested
    background commands which redirect stdout. Children that deliberately leave
    the process group remain outside this lifetime guarantee.
    """
    import ctypes
    import subprocess

    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "Cannot supervise command descendants")

    # A group-wide graceful stop must leave the subreaper alive to collect its
    # children. Caught handlers reset to the default when the program executes.
    signal.signal(signal.SIGTERM, lambda *_: None)
    command = subprocess.Popen(argv)
    # Only the command owns the input pipe. Keeping a reader here would make
    # writes appear successful after the command deliberately closes its stdin.
    os.close(0)
    _, status = os.waitpid(command.pid, 0)
    command.returncode = os.waitstatus_to_exitcode(status)
    while True:
        try:
            os.waitpid(-os.getpgrp(), 0)
        except ChildProcessError:
            break

    if command.returncode < 0:
        sig = -command.returncode
        if sig not in {signal.SIGKILL, signal.SIGSTOP}:
            signal.signal(sig, signal.SIG_DFL)
        os.kill(os.getpid(), sig)
    os._exit(command.returncode)


@dataclass
class Job:
    id: str
    process: asyncio.subprocess.Process
    log_path: Path
    execution_mode: str = "shell"
    command_fingerprint: str = ""
    experiment_id: str | None = None
    hypothesis_id: str | None = None
    input_fingerprint: str | None = None
    timeout_seconds: int = 0
    output_sha256: str | None = None
    output: str = ""
    dropped_chars: int = 0
    logged_bytes: int = 0
    log_truncated: bool = False
    timed_out: bool = False
    stopped: bool = False
    error: str | None = None
    reader: asyncio.Task | None = None
    waiter: asyncio.Task | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class Processes:
    def __init__(self, root: Path, cwd: Path, output_limit: int = 16000):
        self.root, self.cwd, self.output_limit = root, cwd, output_limit
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.jobs: dict[str, Job] = {}

    async def start(
        self,
        command: str,
        cwd: str | None,
        timeout_seconds: int,
        wait_ms: int,
        *,
        experiment_id: str | None = None,
        hypothesis_id: str | None = None,
        input_fingerprint: str | None = None,
    ) -> dict:
        return await self._start(
            ["/bin/bash", "-o", "pipefail", "-c", command],
            cwd,
            timeout_seconds,
            wait_ms,
            execution_mode="shell",
            experiment_id=experiment_id,
            hypothesis_id=hypothesis_id,
            input_fingerprint=input_fingerprint,
        )

    async def start_program(
        self,
        argv: list[str],
        cwd: str | None,
        timeout_seconds: int,
        wait_ms: int,
        *,
        experiment_id: str | None = None,
        hypothesis_id: str | None = None,
        input_fingerprint: str | None = None,
    ) -> dict:
        """Run an argv directly: output collection cannot mask its exit status."""
        if (
            not isinstance(argv, list)
            or not argv
            or not argv[0]
            or any(not isinstance(arg, str) or "\0" in arg for arg in argv)
            or sum(len(arg.encode("utf-8")) for arg in argv) > 64000
        ):
            raise ValueError("argv must contain a program and NUL-free strings totaling at most 64000 bytes")
        return await self._start(
            argv,
            cwd,
            timeout_seconds,
            wait_ms,
            execution_mode="argv",
            experiment_id=experiment_id,
            hypothesis_id=hypothesis_id,
            input_fingerprint=input_fingerprint,
        )

    async def _start(
        self,
        argv: list[str],
        cwd: str | None,
        timeout_seconds: int,
        wait_ms: int,
        *,
        execution_mode: str,
        experiment_id: str | None,
        hypothesis_id: str | None,
        input_fingerprint: str | None,
    ) -> dict:
        if sys.platform != "linux":
            raise ValueError("Managed commands require Linux process supervision")
        if len(self.jobs) >= 64:
            # Completed handles are bounded; the output logs remain on disk.
            for key in list(self.jobs):
                if self.jobs[key].waiter.done():
                    del self.jobs[key]
                    if len(self.jobs) < 64:
                        break
        if len(self.jobs) >= 64 or sum(not j.waiter.done() for j in self.jobs.values()) >= 8:
            raise ValueError("Process limit reached; stop or wait for an existing process")
        directory = Path(cwd).expanduser() if cwd else self.cwd
        if not directory.is_absolute():
            directory = self.cwd / directory
        directory = directory.resolve(strict=True)
        if not directory.is_dir():
            raise ValueError("cwd is not a directory")
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            str(Path(__file__).resolve()),
            "--argv-json",
            json.dumps(argv, ensure_ascii=False),
            cwd=directory,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        jid = uuid.uuid4().hex[:12]
        fingerprint = hashlib.sha256(
            json.dumps({"argv": argv, "cwd": str(directory)}, sort_keys=True).encode("utf-8")
        ).hexdigest()
        job = Job(
            jid,
            proc,
            self.root / f"{jid}.log",
            execution_mode=execution_mode,
            command_fingerprint=fingerprint,
            experiment_id=experiment_id,
            hypothesis_id=hypothesis_id,
            input_fingerprint=input_fingerprint,
            timeout_seconds=timeout_seconds,
        )
        self.jobs[jid] = job
        job.reader = asyncio.create_task(self._read(job))
        job.waiter = asyncio.create_task(self._wait(job, timeout_seconds))
        try:
            return await self.poll(jid, wait_ms)
        except asyncio.CancelledError:
            await self.stop(jid)
            raise

    async def _read(self, job: Job):
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        digest = hashlib.sha256()
        fd = os.open(job.log_path, os.O_CREAT | os.O_WRONLY | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as log:
            while chunk := await job.process.stdout.read(4096):
                digest.update(chunk)
                room = max(0, 8 * 1024 * 1024 - job.logged_bytes)
                if room:
                    log.write(chunk[:room])
                    log.flush()
                    job.logged_bytes += min(room, len(chunk))
                if len(chunk) > room:
                    job.log_truncated = True
                self._buffer(job, decoder.decode(chunk))
            self._buffer(job, decoder.decode(b"", final=True))
            job.output_sha256 = digest.hexdigest()

    def _buffer(self, job: Job, text: str):
        combined = job.output + text
        excess = max(0, len(combined) - self.output_limit)
        job.dropped_chars += excess
        job.output = combined[excess:]

    @staticmethod
    def _signal(job: Job, sig):
        # The supervisor anchors this group until all ordinary descendants exit.
        # Never signal a completed handle: its numeric group ID can be reused.
        if job.process.returncode is not None:
            return
        try:
            os.killpg(job.process.pid, sig)
        except ProcessLookupError:
            pass

    async def _kill(self, job: Job):
        self._signal(job, signal.SIGTERM)
        await asyncio.sleep(0.2)
        self._signal(job, signal.SIGKILL)

    async def _wait(self, job: Job, timeout: int):
        wait_task = asyncio.create_task(job.process.wait())
        try:
            async with asyncio.timeout(timeout or None):
                # Notice output/log failures immediately, including while the child is still alive.
                await asyncio.gather(wait_task, asyncio.shield(job.reader))
        except (TimeoutError, OSError) as exc:
            job.timed_out = isinstance(exc, TimeoutError)
            if not job.timed_out:
                job.error = f"Output capture failed: {exc}"
            await self._kill(job)
            # A child which deliberately escapes its process group may hold the pipe.
            try:
                async with asyncio.timeout(2):
                    await job.process.wait()
                    await asyncio.shield(job.reader)
            except (TimeoutError, OSError):
                job.reader.cancel()
                await asyncio.gather(job.reader, return_exceptions=True)
        finally:
            if not wait_task.done():
                wait_task.cancel()
            await asyncio.gather(wait_task, return_exceptions=True)

    def get(self, jid: str) -> Job:
        if jid not in self.jobs:
            raise ValueError(
                "Unknown process handle. Handles only survive within this CLI process; "
                "inspect the machine and prior output logs after a restart."
            )
        return self.jobs[jid]

    async def poll(self, jid: str, wait_ms: int = 1000) -> dict:
        job = self.get(jid)
        if not job.waiter.done() and wait_ms:
            await asyncio.wait({job.waiter}, timeout=wait_ms / 1000)
        if job.waiter.done():
            job.waiter.result()
        async with job.lock:
            output, job.output = job.output, ""
            dropped, job.dropped_chars = job.dropped_chars, 0
        done = job.waiter.done()
        if job.timed_out:
            status = "timed_out"
        elif job.stopped:
            status = "stopped"
        elif not done:
            status = "running"
        elif job.error or job.process.returncode != 0:
            status = "failed"
        else:
            status = "succeeded"
        return {
            "process_id": jid,
            "status": status,
            # A zero exit status is execution evidence, not verification of the user's objective.
            "outcome_verified": False,
            "execution_mode": job.execution_mode,
            "exit_status_scope": "shell_exit_only" if job.execution_mode == "shell" else "program_exit",
            "command_fingerprint": job.command_fingerprint,
            "experiment_id": job.experiment_id,
            "hypothesis_id": job.hypothesis_id,
            "input_fingerprint": job.input_fingerprint,
            "input_fingerprint_source": "caller_declared" if job.input_fingerprint is not None else None,
            "timeout_seconds": job.timeout_seconds,
            "running": not done,
            "exit_code": job.process.returncode if done else None,
            "timed_out": job.timed_out,
            "stopped": job.stopped,
            "output": output,
            "output_sha256": job.output_sha256,
            "omitted_output_chars": dropped,
            "log_path": str(job.log_path),
            "log_truncated": job.log_truncated,
            "error": job.error,
            "ok": status in {"running", "succeeded"},
        }

    async def write(self, jid: str, text: str, close_stdin: bool = False) -> dict:
        job = self.get(jid)
        if job.waiter.done() or job.process.stdin.is_closing():
            raise ValueError("Process stdin is closed")
        job.process.stdin.write(text.encode())
        # The process lifetime already bounds slow consumers; a short independent
        # stdin deadline would fail tool calls while their command is still healthy.
        await asyncio.wait_for(job.process.stdin.drain(), job.timeout_seconds or None)
        if close_stdin:
            job.process.stdin.close()
        return await self.poll(jid)

    async def stop(self, jid: str) -> dict:
        job = self.get(jid)
        if not job.waiter.done():
            job.stopped = True
            await self._kill(job)
            try:
                await asyncio.wait_for(asyncio.shield(job.waiter), 3)
            except TimeoutError:
                job.reader.cancel()
                job.waiter.cancel()
                await asyncio.gather(job.reader, job.waiter, return_exceptions=True)
                job.waiter = asyncio.create_task(asyncio.sleep(0))
                await job.waiter
        return await self.poll(jid, 0)

    async def close(self):
        await asyncio.gather(*(self.stop(jid) for jid in list(self.jobs)), return_exceptions=True)


if __name__ == "__main__":
    # A running pre-upgrade CLI still loads this supervisor from disk for each
    # command. Preserve its original single-shell-command invocation contract.
    argv = (
        json.loads(sys.argv[2])
        if len(sys.argv) == 3 and sys.argv[1] == "--argv-json"
        else ["/bin/bash", "-o", "pipefail", "-c", sys.argv[1]]
    )
    _supervise(argv)
