"""Optional detached CLI execution using the normal agent and session journal."""

from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from contextlib import closing
from pathlib import Path


def resume_session(state_dir, sid):
    """Read launch metadata without recovering, locking, or changing a live session."""
    database = state_dir / "sessions.sqlite3"
    if not database.is_file():
        raise ValueError(f"Session not found: {sid}")
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        columns = {column["name"] for column in connection.execute("PRAGMA table_info(sessions)")}
        if sid == "latest":
            # Store migrates older journals in the child. Preflight remains read-only.
            roots = "WHERE parent_id IS NULL " if "parent_id" in columns else ""
            row = connection.execute(
                "SELECT * FROM sessions " + roots + "ORDER BY updated DESC, rowid DESC LIMIT 1"
            ).fetchone()
        else:
            row = connection.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
        if row is None:
            raise ValueError(f"Session not found: {sid}")
        has_messages = connection.execute(
            "SELECT 1 FROM messages WHERE session_id=? LIMIT 1", (row["id"],)
        ).fetchone()
        return {"parent_id": None} | dict(row) | {"has_messages": has_messages is not None}


def launch(args, config):
    if args.command != "run":
        raise ValueError("--background requires the explicit run command; interactive chat is not supported")
    if args.prompt_file and str(args.prompt_file) == "-":
        raise ValueError(
            "--background cannot read --prompt-file -; use a UTF-8 prompt file or positional prompt"
        )
    prompt = " ".join(args.arguments) or None
    if args.prompt_file:
        if prompt:
            raise ValueError("Use either a positional prompt or --prompt-file")
        prompt = args.prompt_file.read_text(encoding="utf-8")
    if prompt is None and not args.resume:
        raise ValueError("Provide a prompt, --prompt-file, or --resume")
    if prompt is not None and (not prompt.strip() or len(prompt) > config.max_context_chars // 3):
        raise ValueError("Prompt must be nonempty and smaller than one third of max_context_chars")

    state_dir = Path(config.state_dir).expanduser().resolve()
    session = resume_session(state_dir, args.resume) if args.resume else None
    if session:
        if session["parent_id"] is not None:
            raise ValueError(
                f"Subagent sessions are read-only transcripts; resume parent {session['parent_id']}"
            )
        cwd = Path(session["cwd"])
        if args.cwd and args.cwd.expanduser().resolve() != cwd:
            raise ValueError("A resumed session keeps its original cwd; start a new session to change it")
        if prompt is None and not session["has_messages"]:
            raise ValueError("This session has no request to continue")
        mode = args.mode or session["mode"]
    else:
        cwd = (args.cwd or Path.cwd()).expanduser().resolve()
        mode = config.mode
    if not cwd.is_dir():
        raise ValueError(f"Working directory does not exist: {cwd}")
    if mode == "assisted":
        raise ValueError("--background cannot use assisted mode because there is no approval terminal")

    # Keep the launch cwd and environment so implicit config discovery and module
    # resolution match the foreground invocation. Explicit values freeze effective
    # mode/model/limits, and 'latest' is resolved before another session can appear.
    command = [sys.executable, "-u", "-m", "lillia"]
    values = {
        "config": args.config.resolve() if args.config else None,
        "mode": mode,
        "cwd": cwd,
        # A model profile is re-selected by its key, like --cloud below: the resolved parent name
        # matches no profile, and the child would keep the file's subagent and embedding settings.
        "model": getattr(config, "_model_profile", None) or config.model.name,
        "base-url": config.model.base_url,
        "state-dir": state_dir,
        "resume": session["id"] if session else None,
        "max-steps": config.max_steps,
        "max-seconds": config.max_turn_seconds,
        "output": args.output.resolve() if args.output else None,
    }
    command.extend(f"--{key}={value}" for key, value in values.items() if value is not None)
    # --cloud selects the whole profile, not just the parent model forwarded above: without it the
    # child would keep the local subagent and embedding settings from the file.
    command.extend(f"--{key}" for key in ("cloud", "verbose", "plain") if getattr(args, key, False))
    state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    background_dir = state_dir / "background"
    background_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=background_dir))
    log_path = run_dir / "output.log"
    try:
        if prompt is not None:
            prompt_path = run_dir / "prompt.txt"
            with os.fdopen(
                os.open(prompt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="utf-8"
            ) as handle:
                handle.write(prompt)
            command.append(f"--prompt-file={prompt_path}")
        command.append("run")
        with os.fdopen(os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as log:
            child = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
    except BaseException:
        shutil.rmtree(run_dir)
        raise
    return {"pid": child.pid, "log": str(log_path)}
