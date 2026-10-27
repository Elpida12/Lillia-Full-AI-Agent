from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import signal
import sys
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import httpx
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import __version__
from .agent import Agent
from .config import MODEL_ALIASES, load_config, profile_name
from .dashboard import STYLE, AgentDashboard
from .delegation import ModelScheduler
from .processes import Processes
from .provider import ChatProvider, ModelError
from .runtime import endpoint_origin
from .startup import StartupError, cloud_account_lines, load_startup_models
from .store import Store
from .tools import ToolRegistry

console = Console(highlight=False)
LISTING_LIMIT = 50
CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def safe(text) -> str:
    # Never let model or tool output emit terminal control codes, even across stream chunks.
    return CONTROL.sub("", str(text))


def say(text="", **kwargs):
    console.print(safe(text), markup=False, **kwargs)


def tool_failure_detail(result):
    for key in ("detail", "error"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value
    reasons = []
    if result.get("timed_out"):
        reasons.append("command timed out")
    if result.get("stopped"):
        reasons.append("command was stopped")
    if type(result.get("exit_code")) is int:
        reasons.append(f"exit code {result['exit_code']}")
    status = result.get("status")
    if isinstance(status, str) and status.strip():
        reasons.append(f"status {status}")
    return "; ".join(reasons) or "see command output"


def age(timestamp) -> str:
    # SQLite stores these as UTC via strftime. The raw ISO string is wider than the column Rich
    # gives it, so it used to render truncated ("2026-09-20T05:01:…") and told you nothing.
    try:
        moment = datetime.strptime(str(timestamp)[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return safe(timestamp)[:16]
    seconds = max(0.0, (datetime.now(timezone.utc) - moment).total_seconds())
    for size, unit in ((86400, "d"), (3600, "h"), (60, "m")):
        if seconds >= size:
            return f"{int(seconds // size)}{unit} ago"
    return "just now"


class Display:
    def __init__(self, verbose=False, plain=False):
        self.verbose = verbose
        self.plain = plain
        self.text_parts = []
        self.reasoning_parts = []
        self.answer_delivered = False
        self.activity = 0.0
        self.dashboard = AgentDashboard()

    def emit(self, kind: str, data: dict):
        self.activity = asyncio.get_running_loop().time()
        if kind == "text":
            self.text_parts.append(data["text"])
        elif kind == "reasoning":
            if data["state"] == "delta":
                if self.dashboard.enabled:
                    # The live preview keeps a bounded tail; scrollback needs the whole generation.
                    self.reasoning_parts.append(data["text"])
                    self.dashboard.update_reasoning(data)
            else:
                self.end_reasoning()
        elif kind == "message_end":
            self.end_text(complete=True)
        elif kind == "done":
            self.answer_delivered = data.get("answer_delivered", False)
        elif kind == "input_queued":
            say(f"\nYou (queued) › {data['text']}", style="bold cyan")
            say(f"{data['count']} message(s) queued for the next step.", style="dim")
        elif kind == "input_applied":
            say(f"Applying {data['count']} queued message(s).", style="cyan")
        elif kind == "tool_start":
            try:
                args = json.loads(data["arguments"])
                if not isinstance(args, dict):
                    args = {}
            except ValueError:
                args = {}
            detail = args.get(
                "argv", args.get("command", args.get("path", args.get("url", args.get("process_id", ""))))
            )
            say(f"→ {data['name']} {str(detail)[:1000]}", style="cyan")
        elif kind == "tool_result":
            result = data["result"]
            if self.verbose:
                say(json.dumps(result, ensure_ascii=False, indent=2)[:20000], style="dim")
            elif data["name"] in {
                "shell",
                "execute_program",
                "poll_process",
                "write_process",
                "stop_process",
            }:
                if result.get("output"):
                    say(result["output"].rstrip()[:4000], style="dim")
                if "running" in result:
                    status = "running" if result["running"] else f"exit {result['exit_code']}"
                    if result.get("timed_out"):
                        status += ", timed out"
                    say(f"  {result['process_id']} · {status}", style="dim")
            if result.get("error") == "recall_review_required":
                say(
                    "  Tool postponed (historical recall): Lillia will review earlier work before proceeding.",
                    style="dim",
                )
            elif not result.get("ok"):
                say(
                    f"  Tool failed: {tool_failure_detail(result)}",
                    style="yellow",
                )
        elif kind == "status":
            say(data["text"], style="dim")
        elif kind == "subagent":
            self.dashboard.update_child(data)
            say(f"  Subagent {data['name']} ({data['agent_id']}): {data['status']}", style="cyan")
        elif kind == "step":
            if self.verbose:
                say(
                    f"Step {data['step']}" + (f"/{data['max_steps']}" if data["max_steps"] else ""),
                    style="dim",
                )

    def end_reasoning(self):
        self.dashboard.update_reasoning({"state": "end"})
        if not self.reasoning_parts:
            return
        text = safe("".join(self.reasoning_parts))
        self.reasoning_parts.clear()
        # Print once above the dashboard so its redraws leave the full text in scrollback.
        with console.capture() as capture:
            console.print(Panel(Text(text), title="Reasoning", title_align="left", border_style="cyan"))
        sys.stdout.write(capture.get())
        sys.stdout.flush()

    def end_text(self, complete=False):
        self.end_reasoning()
        if not self.text_parts:
            return
        text = safe("".join(self.text_parts))
        self.text_parts.clear()
        # A single newline-terminated write lets patch_stdout suspend/redraw the
        # dashboard around the entire message, never around partial stream lines.
        with console.capture() as capture:
            say("\nLillia" if complete else "\nLillia (incomplete response)", style="bold magenta")
            if self.plain or not console.is_terminal or not complete:
                say(text, soft_wrap=True)
            else:
                console.print(Markdown(text, hyperlinks=False))
        sys.stdout.write(capture.get())
        sys.stdout.flush()

    async def heartbeat(self):
        while True:
            await asyncio.sleep(15)
            if asyncio.get_running_loop().time() - self.activity >= 15:
                if self.dashboard.enabled:
                    continue
                say("Working… waiting for the model or a tool. Ctrl-C stops this turn.", style="dim")


async def embedding_report(config, store) -> str:
    """Probe the embedding endpoint and report index coverage. Never fails doctor.

    Retrieval degrades to lexical-only when this endpoint is down, so an unreachable
    embedder is a reduced capability, not a broken installation.
    """
    if not config.retrieval.enabled:
        return "Retrieval: disabled (retrieval.enabled = false)"
    from .index import EvidenceIndex

    index = EvidenceIndex(store, None, config.retrieval, config.embedding_model)
    lines = [f"Evidence index: {index.path}"]
    try:
        stats = await index.stats()
        covered = stats["chunks_with_vectors"]
        lines.append(
            f"Indexed artifacts: {stats['artifacts_indexed']:,} · chunks: {stats['chunks']:,} "
            f"· with vectors: {covered:,}"
        )
        if not config.embedding_model.enabled:
            lines.append("Embeddings: disabled (embedding_model.enabled = false); lexical search only")
            return "\n".join(lines)
        embedding = config.embedding_model
        lines.append(
            f"Embedding endpoint: {embedding.base_url} · model: {embedding.name}"
            + (f" · index model: {embedding.vector_model}" if embedding.vector_model != embedding.name else "")
        )
        try:
            probe = await index.provider().probe()
            lines.append(
                f"Embedding endpoint reachable; {probe['native_dimensions']} native dims stored as "
                f"{probe['stored_dimensions']}. This checks the endpoint, not retrieval quality."
            )
            if stats["chunks"] and covered < stats["chunks"]:
                lines.append(
                    f"Vector backfill pending for {stats['chunks'] - covered:,} chunks; "
                    "run 'lillia index' to complete it."
                )
        except Exception as exc:
            lines.append(
                f"Embedding check failed ({type(exc).__name__}): {exc}. Search falls back to "
                "lexical-only. Check the configured endpoint and its model-server logs."
            )
    finally:
        await index.close()
    return "\n".join(lines)


# Environment overrides load_config applies after a profile, so they silently win over it. A model
# profile is the exception for LILLIA_MODEL: the --model that selected it outranks that variable.
ENDPOINT_OVERRIDES = (
    "LILLIA_BASE_URL",
    "LILLIA_MODEL",
    "LILLIA_SUBAGENT_BASE_URL",
    "LILLIA_SUBAGENT_MODEL",
    "LILLIA_EMBEDDING_BASE_URL",
    "LILLIA_EMBEDDING_MODEL",
)


def unselected_profile(config) -> str | None:
    """The model profile whose parent this local run uses without the rest of its placement, if any.

    Passing a profile's gateway ID (--model or LILLIA_MODEL) selects nothing, so only the parent moves."""
    if getattr(config, "_model_profile", None) or getattr(config, "_profile", None) == "cloud":
        return None
    subagent = config.subagent_model.name if config.subagent_model is not None else config.model.name
    for key, (parent, worker, embedder) in getattr(config, "_profile_placements", {}).items():
        if parent is None or parent != config.model.name:
            continue
        if (worker is not None and worker != subagent) or (
            embedder is not None and embedder != config.embedding_model.name
        ):
            return key
    return None


async def report_cloud_account(config) -> None:
    """In the cloud profile, name the Ollama account and plan whose usage the chat models draw on.

    In a model profile, only warn about environment overrides that replace its placement. Otherwise
    warn when the parent is a profile's parent but the rest of that profile's placement is not used."""
    model_profile = getattr(config, "_model_profile", None)
    if model_profile:
        overrides = [name for name in ENDPOINT_OVERRIDES if name != "LILLIA_MODEL" and name in os.environ]
        if overrides:
            say(
                f"Environment overrides take precedence over the [{profile_name(model_profile)}] profile: "
                f"{', '.join(overrides)}",
                style="yellow",
            )
        return
    unselected = unselected_profile(config)
    if unselected:
        say(
            f"{config.model.name} is the parent of the [{profile_name(unselected)}] placement profile, "
            "which this run did not select: the subagents and embedder keep the base placement and can "
            f"evict it on every switch. Use --model {unselected} for the whole layout.",
            style="yellow",
        )
    if getattr(config, "_profile", None) != "cloud":
        return
    overrides = [name for name in ENDPOINT_OVERRIDES if name in os.environ]
    if overrides:
        say(f"Environment overrides take precedence over the [cloud] profile: {', '.join(overrides)}", style="yellow")
    for text, warning in await cloud_account_lines(config):
        say(text, style="yellow" if warning else "dim")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="lillia",
        description="Lillia-Full — a general-purpose local AI agent",
        epilog="With no command, opens chat. A quoted prompt is shorthand for 'run PROMPT'. "
        "Chat commands: /help, /clear, /continue, /status, /agents, /mode, /memory, /exit.",
    )
    result.add_argument(
        "command",
        nargs="?",
        default="chat",
        help="chat, run, doctor, models, index, sessions, show, export, memory; or a quoted prompt",
    )
    result.add_argument("arguments", nargs="*", help="prompt text or session ID")
    result.add_argument("--version", action="version", version=f"Lillia-Full {__version__}")
    result.add_argument("--config", type=Path, help="Explicit TOML config file")
    result.add_argument(
        "--rebuild",
        action="store_true",
        help="For 'index': discard the evidence index and build it again from scratch",
    )
    result.add_argument("--mode", choices=["full", "assisted", "observe"])
    result.add_argument("--cwd", type=Path, help="Working directory for a new session")
    result.add_argument(
        "--model",
        help="Model ID or shortcut. Qwen3.6-35B selects the config's placement profile "
        '[model_profiles."Qwen3.6-35B"]: the 35B on GPUs 0/1 for parent and subagents, embedder '
        "on GPU2 (without that table it is a shortcut for Qwen3.6-35B-A3B). The full ID "
        "Qwen3.6-35B-A3B selects the GPU2+RAM entry without the profile. "
        "Default: Qwen3.8-27B unless overridden by config or LILLIA_MODEL. "
        "With --cloud: an Ollama cloud model such as glm-5.3:cloud.",
    )
    result.add_argument(
        "--cloud",
        action="store_true",
        help="Use the config file's [cloud] profile: parent and subagent models on Ollama Cloud, "
        "billed to the Ollama account the local daemon is signed in to; embeddings stay local",
    )
    result.add_argument("--base-url", help="OpenAI-compatible API endpoint including /v1")
    result.add_argument("--state-dir", type=Path, help="Override session and log storage directory")
    result.add_argument("--resume", nargs="?", const="latest", help="Resume a session (default: latest)")
    result.add_argument("--prompt-file", type=Path, help="Read prompt from a UTF-8 file; '-' reads stdin")
    result.add_argument("--max-steps", type=int)
    result.add_argument(
        "--max-seconds",
        type=int,
        help="Opt-in turn time limit in seconds (0 disables it); prompt durations do not create a cutoff",
    )
    result.add_argument("--verbose", action="store_true", help="Show detailed tool results")
    result.add_argument("--plain", action="store_true", help="Show assistant messages as raw Markdown")
    result.add_argument(
        "--background", action="store_true", help="Detach an explicit run and write output to a private log"
    )
    result.add_argument("--output", type=Path, help="Destination for JSON session export")
    result.add_argument(
        "--all",
        action="store_true",
        help="With 'sessions': also list sessions that have no messages (nothing to resume)",
    )
    result.add_argument(
        "--pick",
        action="store_true",
        help="With 'sessions': choose one from the list by number and resume it",
    )
    result.add_argument(
        "--limit",
        type=int,
        default=LISTING_LIMIT,
        help=f"With 'sessions': how many to list (default {LISTING_LIMIT})",
    )
    return result


def reject_listing_flags(args) -> None:
    # argparse puts these on every command. Ignoring them elsewhere would hand a fresh session to
    # someone who typed 'lillia chat --pick' and thought they had asked for the picker.
    misapplied = [
        flag
        for flag, used in (
            ("--all", args.all),
            ("--pick", args.pick),
            ("--limit", args.limit != LISTING_LIMIT),
        )
        if used
    ]
    if misapplied and args.command != "sessions":
        verb = "only apply" if len(misapplied) > 1 else "only applies"
        raise ValueError(f"{' and '.join(misapplied)} {verb} to the 'sessions' command")


async def run_turn(agent: Agent, prompt: str | None, display: Display) -> tuple[int, bool]:
    display.answer_delivered = False
    display.dashboard.bind(agent)
    display.dashboard.running = True
    output_context = ExitStack()
    if display.dashboard.enabled:
        output_context.enter_context(patch_stdout(raw=True))
    agent.steering_enabled = display.dashboard.enabled

    task = asyncio.create_task(agent.run(prompt))
    heartbeat = asyncio.create_task(display.heartbeat())
    loop = asyncio.get_running_loop()
    terminated = False
    stopping = False

    def submit(text):
        if stopping or task.done() or agent.store.session(agent.sid)["status"] != "running":
            return False  # Keep the draft for the next ordinary prompt.
        if text.strip().startswith("/"):
            say("Slash commands are available after this turn. Ctrl-C stops work.", style="yellow")
            return False
        try:
            agent.queue_message(text)
        except ValueError as exc:
            say(str(exc), style="yellow")
            return False
        return True

    def stop(sig):
        nonlocal stopping, terminated
        terminated = terminated or sig == signal.SIGTERM
        if not stopping:
            stopping = True
            display.end_text()
            say("Stopping this turn and its managed commands…", style="yellow")
            task.cancel()

    old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in old_handlers:
        loop.add_signal_handler(sig, stop, sig)
    try:
        if display.dashboard.enabled:
            say("Working...", style="dim")
        await display.dashboard.start(stop, submit)
        answer = await task
        status = agent.store.session(agent.sid)["status"]
        if status != "answered":
            # A partial result may already have been rendered with its review notice.
            # Only reports that were not delivered through text events need this fallback.
            if not display.answer_delivered:
                say(answer, style="yellow")
            return 2, terminated
        return 0, terminated
    except asyncio.CancelledError:
        say("Interrupted. Progress is saved; /continue resumes the request.", style="yellow")
        return 130, terminated
    except (ModelError, ValueError, TimeoutError) as exc:
        display.end_text()
        detail = str(exc) or "Turn time limit reached"
        say(f"Error: {detail}", style="red")
        say("Progress is saved. Correct the issue and resume this session.", style="dim")
        return 1, terminated
    finally:
        heartbeat.cancel()
        await asyncio.gather(heartbeat, return_exceptions=True)
        display.end_text()
        await display.dashboard.pause()
        display.dashboard.running = False
        agent.steering_enabled = False
        output_context.close()
        say("Tokens overall (conversation + subagents): " + display.dashboard.token_label(), style="dim")
        pending = agent.store.pending_input_count(agent.sid)
        if pending:
            say(
                f"{pending} queued message(s) saved. Use /continue or resume this session to apply them.",
                style="dim",
            )
        for sig, handler in old_handlers.items():
            loop.remove_signal_handler(sig)
            signal.signal(sig, handler)


async def close_session_processes(store, sid, processes, reason, registry=None):
    active = [j.id for j in processes.jobs.values() if not j.waiter.done()]
    await processes.close()
    if registry is not None:
        await registry.close()
    if active:
        store.event(sid, f"jobs_stopped_on_{reason}", {"process_ids": active})
        store.append(
            sid,
            {
                "role": "user",
                "content": f"Runtime notice: {reason} stopped managed jobs "
                + ", ".join(active)
                + ". Inspect actual state before continuing.",
            },
            user_request=False,
        )


async def chat(
    config, store, provider, args, prompt: str | None, interactive: bool, *, subagent_provider=None
) -> int:
    if args.resume:
        session = store.session(args.resume)
        if session["parent_id"] is not None:
            raise ValueError(
                f"Subagent sessions are read-only transcripts. Use 'lillia show {session['id']}' "
                f"or resume parent {session['parent_id']}."
            )
        sid = session["id"]
        cwd = Path(session["cwd"])
        if args.cwd and args.cwd.expanduser().resolve() != cwd:
            raise ValueError("A resumed session keeps its original cwd; start a new session to change it")
        mode = args.mode or session["mode"]
    else:
        cwd = (args.cwd or Path.cwd()).expanduser().resolve()
        mode = config.mode
        if not cwd.is_dir():
            raise ValueError(f"Working directory does not exist: {cwd}")
        sid = store.create(cwd, mode)
    if not cwd.is_dir():
        raise ValueError(f"Session working directory no longer exists: {cwd}")
    with ExitStack() as session_lock:
        session_lock.enter_context(store.lock(sid))
        await report_cloud_account(config)
        try:
            await load_startup_models(config, lambda text: say(text, style="dim"))
        except StartupError as exc:
            say(str(exc), style="red")
            return 1
        input_history = FileHistory(str(store.root / "input-history")) if sys.stdin.isatty() else None
        terminal = PromptSession(history=input_history) if input_history is not None else None

        async def approve(name, arguments):
            if terminal is None:
                return False
            await display.dashboard.pause()
            display.dashboard.approving = True
            say(f"\nApprove {name} in this session?", style="bold yellow")
            say(json.dumps(arguments, ensure_ascii=False, indent=2))
            try:
                # Separate prompt avoids storing approvals in conversation input history.
                response = await PromptSession().prompt_async(
                    "Execute once? [y/N] ",
                    bottom_toolbar=display.dashboard.render,
                    refresh_interval=0.25,
                    style=STYLE,
                    handle_sigint=False,
                )
                return response.strip().lower() in {"y", "yes"}
            except (EOFError, KeyboardInterrupt):
                return False
            finally:
                display.dashboard.approving = False
                if not asyncio.current_task().cancelling():
                    await display.dashboard.start()

        processes = Processes(store.root / "outputs" / sid, cwd, config.max_output_chars)
        registry = ToolRegistry(
            store,
            sid,
            processes,
            mode,
            approve,
            config.max_output_chars,
            config.command_timeout_seconds,
            web_config=config.web,
            retrieval_config=config.retrieval,
            embedding_config=config.embedding_model,
        )
        display = Display(args.verbose, plain=args.plain)
        display.dashboard.history = input_history
        agent = Agent(
            config, store, sid, provider, registry, display.emit, subagent_provider=subagent_provider
        )
        display.dashboard.bind(agent)
        say(f"Lillia-Full {__version__} · {config.model.name}", style="bold magenta")
        if getattr(config, "_profile", None) == "cloud":
            embedding = config.embedding_model
            say(
                f"Cloud profile: chat models through {endpoint_origin(config.model.base_url)} · "
                + (
                    f"embeddings local: {embedding.name} at {endpoint_origin(embedding.base_url)}"
                    if config.retrieval.enabled and embedding.enabled
                    else "embeddings off"
                )
                + " · conversation, tool output and file contents are sent to the cloud models",
                style="dim",
            )
        model_profile = getattr(config, "_model_profile", None)
        if model_profile:
            embedding = config.embedding_model
            worker = config.subagent_model or config.model
            say(
                f"Model profile {model_profile}: parent {config.model.name} · subagents {worker.name} · "
                + (
                    f"embeddings {embedding.name}"
                    if config.retrieval.enabled and embedding.enabled
                    else "embeddings off"
                ),
                style="dim",
            )
        say(f"Session {sid} · {mode} access · {cwd}", style="dim")
        runtime = agent.runtime.snapshot(compare_disk=True)
        extra = config.model.extra_body
        reasoning = (
            f"reasoning effort {extra['reasoning_effort']}"
            if "reasoning_effort" in extra
            else f"reasoning budget {extra.get('reasoning_budget_tokens', 'server default')}"
        )
        say(
            f"Loaded code {runtime['startup_source_sha256'][:12]} · "
            f"output limit {config.model.max_tokens:,} · {reasoning} · "
            f"no-action retries {config.length_retries} · degenerate-stream abort "
            f"{'on' if config.model.degenerate_detection else 'off'}",
            style="dim",
        )
        if config.subagent_model is not None:
            worker = config.subagent_model
            # Generation gates are keyed by backend: children of the parent's own model queue on its gate.
            slots = (
                f"{config.delegation.model_concurrency or 'unlimited'} generation slot(s) shared with the parent"
                if ModelScheduler.backend_key(subagent_provider) == ModelScheduler.backend_key(provider)
                else f"{config.delegation.subagent_model_concurrency or 'unlimited'} generation slot(s)"
            )
            say(
                f"Subagents: {worker.name} · {worker.base_url} · {slots} · "
                f"context {worker.context_window_tokens:,} · output limit {worker.max_tokens:,}",
                style="dim",
            )
        say(f"Resume: lillia --resume {sid}", style="dim")
        if mode == "full":
            say("Tools execute on this host without per-command approval.", style="dim")
        try:
            code = 0
            if prompt is not None or (not interactive and args.resume):
                code, terminated = await run_turn(agent, prompt, display)
                if terminated or not interactive:
                    return code
            if not interactive:
                raise ValueError("Provide a prompt, --prompt-file, or --resume")
            if terminal is None:
                raise ValueError("Interactive chat requires a terminal. Use: lillia run 'your request'")
            say(
                "Type /help for commands. During work, Enter queues a message; Ctrl-C stops work. "
                "Ctrl-D exits at the idle prompt.",
                style="dim",
            )
            while True:
                try:
                    line = (
                        await terminal.prompt_async(
                            "\nYou › ",
                            default=display.dashboard.take_draft(),
                            bottom_toolbar=display.dashboard.render if display.dashboard.enabled else None,
                            refresh_interval=0.25,
                            style=STYLE,
                        )
                    ).strip()
                except KeyboardInterrupt:
                    continue
                except EOFError:
                    return 0
                if not line:
                    continue
                if line in {"/exit", "/quit"}:
                    return 0
                if line == "/help":
                    say(
                        "/continue  Resume outstanding work\n/status    Show session, progress notes and jobs\n"
                        "/clear     Start a fresh conversation; keep saved memories\n"
                        "/agents    List subagent tasks, reported progress and advisory questions\n"
                        "/mode full|assisted|observe  Change autonomy\n/memory    Show durable memory\n"
                        "/forget KEY  Remove a memory\n/exit      Save and exit\n"
                        "During work, type a message and press Enter to queue it for the next step.\n"
                        "Use --prompt-file for large prompts. Jobs are stopped when this CLI exits."
                    )
                elif line == "/clear":
                    if agent.subagents:
                        await agent.subagents.close()
                    await close_session_processes(store, sid, processes, "clear", registry)
                    previous_sid = sid
                    mode = registry.mode
                    session_lock.close()
                    sid = store.create(cwd, mode)
                    session_lock.enter_context(store.lock(sid))
                    processes = Processes(store.root / "outputs" / sid, cwd, config.max_output_chars)
                    registry = ToolRegistry(
                        store,
                        sid,
                        processes,
                        mode,
                        approve,
                        config.max_output_chars,
                        config.command_timeout_seconds,
                        web_config=config.web,
                        retrieval_config=config.retrieval,
                        embedding_config=config.embedding_model,
                    )
                    agent = Agent(
                        config,
                        store,
                        sid,
                        provider,
                        registry,
                        display.emit,
                        subagent_provider=subagent_provider,
                    )
                    display.dashboard.bind(agent)
                    store.event(previous_sid, "conversation_cleared", {"new_session_id": sid})
                    display.end_text()
                    if console.is_terminal:
                        console.clear()
                    say("Fresh conversation started. Saved memories are retained.", style="bold magenta")
                    say(f"Session {sid} · {mode} access · {cwd}", style="dim")
                    say(f"Previous conversation: lillia --resume {previous_sid}", style="dim")
                elif line == "/agents":
                    say(json.dumps(agent.subagents.status() if agent.subagents else [], indent=2))
                elif line == "/status":
                    say(
                        json.dumps(
                            {
                                "session": store.session(sid),
                                "profile": getattr(config, "_profile", None),
                                "model_profile": getattr(config, "_model_profile", None),
                                "model": config.model.name,
                                "runtime": agent.runtime.snapshot(compare_disk=True),
                                "subagents": agent.subagents.status() if agent.subagents else [],
                                "model_calls": agent.scheduler.calls,
                                "max_model_calls": agent.scheduler.max_calls,
                                "tokens": agent.scheduler.telemetry.snapshot(),
                                "progress": await registry.work_notes.inspect(),
                                "running_jobs": [
                                    j.id for j in processes.jobs.values() if not j.waiter.done()
                                ],
                                "queued_messages": store.pending_inputs(sid),
                            },
                            ensure_ascii=False,
                            indent=2,
                        )
                    )
                elif line.startswith("/mode"):
                    parts = line.split()
                    if len(parts) != 2 or parts[1] not in {"full", "assisted", "observe"}:
                        say(f"Current mode: {registry.mode}. Usage: /mode full|assisted|observe")
                    else:
                        if parts[1] == "observe":
                            if agent.subagents:
                                await agent.subagents.close()
                            await processes.close()
                        registry.mode = parts[1]
                        store.update(sid, mode=parts[1])
                        store.event(sid, "mode_changed_by_user", {"mode": parts[1]})
                        say(f"Mode: {registry.mode}")
                elif line == "/memory":
                    memories = store.memories()
                    if memories:
                        say(json.dumps(memories, ensure_ascii=False, indent=2))
                    else:
                        say(
                            "No durable memories saved yet. Conversation history is saved separately. "
                            "Memories are added when the agent uses remember. "
                            "Try: Remember that I prefer concise answers."
                        )
                elif line.startswith("/forget "):
                    store.forget(line.removeprefix("/forget ").strip())
                    say("Memory removed.")
                elif line.startswith("/") and line != "/continue":
                    say("Unknown command; use /help.")
                else:
                    code, terminated = await run_turn(agent, None if line == "/continue" else line, display)
                    if terminated:
                        return code
        finally:
            if agent.subagents:
                await agent.subagents.close()
            await close_session_processes(store, sid, processes, "exit", registry)


async def async_main(args) -> int:
    config = load_config(args.config, cloud=args.cloud, model=args.model)
    for argument, field in (
        ("mode", "mode"),
        ("max_steps", "max_steps"),
        ("max_seconds", "max_turn_seconds"),
    ):
        if getattr(args, argument) is not None:
            setattr(config, field, getattr(args, argument))
    if args.state_dir:
        config.state_dir = str(args.state_dir)
    # A placement profile named by --model already chose the parent; the name is not a gateway ID.
    if args.model and not getattr(config, "_model_profile", None):
        config.model.name = args.model
    config.model.name = MODEL_ALIASES.get(config.model.name, config.model.name)
    if args.base_url:
        config.model.base_url = args.base_url.rstrip("/")
    config.validate()
    reject_listing_flags(args)
    if args.background:
        from .background import launch

        background = launch(args, config)
        say(f"Background process launched: PID {background['pid']}")
        say(f"Log: {background['log']}", soft_wrap=True)
        say(f"Stop gracefully: kill -INT {background['pid']}")
        return 0
    store = Store(Path(config.state_dir))
    provider = ChatProvider(config.model)
    subagent_provider = None
    try:
        if config.subagent_model is not None:
            subagent_provider = ChatProvider(config.subagent_model)
        command = args.command
        if command in {"doctor", "models"}:
            if command == "doctor":
                model_profile = getattr(config, "_model_profile", None)
                say(
                    f"Python: {sys.version.split()[0]}\nState: {store.root}\n"
                    f"Profile: {getattr(config, '_profile', None) or 'local'}\n"
                    + (f"Model profile: {model_profile}\n" if model_profile else "")
                    + f"Endpoint: {config.model.base_url}\nModel: {config.model.name}\nMode: {config.mode}\n"
                    f"API key: {'set' if config.model.api_key else 'not set (local servers and a signed-in Ollama daemon need none)'}"
                )
                await report_cloud_account(config)
            try:
                models = await provider.models()
                if command == "models":
                    for model in models:
                        say(model)
                else:
                    say(f"Model endpoint reachable; {len(models)} models advertised.", style="green")
                    if config.model.name not in models:
                        say(
                            "Configured model is not advertised. Use 'lillia models' to choose one.",
                            style="yellow",
                        )
                        return 1
                    say(
                        "Configured model is advertised. This checks discovery, not generation or tool quality."
                    )
                if command == "doctor" and subagent_provider is not None and config.delegation.enabled:
                    worker = config.subagent_model
                    say(
                        f"Subagent endpoint: {worker.base_url}\nSubagent model: {worker.name}\n"
                        f"Subagent context: {worker.context_window_tokens:,}\n"
                        f"Subagent output limit: {worker.max_tokens:,}\n"
                        f"Subagent generation slots: {config.delegation.subagent_model_concurrency or 'unlimited'}"
                    )
                    worker_models = await subagent_provider.models()
                    if worker.name not in worker_models:
                        say("Configured subagent model is not advertised.", style="yellow")
                        return 1
                    say("Subagent endpoint reachable; configured model is advertised.", style="green")
                if command == "doctor":
                    await load_startup_models(config, lambda text: say(text, style="dim"))
                    say(await embedding_report(config, store))
                return 0
            except StartupError as exc:
                say(str(exc), style="red")
                return 1
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                say(
                    f"Endpoint check failed ({type(exc).__name__}). Check the URL, server, and API key.",
                    style="red",
                )
                return 1
        if command == "index":
            from .index import EvidenceIndex

            if not config.retrieval.enabled:
                say("retrieval.enabled is false in this configuration; nothing to index.", style="yellow")
                return 1
            index = EvidenceIndex(store, None, config.retrieval, config.embedding_model)
            say(
                f"Indexing artifacts under {store.root / 'artifacts'}\n"
                f"Lexical: sqlite FTS5. Vectors: "
                + (
                    f"{config.embedding_model.name} at {config.embedding_model.base_url} "
                    f"({config.embedding_model.dimensions or 'native'} dims, "
                    f"index model {config.embedding_model.vector_model})"
                    if config.embedding_model.enabled
                    else "disabled"
                )
            )
            if args.rebuild:
                say("Rebuilding from scratch; existing vectors are discarded.", style="yellow")
            try:
                report = await index.build(
                    rebuild=args.rebuild,
                    on_progress=lambda n: say(f"  embedded {n:,} chunks…", style="dim"),
                )
            except Exception as exc:
                say(f"Indexing failed ({type(exc).__name__}): {exc}", style="red")
                return 1
            finally:
                await index.close()
            if not report.get("ok"):
                say(report.get("error", "Indexing did not run"), style="yellow")
                return 1
            say(
                f"Artifacts indexed: {report['artifacts_indexed']:,}\n"
                f"Chunks: {report['chunks']:,}\n"
                f"Chunks with vectors: {report['chunks_with_vectors']:,}\n"
                f"Embedded this run: {report['embedded']:,}",
                style="green",
            )
            if report["chunks"] and report["chunks_with_vectors"] < report["chunks"]:
                say(
                    "Vector coverage is partial; semantic recall only sees the covered chunks. "
                    "Lexical search covers everything indexed.",
                    style="yellow",
                )
            return 0
        if command == "sessions":
            # Launching and exiting without a prompt leaves a titleless shell with nothing to
            # resume, and /clear leaves one behind every time. Hide them unless asked.
            listed = store.sessions(limit=max(1, args.limit), with_messages=not args.all)
            if not listed:
                say(
                    "No sessions yet."
                    if args.all
                    else "No sessions with messages yet. Add --all to include empty ones."
                )
                return 0
            # The directory column only earns its width when the listed sessions actually span more
            # than one; most people only ever run in a single project.
            show_cwd = len({session["cwd"] for session in listed}) > 1
            table = Table()
            # min_width keeps the discriminating columns from being squeezed to nothing: left to
            # itself Rich spends the whole terminal on the ~100-character title and collapses them.
            for header, width in (("#", 3), ("ID", 12), ("Status", 11), ("Msgs", 4), ("Agents", 6)):
                table.add_column(header, no_wrap=True, width=width)
            table.add_column("Updated", no_wrap=True, width=8)
            if show_cwd:
                table.add_column("Directory", no_wrap=True, overflow="ellipsis", width=16)
            # One row per session. Wrapping the title made a 127-message session ten lines tall,
            # which is the opposite of scannable when several share the same opening prompt.
            # Size it from the real terminal, or Rich shrinks the fixed columns instead and the
            # numbers you pick by vanish on an 80-column screen.
            spent = 44 + (16 if show_cwd else 0) + 3 * (8 if show_cwd else 7) + 1
            table.add_column(
                "Title", no_wrap=True, overflow="ellipsis", width=max(12, min(46, console.width - spent))
            )
            for number, session in enumerate(listed, 1):
                cells = [
                    str(number),
                    safe(session["id"]),
                    safe(session["status"]),
                    str(session["message_count"]),
                    str(session["subagent_count"]),
                    age(session["updated"]),
                ]
                if show_cwd:
                    cells.append(safe(Path(session["cwd"]).name))
                table.add_row(*cells, safe(session["title"]))
            console.print(table)
            if not args.pick:
                return 0
            if not sys.stdin.isatty():
                say("--pick needs an interactive terminal; resume by ID with --resume.", style="red")
                return 1
            try:
                answer = (await PromptSession().prompt_async("Resume which number (blank cancels)? ")).strip()
            except (EOFError, KeyboardInterrupt):
                say()
                return 0
            if not answer:
                return 0
            if not answer.isdigit() or not 1 <= int(answer) <= len(listed):
                say(f"Enter a number from 1 to {len(listed)}.", style="red")
                return 1
            # Hand off to exactly the path '--resume ID' takes, so picking has no separate semantics.
            args.resume = listed[int(answer) - 1]["id"]
            return await chat(config, store, provider, args, None, True, subagent_provider=subagent_provider)
        if command in {"show", "export"}:
            sid = store.session(args.arguments[0] if args.arguments else args.resume or "latest")["id"]
            if command == "show":
                for _, message in store.messages(sid):
                    say(f"\n{message['role']}", style="bold cyan")
                    say(message.get("content") or "")
                    if message.get("tool_calls"):
                        say(json.dumps(message["tool_calls"], ensure_ascii=False, indent=2))
            else:
                exported = {
                    "session": store.session(sid),
                    "messages": [m for _, m in store.messages(sid)],
                    "events": store.events(sid),
                    "subagents": [
                        {
                            **row,
                            "messages": [m for _, m in store.messages(row["child_id"])],
                            "events": store.events(row["child_id"]),
                        }
                        for row in store.subagents(sid)
                    ],
                }
                data = json.dumps(exported, ensure_ascii=False, indent=2)
                if args.output:
                    # Refuse accidental overwrite of an existing export.
                    with args.output.open("x", encoding="utf-8") as handle:
                        handle.write(data + "\n")
                    say(f"Exported {sid} to {args.output}")
                else:
                    print(data)
            return 0
        if command == "memory":
            say(json.dumps(store.memories(), ensure_ascii=False, indent=2))
            return 0
        if command in {"chat", "run"}:
            prompt = " ".join(args.arguments) or None
            interactive = command == "chat"
        else:
            prompt = " ".join([command, *args.arguments])
            interactive = False
        if args.prompt_file:
            if prompt:
                raise ValueError("Use either a positional prompt or --prompt-file")
            prompt = (
                sys.stdin.read()
                if str(args.prompt_file) == "-"
                else args.prompt_file.read_text(encoding="utf-8")
            )
            # Reading a prompt from stdin is a one-shot operation.
            if str(args.prompt_file) == "-":
                interactive = False
        return await chat(
            config, store, provider, args, prompt, interactive, subagent_provider=subagent_provider
        )
    finally:
        try:
            if subagent_provider is not None:
                await subagent_provider.close()
        finally:
            try:
                await provider.close()
            finally:
                store.close()


def main():
    args = parser().parse_args()
    try:
        raise SystemExit(asyncio.run(async_main(args)))
    except (ValueError, OSError, TypeError) as exc:
        say(f"Error: {exc}", style="red")
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        raise SystemExit(130) from None
