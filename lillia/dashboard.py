from __future__ import annotations

import asyncio
import os
import re
import signal
import sys

from prompt_toolkit import PromptSession
from prompt_toolkit.application import get_app
from prompt_toolkit.document import Document
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from prompt_toolkit.utils import get_cwidth
from rich.console import Console
from rich.text import Text

REASONING_CHARS = 16000
WRAP_CONSOLE = Console()

STYLE = Style.from_dict(
    {
        # Override prompt_toolkit's reversed toolbar colors explicitly.
        "bottom-toolbar": "noreverse bg:#303030 #cdd6df",
        "bottom-toolbar.text": "noreverse bg:#303030 #cdd6df",
        "dashboard.heading": "noreverse bg:#303030 #8bd5ff bold",
        "dashboard.active": "noreverse bg:#303030 #f5cf83",
        "dashboard.done": "noreverse bg:#303030 #a6d189",
        "reasoning.border": "noreverse bg:#202630 #8bd5ff",
        "reasoning.text": "noreverse bg:#202630 #cdd6df",
    }
)


def clip(text, width):
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", str(text))
    result, used = [], 0
    for char in text:
        size = get_cwidth(char)
        if used + size > width:
            break
        result.append(char)
        used += size
    return "".join(result) + " " * (width - used)


class AgentDashboard:
    """A prompt_toolkit bottom toolbar, shared by input, work, and approval prompts."""

    def __init__(self):
        self.agent = None
        self.children = {}
        self.running = False
        self.approving = False
        self.prompt_task = None
        self.stop_callback = None
        self.submit_callback = None
        self.history = None
        self.draft = Document()
        self.app = None
        self.reasoning_text = ""
        self.reasoning_truncated = False

    @property
    def enabled(self):
        return sys.stdin.isatty() and sys.stdout.isatty() and os.environ.get("TERM") != "dumb"

    def bind(self, agent):
        self.agent = agent
        self.children = {row["agent_id"]: row for row in agent.subagents.status()} if agent.subagents else {}
        self.update_reasoning({"state": "end"})

    def update_reasoning(self, data):
        if data["state"] == "delta":
            # Preserve paragraphs, but never interpret model text as terminal controls or markup.
            text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]", "", data["text"])
            text = self.reasoning_text + text
            self.reasoning_truncated |= len(text) > REASONING_CHARS
            self.reasoning_text = text[-REASONING_CHARS:]
        else:
            self.reasoning_text = ""
            self.reasoning_truncated = False
        if self.app is not None:
            self.app.invalidate()

    def reasoning_lines(self, width, height):
        rows = max(0, min(6, height - 9))
        if not self.running or not self.reasoning_text or not rows:
            return []
        lines = Text(self.reasoning_text).wrap(WRAP_CONSOLE, width - 4)
        scrolled = self.reasoning_truncated or len(lines) > rows

        def border(left, label, right):
            title = clip(label, width - 2).rstrip()
            return left + title + "─" * (width - 2 - get_cwidth(title)) + right

        return [
            ("class:reasoning.border", border("┌", " Reasoning (live) ", "┐")),
            *[("class:reasoning.text", "│ " + clip(line.plain, width - 4) + " │") for line in lines[-rows:]],
            ("class:reasoning.border", border("└", " latest lines " if scrolled else "", "┘")),
        ]

    def update_child(self, data):
        self.children[data["agent_id"]] = dict(data)

    def token_label(self):
        data = self.agent.scheduler.telemetry.snapshot()
        label = f"{'~' if data['estimated'] else ''}{data['tokens']:,}"
        if data["estimated"]:
            label += " (estimated)"
        if data["unknown_requests"]:
            label += f" + {data['unknown_requests']} unmetered request(s)"
        return label

    def render(self, width=None, height=None):
        if self.agent is None:
            return []
        size = get_app().output.get_size()
        width = max(8, width if width is not None else size.columns)
        height = height if height is not None else size.rows
        meter = self.agent.scheduler.telemetry
        active = [row for row in self.children.values() if row["status"] in {"started", "running"}]
        queued = [row for row in self.children.values() if row["status"] == "queued"]
        completed = [row for row in self.children.values() if row not in active and row not in queued]
        # Keep active investigations visible first, followed by the most recent completed ones.
        rows = active + queued + list(reversed(completed))
        reasoning = self.reasoning_lines(width, height)
        capacity = max(0, min(8, height - 8 - len(reasoning)))
        visible = rows[:capacity]
        phase = (
            "awaiting approval"
            if self.approving
            else (meter.statuses.get(self.agent.sid, "working") if self.running else "ready")
        )
        pending = self.agent.store.pending_input_count(self.agent.sid)
        if pending:
            phase += f" | {pending} message(s) queued"
        lines = [
            (
                "class:dashboard.heading",
                f" Agents | {len(active)} active subagent(s)"
                + (f" | {len(queued)} queued" if queued else ""),
            ),
            ("", " Tokens overall: " + self.token_label()),
            ("", " Current conversation + subagents (input + output, including thinking)"),
            ("class:dashboard.active" if self.running else "", f" Parent: {phase}"),
        ]
        for row in visible:
            status = meter.statuses.get(row["agent_id"], row["status"])
            style = "class:dashboard.active" if row in active else "class:dashboard.done"
            if row["status"] not in {"started", "running", "completed"}:
                style = "class:dashboard.active"
            lines.append((style, f" {row['name']} [{row['agent_id'][:6]}]: {status}"))
        if len(rows) > capacity:
            lines.append(("", f" + {len(rows) - capacity} more; /agents lists all"))
        if not rows:
            lines.append(("", " No subagents in this conversation"))
        # Short terminals retain the total and parent state, rather than covering the input area.
        if height < 10:
            lines = [lines[1], lines[3]]
        lines = reasoning + lines
        fragments = []
        for i, (style, text) in enumerate(lines):
            fragments.append((style, clip(text, width)))
            if i < len(lines) - 1:
                fragments.append(("", "\n"))
        return fragments

    async def start(self, stop_callback=None, submit_callback=None):
        if stop_callback is not None:
            self.stop_callback = stop_callback
        if submit_callback is not None:
            self.submit_callback = submit_callback
        if not self.enabled or self.prompt_task is not None:
            return
        bindings = KeyBindings()

        @bindings.add("enter")
        def submit(event):
            buffer = event.current_buffer
            if not buffer.text.strip():
                return
            if self.submit_callback and self.submit_callback(buffer.text):
                buffer.append_to_history()
                buffer.reset()

        @bindings.add("c-c")
        @bindings.add("c-d")
        def interrupt(event):
            if self.stop_callback:
                self.stop_callback(signal.SIGINT)

        session = PromptSession(
            bottom_toolbar=self.render,
            refresh_interval=0.25,
            style=STYLE,
            key_bindings=bindings,
            history=self.history,
            erase_when_done=True,
            reserve_space_for_menu=0,
        )
        self.app = session.app
        # Coalesce bursts of tokens into smooth frames without delaying the HTTP stream.
        self.app.min_redraw_interval = 1 / 30
        self.prompt_task = asyncio.create_task(
            session.prompt_async(
                "You (queue) › ",
                default=self.take_draft(),
                handle_sigint=False,
            )
        )
        await asyncio.sleep(0)

    async def pause(self):
        if self.prompt_task is not None:
            if self.app is not None:
                self.draft = self.app.current_buffer.document
            self.prompt_task.cancel()
            await asyncio.gather(self.prompt_task, return_exceptions=True)
            self.prompt_task = None
        self.app = None

    def take_draft(self):
        draft, self.draft = self.draft, Document()
        return draft
