"""Elapsed task time and optional deadlines explicitly enabled in configuration."""

from __future__ import annotations

import time
from datetime import datetime, timezone


class TaskBudget:
    def __init__(self, store, sid):
        self.store, self.sid = store, sid
        self.started = time.time()
        self.deadline = None
        self.reserve = 0
        self.source = "unlimited"

    def begin(self, prompt, configured_seconds=0, inherited_deadline=None):
        now = time.time()
        request = self.store.latest_request(self.sid)
        request_id = request[0] if request else None
        prior = self.store.latest_event(self.sid, "task_budget")
        saved = (
            prior["body"]
            if prompt is None and prior and prior["body"].get("request_id") == request_id
            else None
        )
        self.started = saved["started"] if saved else now
        self.deadline, self.reserve, self.source = None, 0, "unlimited"
        limits = []
        if configured_seconds:
            deadline = now + configured_seconds
            # A continued request keeps an explicitly configured deadline only
            # while a positive limit remains enabled. Zero disarms old limits.
            # Legacy prompt-inferred deadlines must never be restored.
            if saved and saved["source"] == "configured turn limit" and saved["deadline"] is not None:
                deadline = min(deadline, saved["deadline"])
                if deadline == saved["deadline"]:
                    self.reserve = saved["reserve"]
            limits.append((deadline, "configured turn limit"))
        if inherited_deadline is not None:
            limits.append((inherited_deadline, "parent deadline"))
        if limits:
            self.deadline, self.source = min(limits)
            if not self.reserve:
                self.reserve = min(300, max(0, self.deadline - now) / 10)
        self.store.event(
            self.sid,
            "task_budget",
            {
                "request_id": request_id,
                "started": self.started,
                "deadline": self.deadline,
                "reserve": self.reserve,
                "source": self.source,
            },
        )

    def remaining(self):
        return None if self.deadline is None else max(0, self.deadline - time.time())

    def closing(self):
        remaining = self.remaining()
        return remaining is not None and remaining <= self.reserve

    def context(self):
        if self.deadline is None:
            return (
                "No task deadline configured. Take the time needed to complete and verify the request. "
                "Elapsed time, runtime status updates and advisory reviews do not imply user impatience "
                "or a request to hurry. Continue useful work and wait for needed results."
            )
        elapsed = max(0, int(time.time() - self.started))
        deadline = datetime.fromtimestamp(self.deadline, timezone.utc).isoformat(timespec="seconds")
        remaining = int(self.remaining())
        phase = (
            "Closing phase: stop expanding scope and spawning children. Verify existing work, collect reports, "
            "clean up owned experiments, and write a final report with limitations now."
            if self.closing()
            else "Preserve time for verification, cleanup and a final report."
        )
        return (
            f"Elapsed task time: {elapsed}s. Hard deadline: {deadline} ({self.source}); "
            f"remaining: {remaining}s; closing reserve: {int(self.reserve)}s. {phase}"
        )
