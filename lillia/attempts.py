"""Compact, durable observations of attempted actions, including negative outcomes.

These records are historical reference data. A successful tool is not necessarily a
successful investigation; a failed tool is not proof that its hypothesis was wrong.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import suppress

from .recall_history import SUPPORTED_TOOLS


def signature(name: str, arguments: dict) -> str:
    value = json.dumps([name, arguments], sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(value.encode()).hexdigest()


def excerpt(value, limit):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + "… [excerpt]"


def observed_status(name, result):
    if result.get("error") == "interrupted_outcome_unknown":
        return "uncertain"
    if result.get("timed_out"):
        return "timed_out"
    if result.get("stopped"):
        return "stopped"
    if result.get("error") or result.get("ok") is not True:
        return "failed"
    if result.get("running") is True:
        return "running"
    if name in {"shell", "execute_program"}:
        code = result.get("exit_code")
        if type(code) is not int or result.get("running") is not False:
            return "uncertain"
        return "succeeded" if code == 0 else "failed"
    return "succeeded"


class AttemptMemory:
    def __init__(self, store, sid):
        self.store, self.sid = store, sid
        self.reported_failure = False

    def record(self, name, arguments, result, call_id, event_id):
        """Called only for executed tools. Polls update their launch, never add attempts."""
        try:
            self._record(name, arguments, result, call_id, event_id)
        except Exception as exc:
            # A memory storage problem must not turn a completed side effect into a
            # failed tool result, which would itself encourage a dangerous retry.
            if not self.reported_failure:
                self.reported_failure = True
                with suppress(Exception):
                    self.store.event(
                        self.sid, "attempt_memory_unavailable", {"error_type": type(exc).__name__}
                    )

    def _record(self, name, arguments, result, call_id, event_id):
        process_id = result.get("process_id")
        if name in {"poll_process", "stop_process"}:
            process_id = process_id or arguments.get("process_id")
            if not process_id or result.get("running") is not False:
                return
            row = self.store.db.execute(
                "SELECT * FROM attempts WHERE session_id=? AND process_id=? AND status='running' "
                "ORDER BY event_id DESC LIMIT 1",
                (self.sid, process_id),
            ).fetchone()
            if row is None:
                return
            name, arguments, call_id = row["name"], json.loads(row["arguments"]), row["call_id"]
            request_id = row["request_id"]
        elif name in SUPPORTED_TOOLS:
            request = self.store.latest_request(self.sid)
            request_id = request[0] if request else 0
        else:
            return
        status = observed_status(name, result)
        # Keep decisive status fields even when the output itself is very large.
        details = {
            key: result[key]
            for key in (
                "error",
                "exit_code",
                "running",
                "timed_out",
                "stopped",
                "saved_output",
                "log_path",
            )
            if key in result
        }
        outcome = excerpt(details, 700) + "\n" + excerpt(result, 1000)
        artifact = None
        if status != "running":
            text = (
                f"Historical attempt: {name}\nObserved outcome: {status}\n"
                f"Action: {excerpt(arguments, 500)}\n"
                f"Result: {excerpt(outcome, 650)}\n"
                f"Session: {self.sid}; request: {request_id}; action evidence: {call_id}; "
                f"outcome event: {event_id}.\n"
                "Historical observation, not current verification or proof of a hypothesis."
            )
            artifact = str(self.store.save_artifact(self.sid, text))
        with self.store.db:
            self.store.db.execute(
                "INSERT OR REPLACE INTO attempts "
                "(session_id,call_id,request_id,signature,name,arguments,process_id,status,outcome,event_id,artifact) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    self.sid,
                    call_id,
                    request_id,
                    signature(name, arguments),
                    name,
                    json.dumps(arguments, ensure_ascii=False),
                    process_id,
                    status,
                    outcome,
                    event_id,
                    artifact,
                ),
            )

    def find(self, calls, request_id, limit=3):
        found = []
        for call in calls:
            function = call["function"]
            arguments = json.loads(function["arguments"])
            row = self.store.db.execute(
                "SELECT * FROM attempts WHERE session_id=? AND request_id=? AND signature=? "
                "ORDER BY event_id DESC LIMIT 1",
                (self.sid, request_id, signature(function["name"], arguments)),
            ).fetchone()
            if row:
                found.append({**dict(row), "arguments": arguments, "proposed_call_id": call["id"]})
            if len(found) >= limit:
                break
        return found
