from __future__ import annotations

import json
import math
import uuid


def token_estimate(value) -> int:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return math.ceil(len(text) / 4)


def output_estimate(row) -> int:
    """In-flight output tokens: streamed chunks are usually one token each, and dense numeric or
    hex text runs near one character per token, so the character estimate alone under-counts it."""
    return max(math.ceil(row["output_chars"] / 4), row.get("output_chunks", 0))


def reported_usage(usage) -> int | None:
    """Reasoning tokens are already part of completion_tokens; never add them again."""
    if not isinstance(usage, dict):
        return None

    def valid(value):
        return type(value) is int and value >= 0

    if valid(usage.get("total_tokens")):
        return usage["total_tokens"]
    if valid(usage.get("prompt_tokens")) and valid(usage.get("completion_tokens")):
        return usage["prompt_tokens"] + usage["completion_tokens"]
    return None


class Telemetry:
    """Session-wide token accounting plus transient request/agent status."""

    def __init__(self, store=None, sid=None):
        self.store, self.sid = store, sid
        self.statuses: dict[str, str] = {}
        self.totals: dict[str, int] = {}
        self.estimated = 0
        self.unknown = 0
        self.active: dict[str, dict] = {}
        if store is not None:
            rows = store.db.execute(
                "SELECT id,session_id,kind,body FROM events WHERE "
                "(session_id=? OR session_id IN (SELECT child_id FROM subagents WHERE parent_id=?)) "
                "AND kind IN ('model_usage','model_response','context_compacted') ORDER BY id",
                (sid, sid),
            ).fetchall()
            first_new = {}
            for row in rows:
                if row["kind"] == "model_usage":
                    first_new.setdefault(row["session_id"], row["id"])
            for row in rows:
                body = json.loads(row["body"])
                if row["kind"] == "model_usage":
                    self._add(row["session_id"], body)
                elif row["id"] < first_new.get(row["session_id"], float("inf")):
                    total = reported_usage(body.get("usage"))
                    self._add(
                        row["session_id"],
                        {"tokens": total or 0, "source": "reported" if total is not None else "unknown"},
                    )

    def _add(self, sid, record):
        self.totals[sid] = self.totals.get(sid, 0) + record["tokens"]
        self.estimated += record["source"] == "estimated"
        self.unknown += record["source"] == "unknown"

    def begin(self, sid, messages, tools):
        request_id = uuid.uuid4().hex
        self.active[request_id] = {
            "sid": sid,
            "input_estimate": token_estimate([messages, tools]),
            "output_chars": 0,
            "output_chunks": 0,
            "usage": None,
            "started": False,
        }
        self.statuses[sid] = "waiting for model"
        return request_id

    def progress(self, request_id, data):
        row = self.active[request_id]
        row["started"] = True
        row["output_chars"] = data.get("output_chars", row["output_chars"])
        row["output_chunks"] = data.get("output_chunks", row["output_chunks"])
        if reported_usage(data.get("usage")) is not None:
            row["usage"] = data["usage"]
        if data.get("phase"):
            self.statuses[row["sid"]] = data["phase"]

    def finish(self, request_id, reply=None):
        row = self.active.pop(request_id)
        usage = reply.usage if reply and reported_usage(reply.usage) is not None else row["usage"]
        total = reported_usage(usage)
        source = "reported"
        if total is None:
            if reply is not None:
                total = row["input_estimate"] + token_estimate(reply.message)
                source = "estimated"
            elif row["started"]:
                total = row["input_estimate"] + output_estimate(row)
                source = "estimated"
            else:
                total, source = 0, "unknown"
        record = {"request_id": request_id, "tokens": total, "source": source, "usage": usage or {}}
        self._add(row["sid"], record)
        if self.store is not None:
            self.store.event(row["sid"], "model_usage", record)
        self.statuses[row["sid"]] = "working"

    def snapshot(self):
        active_total, active_estimated = 0, False
        for row in self.active.values():
            if not row["started"]:
                continue
            reported = reported_usage(row["usage"])
            if reported is not None:
                active_total += reported
            else:
                active_total += row["input_estimate"] + output_estimate(row)
                active_estimated = True
        return {
            "tokens": sum(self.totals.values()) + active_total,
            "estimated": bool(self.estimated or active_estimated),
            "unknown_requests": self.unknown,
            "in_flight": len(self.active),
        }
