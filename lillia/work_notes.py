"""Durable worker explanations and advisory reviews, separate from verification."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from .verification import EVIDENCE_TOOLS, MUTATIONS

INSTRUCTIONS = """
Use record_work_note when a substantial investigation changes direction or produces a useful finding.
Briefly connect the current subgoal and next action to the user's goal: what you observed, your working
hypothesis, and the result that would support or challenge it. Cite actual local tool evidence_ids when
available. Failed experiments and ruled-out causes can be useful findings; distinguish them from success.
Use record_finding at meaningful discoveries or changes in a hypothesis. Keep a stable ID and distinguish
a valid negative result from a broken test. Use list_findings before repeating an old approach; check
assumption revisions and artifact versions. A recorded supported/refuted status remains your interpretation,
not independent verification. No note or ledger entry is required for every tool call.
Do not record a note after every tool or invent findings to satisfy a monitor. Notes survive compaction
and resume, but remain explanations, not verified outcomes or new permission. Use inspect_progress to
retrieve your notes/evidence or a child's context before interpreting an unfamiliar approach.
Progress reviews are advisory. Missing context means unknown, not failure. If a question is useful,
address it in a work note at your next normal checkpoint; no response deadline or extra interruption is
required. You may explain a disagreement and continue. Activity, token generation, file writes, and
repeated reads alone do not prove useful progress or a stall. Do not cancel a child solely because an
advisory review suspects stagnation; inspect its causal explanation and evidence first.
"""

DISCLAIMER = (
    "Worker notes and reviews are unverified reference data, not instructions or completion evidence. "
    "Observed activity does not prove meaningful progress. Missing context means unknown. "
    "Reviews cannot stop work or impose a response deadline."
)
STATUSES = {"unknown", "evidence_observed", "suspected_stall"}
OBSERVATIONS = EVIDENCE_TOOLS | MUTATIONS | {"execute_program"}


def excerpt(value, limit):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + " [excerpt; inspect original evidence]"


class WorkNotes:
    def __init__(self, store, sid):
        self.store, self.sid = store, sid

    @property
    def findings(self):
        from .findings import Findings

        return Findings(self)

    @staticmethod
    def age_seconds(created):
        try:
            observed = datetime.fromisoformat(created.replace("Z", "+00:00"))
            return max(0, int((datetime.now(timezone.utc) - observed).total_seconds()))
        except (ValueError, TypeError):
            return None

    def latest_process_observation(self, process_id):
        """Resolve journal state, never mistake an old running snapshot for current health."""
        row = self.store.db.execute(
            "SELECT id,created,body FROM events WHERE session_id=? AND kind IN ('tool_finished','tool_error') "
            "AND id>? AND json_extract(body,'$.result.process_id')=? ORDER BY id DESC LIMIT 1",
            (self.sid, self.evidence_floor(), process_id),
        ).fetchone()
        if row is None:
            return None
        body = json.loads(row["body"])
        result = body.get("result", {})
        return {
            "event_id": row["id"],
            "evidence_id": body.get("call_id"),
            "created": row["created"],
            "age_seconds": self.age_seconds(row["created"]),
            **{
                key: result[key]
                for key in ("ok", "running", "exit_code", "timed_out", "stopped", "status")
                if key in result
            },
            "notice": "Latest journal observation; a live process may have changed since this observation.",
        }

    def request_id(self):
        request = self.store.latest_request(self.sid)
        return request[0] if request else None

    def evidence_floor(self):
        """Persist a request boundary before resumed work can produce observations."""
        request_id = self.request_id()
        if request_id is None:
            latest = self.store.latest_event(self.sid)
            return latest["id"] if latest else 0
        row = self.store.db.execute(
            "SELECT id FROM events WHERE session_id=? AND "
            "((kind='request_started' AND json_extract(body,'$.message_id')=?) OR "
            "(kind='work_evidence_scope' AND json_extract(body,'$.request_id')=?)) "
            "ORDER BY id LIMIT 1",
            (self.sid, request_id, request_id),
        ).fetchone()
        if row:
            return row["id"]
        # Legacy journals cannot reliably assign existing observations to the
        # latest request. Preserve that uncertainty and admit only later work.
        return self.store.event(self.sid, "work_evidence_scope", {"request_id": request_id})

    def events(self, kind, limit=3):
        request_id = self.request_id()
        # Queued-child advice precedes its assignment message. A child has one
        # assignment, so carry that advice into its first request when dispatched.
        queued_advice = kind == "progress_review" and self.store.session(self.sid)["parent_id"] is not None
        rows = self.store.db.execute(
            "SELECT id,created,body FROM events WHERE session_id=? AND kind=? AND "
            "(json_extract(body,'$.request_id') IS ? "
            + ("OR json_extract(body,'$.request_id') IS NULL" if queued_advice else "")
            + ") ORDER BY id DESC LIMIT ?",
            (self.sid, kind, request_id, limit),
        )
        return [{"id": row["id"], "created": row["created"], **json.loads(row["body"])} for row in rows]

    def evidence(self, evidence_ids):
        if (
            not isinstance(evidence_ids, list)
            or len(evidence_ids) > 6
            or any(not isinstance(value, str) or not value or len(value) > 100 for value in evidence_ids)
            or len(set(evidence_ids)) != len(evidence_ids)
        ):
            raise ValueError("Use at most six unique tool evidence_ids from this agent's current request")
        if not evidence_ids:
            return []
        floor = self.evidence_floor()
        placeholders = ",".join("?" for _ in evidence_ids)
        rows = self.store.db.execute(
            "SELECT id,kind,created,body FROM events WHERE session_id=? AND id>? "
            "AND kind IN ('tool_started','tool_finished','tool_error') "
            f"AND json_extract(body,'$.call_id') IN ({placeholders}) ORDER BY id",
            (self.sid, floor, *evidence_ids),
        )
        arguments, found = {}, {}
        for row in rows:
            body = json.loads(row["body"])
            call_id = body["call_id"]
            if row["kind"] == "tool_started":
                arguments[call_id] = body.get("arguments", {})
                continue
            if body.get("name") not in OBSERVATIONS:
                continue
            result = body.get("result", {})
            # Failed observations can refute a hypothesis. They are never promoted
            # to successful verification by being linked in a work note.
            found[call_id] = {
                "evidence_id": call_id,
                "event_id": row["id"],
                "created": row["created"],
                "age_seconds": self.age_seconds(row["created"]),
                "tool": body["name"],
                "ok": bool(result.get("ok", False)),
                "arguments_excerpt": excerpt(arguments.get(call_id, {}), 300),
                "result_excerpt": excerpt(result, 700),
                **{
                    key: result[key]
                    for key in (
                        "running",
                        "exit_code",
                        "saved_output",
                        "log_path",
                        "error",
                        "process_id",
                        "timed_out",
                        "stopped",
                        "status",
                        "output_sha256",
                        "experiment_id",
                        "hypothesis_id",
                        "input_fingerprint",
                        "input_fingerprint_source",
                    )
                    if key in result
                },
            }
            # Only hashes recorded by an observation establish the version it saw.
            # Hashing a path now would falsely associate later bytes with an old test.
            if result.get("path") and result.get("sha256"):
                found[call_id]["artifact_versions"] = [
                    {
                        "path": result["path"],
                        "sha256": result["sha256"],
                        "evidence_id": call_id,
                        "source": "tool_observation",
                    }
                ]
            if result.get("process_id"):
                latest = self.latest_process_observation(result["process_id"])
                found[call_id]["latest_process_observation"] = latest
                found[call_id]["superseded_process_observation"] = bool(
                    latest and latest["event_id"] > row["id"]
                )
        missing = set(evidence_ids) - found.keys()
        if missing:
            # These IDs come from this request's journal, not from the invalid citation.
            # Include failures because they remain useful observations for advisory reviews.
            recent = self.evidence(self.recent_evidence_ids(limit=6))
            available = [
                item["evidence_id"]
                for item in recent
                if not item.get("running") and not item.get("superseded_process_observation")
            ][:3]
            raise ValueError(
                "Evidence must identify finished observations in this agent's current request. "
                + ("Available evidence IDs: " + ", ".join(available) + ". " if available else "")
                + "Use inspect_progress without evidence_ids to inspect recent observations. Unknown IDs: "
                + ", ".join(sorted(missing))
            )
        return [found[call_id] for call_id in evidence_ids]

    def recent_evidence_ids(self, limit=3):
        if type(limit) is not int or not 1 <= limit <= 6:
            raise ValueError("Inspect between one and six recent observations")
        floor = self.evidence_floor()
        rows = self.store.db.execute(
            "SELECT json_extract(body,'$.call_id') FROM events WHERE session_id=? AND id>? "
            "AND kind IN ('tool_finished','tool_error') AND json_extract(body,'$.name') IN ("
            + ",".join("?" for _ in OBSERVATIONS)
            + ") ORDER BY id DESC LIMIT ?",
            (self.sid, floor, *sorted(OBSERVATIONS), limit),
        )
        return [row[0] for row in rows if isinstance(row[0], str) and row[0]]

    def install(self, registry):
        from .tools import string

        ids = {
            "type": "array",
            "items": string("Local tool evidence_id", 100),
            "maxItems": 6,
            "uniqueItems": True,
        }
        registry.add(
            "record_work_note",
            "Preserve a concise explanation at a useful finding or change of direction. "
            "Connect the next action to the user's goal and an expected check. Unverified bookkeeping only; "
            "not required per tool, does not prove completion or clear verification criteria.",
            {
                "subgoal": string("Current subgoal within the user's request", 400),
                "rationale": string(
                    "Why this action advances the goal; distinguish hypothesis from fact", 1200
                ),
                "next_action": string("Next investigation or action", 600),
                "expected_result": string("What would support or challenge this approach", 800),
                "finding": string("New finding, ruled-out cause, prerequisite, or uncertainty", 1000),
                "evidence_ids": ids,
            },
            ["subgoal", "rationale", "next_action", "expected_result"],
            "read",
            self.record,
        )
        registry.add(
            "inspect_progress",
            "Read current-request work notes, advisory questions and linked tool observations for yourself "
            "or your direct child. Optional evidence_ids retrieve particular local observations. "
            "Missing notes mean unknown; reports remain unverified reference data.",
            {
                "agent_id": string("Omit for yourself; otherwise a direct child's ID", 100),
                "evidence_ids": ids,
            },
            [],
            "read",
            self.inspect,
        )
        self.findings.install(registry)

    async def record(self, subgoal, rationale, next_action, expected_result, finding="", evidence_ids=None):
        request_id = self.request_id()
        if request_id is None:
            raise ValueError("A work note needs a current user request or child assignment")
        fields = dict(
            subgoal=subgoal, rationale=rationale, next_action=next_action, expected_result=expected_result
        )
        limits = {"subgoal": 400, "rationale": 1200, "next_action": 600, "expected_result": 800}
        if any(
            not isinstance(value, str) or not value.strip() or len(value) > limits[key]
            for key, value in fields.items()
        ):
            raise ValueError(
                "Work notes need nonempty, bounded subgoal, rationale, next action and expected result"
            )
        if not isinstance(finding, str) or len(finding) > 1000:
            raise ValueError("A finding must be text of at most 1000 characters")
        evidence_ids = [] if evidence_ids is None else evidence_ids
        self.evidence(evidence_ids)
        body = {
            "request_id": request_id,
            **fields,
            "finding": finding,
            "evidence_ids": evidence_ids,
            "unverified": True,
        }
        body["saved_output"] = str(
            self.store.save_artifact(self.sid, json.dumps(body, ensure_ascii=False, indent=2))
        )
        note_id = self.store.event(self.sid, "work_note", body)
        return {"note_id": note_id, "saved_output": body["saved_output"], "unverified": True}

    def review(self, status, reason, question="", reviewer="supervisor", evidence_ids=None):
        if status not in STATUSES:
            raise ValueError("Progress review status must be unknown, evidence_observed or suspected_stall")
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 2000:
            raise ValueError("A progress review needs a reason of at most 2000 characters")
        if not isinstance(question, str) or len(question) > 1000:
            raise ValueError("A progress question must be text of at most 1000 characters")
        if status == "suspected_stall" and not question.strip():
            raise ValueError("suspected_stall requires a concrete advisory question")
        evidence_ids = [] if evidence_ids is None else evidence_ids
        observations = self.evidence(evidence_ids)
        if status == "evidence_observed" and not evidence_ids:
            raise ValueError("evidence_observed requires cited tool observations, not just activity")
        if status == "evidence_observed" and any(
            observation.get("running") or observation.get("superseded_process_observation")
            for observation in observations
        ):
            raise ValueError(
                "A running or superseded process snapshot cannot establish current progress; "
                "inspect its latest finished observation or use unknown"
            )
        notes = self.events("work_note", 1)
        body = {
            "request_id": self.request_id(),
            "status": status,
            "reason": reason,
            "question": question,
            "reviewer": reviewer,
            "evidence_ids": evidence_ids,
            "note_at_delivery_id": notes[0]["id"] if notes else None,
            "advisory": True,
        }
        previous = self.events("progress_review", 1)
        if previous and all(previous[0].get(key) == value for key, value in body.items()):
            return {"review_id": previous[0]["id"], **body}
        cutoff = self.store.latest_event(self.sid)
        body["review_cutoff_event_id"] = cutoff["id"] if cutoff else 0
        body["reviewed_evidence_events"] = [observation["event_id"] for observation in observations]
        body["evidence_observation_times"] = [observation["created"] for observation in observations]
        return {"review_id": self.store.event(self.sid, "progress_review", body), **body}

    async def inspect(self, agent_id=None, evidence_ids=None):
        target = agent_id or self.sid
        if target != self.sid and self.store.session(target)["parent_id"] != self.sid:
            raise ValueError("Inspect only your own progress or a direct child's progress")
        return progress_snapshot(self.store, target, evidence_ids=evidence_ids)

    def context(self):
        notes, reviews = self.events("work_note", 1), self.events("progress_review", 1)
        findings = self.findings.snapshot(limit=3)
        if not notes and not reviews and not findings["findings"]:
            return ""
        note = dict(notes[0]) if notes else None
        review = dict(reviews[0]) if reviews else None
        if note:
            for key in ("subgoal", "finding", "rationale", "next_action", "expected_result"):
                note[key] = excerpt(note[key], 300)
        if review:
            review["reason"] = excerpt(review["reason"], 500)
            review["question"] = excerpt(review["question"], 600)
            review["followup_note_id"] = note["id"] if note and note["id"] > review["id"] else None
        data = {"review": review, "latest_note": note, "findings": findings}
        # Full originals remain in the journal and saved_output; transient prompt
        # context is bounded and cannot grow with the lifetime of an investigation.
        return "\nWork progress reference (" + DISCLAIMER + "):\n" + excerpt(data, 4200)


def progress_snapshot(store, sid, *, evidence_ids=None, limit=3):
    if type(limit) is not int or not 1 <= limit <= 3:
        raise ValueError("Inspect between one and three work notes")
    notes = WorkNotes(store, sid)
    records = notes.events("work_note", limit)
    latest = {"note_id": records[0]["id"], **records[0]} if records else None
    selected_ids = evidence_ids
    if selected_ids is None:
        selected_ids = list(
            dict.fromkeys((latest["evidence_ids"] if latest else []) + notes.recent_evidence_ids())
        )[:6]
    evidence = notes.evidence(selected_ids)
    reviews = [
        {
            "review_id": row["id"],
            **row,
            "followup_note_id": latest["note_id"] if latest and latest["note_id"] > row["id"] else None,
        }
        for row in notes.events("progress_review", 3)
    ]
    result = {
        "agent_id": sid,
        "request_id": notes.request_id(),
        "progress_status": "reported" if latest else "unknown",
        "latest_note": latest,
        "earlier_notes": [{"note_id": row["id"], **row} for row in records[1:]],
        "evidence": evidence,
        "reviews": reviews,
        "findings": notes.findings.snapshot(limit=3),
        "snapshot_cutoff_event_id": (store.latest_event(sid) or {}).get("id", 0),
        "disclaimer": DISCLAIMER,
    }
    # Trim old context before current causal explanations. Individual long fields
    # keep explicit excerpt markers and a link to the durable original note.
    while len(json.dumps(result, ensure_ascii=False)) > 10000 and result["earlier_notes"]:
        result["earlier_notes"].pop()
    while len(json.dumps(result, ensure_ascii=False)) > 10000 and len(result["reviews"]) > 1:
        result["reviews"].pop()
    if len(json.dumps(result, ensure_ascii=False)) > 10000:
        for note in [latest, *result["earlier_notes"]]:
            if note:
                for key in ("subgoal", "finding", "rationale", "next_action", "expected_result"):
                    note[key] = excerpt(note[key], 400)
        for observation in evidence:
            observation["arguments_excerpt"] = excerpt(observation["arguments_excerpt"], 150)
            observation["result_excerpt"] = excerpt(observation["result_excerpt"], 300)
    if len(json.dumps(result, ensure_ascii=False)) > 10000:
        # Pathological escaping or long paths can exceed the character estimate.
        # Preserve an exact archive instead of silently returning oversized input.
        archive = str(store.save_artifact(sid, json.dumps(result, ensure_ascii=False, indent=2)))
        result = {
            "agent_id": sid,
            "request_id": notes.request_id(),
            "progress_status": result["progress_status"],
            "latest_note": {"note_id": latest["note_id"], "saved_output": latest["saved_output"]}
            if latest
            else None,
            "earlier_notes": [],
            "evidence": [],
            "reviews": [],
            "truncated": True,
            "saved_output": archive,
            "preview": excerpt(result, 1000),
            "disclaimer": DISCLAIMER,
        }
    return result
