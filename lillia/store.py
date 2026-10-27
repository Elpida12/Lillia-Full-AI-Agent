from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path

RUNTIME_MESSAGES = (
    "Subagent task reports (",
    "Subagent investigation reports (",
    "Continue the outstanding request.",
    "Verification feedback (runtime):",
    "Orchestration feedback (runtime):",
    "Delegation feedback (runtime):",
    "Runtime notice:",
)


def is_user_request(message: dict) -> bool:
    content = message.get("content")
    return (
        message.get("role") == "user"
        and isinstance(content, str)
        and not content.startswith(RUNTIME_MESSAGES)
    )


class Store:
    """Durable conversation and tool journal. Never replay an unresolved tool call."""

    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        db_path = self.root / "sessions.sqlite3"
        fd = os.open(db_path, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(fd)
        self.db = sqlite3.connect(db_path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY, title TEXT NOT NULL, cwd TEXT NOT NULL,
                mode TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'idle',
                summary TEXT NOT NULL DEFAULT '', summary_through INTEGER NOT NULL DEFAULT 0,
                created TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                updated TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id), body TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS messages_session ON messages(session_id, id);
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id), kind TEXT NOT NULL,
                body TEXT NOT NULL, created TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
            );
            CREATE TABLE IF NOT EXISTS memory (
                key TEXT PRIMARY KEY, value TEXT NOT NULL,
                updated TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
            );
            CREATE TABLE IF NOT EXISTS subagents (
                child_id TEXT PRIMARY KEY REFERENCES sessions(id),
                parent_id TEXT NOT NULL REFERENCES sessions(id),
                name TEXT NOT NULL, task TEXT NOT NULL, context TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'running', result TEXT,
                collected INTEGER NOT NULL DEFAULT 0,
                created TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
            );
            CREATE INDEX IF NOT EXISTS subagents_parent ON subagents(parent_id);
            CREATE INDEX IF NOT EXISTS events_session_kind ON events(session_id, kind, id);
            CREATE TABLE IF NOT EXISTS queued_inputs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id),
                request_id INTEGER NOT NULL REFERENCES messages(id),
                content TEXT NOT NULL,
                delivered_message_id INTEGER REFERENCES messages(id)
            );
            CREATE INDEX IF NOT EXISTS queued_inputs_session ON queued_inputs(session_id, request_id, id);
            CREATE TABLE IF NOT EXISTS attempts (
                session_id TEXT NOT NULL REFERENCES sessions(id),
                call_id TEXT NOT NULL,
                request_id INTEGER NOT NULL,
                signature TEXT NOT NULL,
                name TEXT NOT NULL,
                arguments TEXT NOT NULL,
                process_id TEXT,
                status TEXT NOT NULL,
                outcome TEXT NOT NULL,
                event_id INTEGER NOT NULL,
                artifact TEXT,
                PRIMARY KEY(session_id, call_id)
            );
            CREATE INDEX IF NOT EXISTS attempts_action
                ON attempts(session_id, request_id, signature, event_id DESC);
            CREATE INDEX IF NOT EXISTS attempts_process ON attempts(session_id, process_id);
        """)
        # Additive migration preserves existing sessions and memories.
        columns = {r["name"] for r in self.db.execute("PRAGMA table_info(sessions)")}
        if "parent_id" not in columns:
            with self.db:
                self.db.execute("ALTER TABLE sessions ADD COLUMN parent_id TEXT REFERENCES sessions(id)")
        if "request_message_id" not in columns:
            with self.db:
                # NULL means a legacy session whose pointer needs a one-time, streaming lookup.
                # Zero means it is indexed and has no user request yet.
                self.db.execute("ALTER TABLE sessions ADD COLUMN request_message_id INTEGER")

        # Legacy memories remain available, but must earn topical relevance before injection.
        memory_columns = {r["name"] for r in self.db.execute("PRAGMA table_info(memory)")}
        for name, definition in (
            ("scope", "TEXT NOT NULL DEFAULT 'relevant'"),
            ("project", "TEXT"),
            ("source_session", "TEXT"),
        ):
            if name not in memory_columns:
                with self.db:
                    self.db.execute(f"ALTER TABLE memory ADD COLUMN {name} {definition}")

    def close(self):
        self.db.close()

    def create(self, cwd: Path, mode: str, title: str = "New conversation") -> str:
        sid = uuid.uuid4().hex[:12]
        with self.db:
            self.db.execute(
                "INSERT INTO sessions(id,title,cwd,mode,request_message_id) VALUES(?,?,?,?,0)",
                (sid, title[:100], str(cwd), mode),
            )
        return sid

    def session(self, sid: str) -> dict:
        if sid == "latest":
            row = self.db.execute(
                "SELECT * FROM sessions WHERE parent_id IS NULL ORDER BY updated DESC, rowid DESC LIMIT 1"
            ).fetchone()
        else:
            row = self.db.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
        if row is None:
            raise ValueError(f"Session not found: {sid}")
        return dict(row)

    def sessions(self, limit: int = 50, with_messages: bool = False) -> list[dict]:
        # message_count and subagent_count are what tell otherwise identical rows apart: a title is
        # the first 100 characters of the opening prompt, so repeating a prompt collides verbatim.
        # with_messages defaults off so the count is unchanged for callers that expect every
        # top-level session, including the empty shell /clear leaves behind. Filtering here rather
        # than after the fetch is what keeps a caller's limit an exact row count.
        return [
            dict(r)
            for r in self.db.execute(
                f"""SELECT s.*,
                       (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS message_count,
                       (SELECT COUNT(*) FROM subagents g WHERE g.parent_id = s.id) AS subagent_count
                   FROM sessions s WHERE s.parent_id IS NULL
                   {"AND EXISTS(SELECT 1 FROM messages m WHERE m.session_id = s.id)" if with_messages else ""}
                   ORDER BY s.updated DESC, s.rowid DESC LIMIT ?""",
                (limit,),
            )
        ]

    def create_subagent(self, parent_id: str, name: str, task: str, context: str) -> str:
        parent = self.session(parent_id)
        if parent["parent_id"] is not None:
            raise ValueError("Subagents cannot create subagents")
        child_id = uuid.uuid4().hex[:12]
        with self.db:
            self.db.execute(
                "INSERT INTO sessions(id,title,cwd,mode,parent_id,request_message_id) VALUES(?,?,?,?,?,0)",
                (child_id, name, parent["cwd"], parent["mode"], parent_id),
            )
            self.db.execute(
                "INSERT INTO subagents(child_id,parent_id,name,task,context) VALUES(?,?,?,?,?)",
                (child_id, parent_id, name, task, context),
            )
        return child_id

    def subagents(self, parent_id: str) -> list[dict]:
        rows = self.db.execute("SELECT * FROM subagents WHERE parent_id=? ORDER BY rowid", (parent_id,))
        return [{**dict(r), "result": json.loads(r["result"]) if r["result"] else None} for r in rows]

    def finish_subagent(self, child_id: str, status: str, result: dict):
        submitted = self.submitted_result(child_id)
        if submitted:
            # Cleanup may run after a parent failure. A committed report is immutable;
            # only its recoverable artifact pointers may be attached afterwards.
            existing = self.db.execute(
                "SELECT result FROM subagents WHERE child_id=?", (child_id,)
            ).fetchone()
            prior = json.loads(existing["result"]) if existing and existing["result"] else {}
            for key in ("saved_output", "report_output"):
                if prior.get(key):
                    submitted[key] = prior[key]
                if result.get("submitted") and result.get(key):
                    submitted[key] = result[key]
            result, status = submitted, submitted["status"]
        with self.db:
            self.db.execute(
                "UPDATE subagents SET status=?,result=? WHERE child_id=?",
                (status, json.dumps(result), child_id),
            )

    def submitted_result(self, child_id: str) -> dict | None:
        event = self.latest_event(child_id, "child_submission")
        return event["body"] if event else None

    def submit_subagent_result(self, child_id: str, result: dict, verification: dict):
        """Commit the validated outcome, report and delivery state in one transaction.

        The caller validates evidence and permissions before this synchronous commit.
        No model request or filesystem artifact is required to recover the report.
        """
        if self.submitted_result(child_id):
            raise ValueError("This child already submitted its result")
        row = self.db.execute("SELECT name FROM subagents WHERE child_id=?", (child_id,)).fetchone()
        if row is None:
            raise ValueError("Only a child assignment can submit a result")
        serialized = json.dumps(result, ensure_ascii=False)
        session_status = "answered" if result["status"] == "completed" else result["status"]
        with self.db:
            self.db.execute(
                "INSERT INTO events(session_id,kind,body) VALUES(?,?,?)",
                (child_id, "verification_state", json.dumps(verification, ensure_ascii=False)),
            )
            self.db.execute(
                "INSERT INTO events(session_id,kind,body) VALUES(?,?,?)",
                (child_id, "child_submission", serialized),
            )
            self.db.execute(
                "UPDATE subagents SET status=?,result=? WHERE child_id=?",
                (result["status"], serialized, child_id),
            )
            self.db.execute(
                "UPDATE sessions SET status=?,updated=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE id=?",
                (session_status, child_id),
            )

    def collect_subagents(self, parent_id: str, child_ids: list[str]):
        with self.db:
            self.db.executemany(
                "UPDATE subagents SET collected=1 WHERE parent_id=? AND child_id=?",
                [(parent_id, sid) for sid in child_ids],
            )

    def update(self, sid: str, **values):
        allowed = {"title", "cwd", "mode", "status", "summary", "summary_through"}
        if not values or values.keys() - allowed:
            raise ValueError("Invalid session update")
        assignments = ",".join(f"{key}=?" for key in values)
        with self.db:
            self.db.execute(
                f"UPDATE sessions SET {assignments}, updated=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE id=?",
                (*values.values(), sid),
            )

    def append(self, sid: str, message: dict, *, user_request: bool | None = None) -> int:
        request = is_user_request(message) if user_request is None else user_request
        if request and (message.get("role") != "user" or not isinstance(message.get("content"), str)):
            raise ValueError("A user request must be a user message with text content")
        with self.db:
            cursor = self.db.execute(
                "INSERT INTO messages(session_id,body) VALUES(?,?)", (sid, json.dumps(message))
            )
            self.db.execute(
                "UPDATE sessions SET updated=strftime('%Y-%m-%dT%H:%M:%fZ','now')"
                + (",request_message_id=?" if request else "")
                + " WHERE id=?",
                (cursor.lastrowid, sid) if request else (sid,),
            )
            if request:
                # The message and observation boundary must survive or roll back
                # together; verification ledgers may not exist in older sessions.
                self.db.execute(
                    "INSERT INTO events(session_id,kind,body) VALUES(?,?,?)",
                    (sid, "request_started", json.dumps({"message_id": cursor.lastrowid})),
                )
        return cursor.lastrowid

    def latest_request(self, sid: str) -> tuple[int, dict] | None:
        """Current request plus delivered steering, retaining its original evidence/budget boundary."""
        request_id = self.session(sid)["request_message_id"]
        if request_id is None:
            # Legacy histories contain runtime feedback in user-role messages. Migrate
            # lazily, reading one row at a time rather than materializing the transcript.
            request_id = 0
            for row in self.db.execute(
                "SELECT id,body FROM messages WHERE session_id=? ORDER BY id DESC", (sid,)
            ):
                if is_user_request(json.loads(row["body"])):
                    request_id = row["id"]
                    break
            with self.db:
                self.db.execute("UPDATE sessions SET request_message_id=? WHERE id=?", (request_id, sid))
        if not request_id:
            return None
        row = self.db.execute(
            "SELECT id,body FROM messages WHERE session_id=? AND id=?", (sid, request_id)
        ).fetchone()
        if not row:
            return None
        message = json.loads(row["body"])
        updates = self.db.execute(
            "SELECT content FROM queued_inputs WHERE session_id=? AND request_id=? "
            "AND delivered_message_id IS NOT NULL ORDER BY id",
            (sid, request_id),
        ).fetchall()
        if updates:
            message["content"] += "\n\nUser updates during this request (in order):\n" + "\n\n".join(
                item["content"] for item in updates
            )
        return row["id"], message

    def pending_inputs(self, sid: str) -> list[dict]:
        return [
            dict(row)
            for row in self.db.execute(
                "SELECT id,request_id,content FROM queued_inputs "
                "WHERE session_id=? AND delivered_message_id IS NULL ORDER BY id",
                (sid,),
            )
        ]

    def pending_input_count(self, sid: str) -> int:
        return self.db.execute(
            "SELECT COUNT(*) FROM queued_inputs WHERE session_id=? AND delivered_message_id IS NULL",
            (sid,),
        ).fetchone()[0]

    def queue_input(self, sid: str, content: str) -> int:
        request = self.latest_request(sid)
        if request is None:
            raise ValueError("Start a request before queuing a follow-up message")
        with self.db:
            cursor = self.db.execute(
                "INSERT INTO queued_inputs(session_id,request_id,content) VALUES(?,?,?)",
                (sid, request[0], content),
            )
            self.db.execute(
                "UPDATE sessions SET updated=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE id=?",
                (sid,),
            )
        return cursor.lastrowid

    def deliver_queued_inputs(self, sid: str) -> list[int]:
        """Append queued user messages atomically, after a complete assistant/tool exchange."""
        delivered = []
        with self.db:
            for row in self.pending_inputs(sid):
                cursor = self.db.execute(
                    "INSERT INTO messages(session_id,body) VALUES(?,?)",
                    (sid, json.dumps({"role": "user", "content": row["content"]})),
                )
                self.db.execute(
                    "UPDATE queued_inputs SET delivered_message_id=? WHERE id=?",
                    (cursor.lastrowid, row["id"]),
                )
                delivered.append(cursor.lastrowid)
            if delivered:
                self.db.execute(
                    "UPDATE sessions SET updated=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE id=?",
                    (sid,),
                )
        return delivered

    def messages(self, sid: str, after: int = 0) -> list[tuple[int, dict]]:
        return [
            (r["id"], json.loads(r["body"]))
            for r in self.db.execute(
                "SELECT id,body FROM messages WHERE session_id=? AND id>? ORDER BY id", (sid, after)
            )
        ]

    def save_artifact(self, sid: str, text: str) -> Path:
        directory = self.root / "artifacts" / sid
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        data = text.encode("utf-8")
        path = directory / (hashlib.sha256(data).hexdigest() + ".txt")
        if path.exists():
            return path
        # Publish only complete files. A live indexer must never permanently
        # ingest the prefix of a content-addressed artifact still being written.
        temporary = directory / f".{uuid.uuid4().hex}.tmp"
        fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            try:
                os.link(temporary, path)
            except FileExistsError:
                pass  # Another writer has already published these same bytes.
        finally:
            temporary.unlink(missing_ok=True)
        return path

    def event(self, sid: str, kind: str, body: dict):
        with self.db:
            cursor = self.db.execute(
                "INSERT INTO events(session_id,kind,body) VALUES(?,?,?)", (sid, kind, json.dumps(body))
            )
        return cursor.lastrowid

    def events(self, sid: str, *, after: int = 0, kinds: tuple[str, ...] = ()) -> list[dict]:
        query = "SELECT * FROM events WHERE session_id=? AND id>?"
        params = [sid, after]
        if kinds:
            query += " AND kind IN (" + ",".join("?" for _ in kinds) + ")"
            params.extend(kinds)
        return [
            {**dict(r), "body": json.loads(r["body"])}
            for r in self.db.execute(query + " ORDER BY id", params)
        ]

    def latest_event(self, sid: str, kind: str | None = None) -> dict | None:
        query = "SELECT * FROM events WHERE session_id=?"
        params = [sid]
        if kind:
            query += " AND kind=?"
            params.append(kind)
        row = self.db.execute(query + " ORDER BY id DESC LIMIT 1", params).fetchone()
        return {**dict(row), "body": json.loads(row["body"])} if row else None

    def recover(self, sid: str) -> int:
        pending = {}
        for _, message in self.messages(sid):
            for call in message.get("tool_calls", []):
                pending[call["id"]] = call
            if message["role"] == "tool":
                pending.pop(message["tool_call_id"], None)
        for call_id, call in pending.items():
            submitted = self.submitted_result(sid)
            if submitted and call["function"]["name"] == "submit_result":
                # The atomic submission can precede the ordinary tool-result journal
                # by one statement. Recover that known outcome instead of retrying it.
                result = {"ok": True, "submitted": True, "status": submitted["status"]}
                self.append(sid, {"role": "tool", "tool_call_id": call_id, "content": json.dumps(result)})
                continue
            result = {
                "ok": False,
                "error": "interrupted_outcome_unknown",
                "detail": "This call has no durable result. It may have executed partly or fully. "
                "Inspect actual state before considering a retry; do not blindly repeat it.",
            }
            self.append(sid, {"role": "tool", "tool_call_id": call_id, "content": json.dumps(result)})
            self.event(sid, "recovered_unresolved_call", {"call": call})
        return len(pending)

    def memories(self) -> dict[str, str]:
        return {r["key"]: r["value"] for r in self.db.execute("SELECT key,value FROM memory ORDER BY key")}

    def memory_records(self):
        return [dict(row) for row in self.db.execute("SELECT * FROM memory ORDER BY key")]

    def remember(self, key: str, value: str, *, scope=None, project=None, source_session=None):
        prior = self.db.execute("SELECT value,scope,project FROM memory WHERE key=?", (key,)).fetchone()
        if scope is None:
            scope = prior["scope"] if prior else "relevant"
            project = prior["project"] if prior else None
        if scope not in {"relevant", "project", "global"} or (scope == "project" and not project):
            raise ValueError("Memory scope must be relevant, global, or project with an owning project")
        if scope != "project":
            project = None
        if prior and (prior["value"], prior["scope"], prior["project"]) == (value, scope, project):
            # Repeating the same fact is not a new observation. Preserve its date and provenance,
            # and do not invalidate checks that already verified this exact memory state.
            return
        existing = self.memories()
        if key not in existing and len(existing) >= 50:
            raise ValueError("Memory limit reached (50 entries); forget or update an existing entry")
        if sum(len(v) for k, v in existing.items() if k != key) + len(value) > 12000:
            raise ValueError("Memory text limit reached (12000 characters)")
        with self.db:
            self.db.execute(
                "INSERT INTO memory(key,value,scope,project,source_session) VALUES(?,?,?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, scope=excluded.scope, "
                "project=excluded.project, source_session=excluded.source_session, "
                "updated=strftime('%Y-%m-%dT%H:%M:%fZ','now')",
                (key, value, scope, project, source_session),
            )

    def forget(self, key: str):
        with self.db:
            self.db.execute("DELETE FROM memory WHERE key=?", (key,))

    @contextmanager
    def lock(self, sid: str):
        # Resolve through SQLite first; never use arbitrary user input as a lock path.
        sid = self.session(sid)["id"]
        path = self.root / f"{sid}.lock"
        with path.open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError(f"Session {sid} is already open in another process") from None
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
