"""Read-only, bounded recall of exact actions retired by context compaction.

This is historical execution evidence, never proof that the current task is done
or that a file, command input, or remote resource is still unchanged.
"""

from __future__ import annotations

import json

from .store import RUNTIME_MESSAGES, Store

SUPPORTED_TOOLS = frozenset(
    {
        "execute_program",
        "shell",
        "read_file",
        "write_file",
        "edit_file",
        "list_directory",
        "search_files",
        "web_search",
        "fetch_url",
    }
)
_PROCESS_TOOLS = frozenset({"execute_program", "shell"})
_MAX_ARGUMENT_CHARS = 131072


def _arguments(value) -> tuple[dict, str] | None:
    """Canonicalize JSON object order only; preserve every argument's meaning."""
    if isinstance(value, str):
        if len(value) > _MAX_ARGUMENT_CHARS:
            return None
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return None
    if not isinstance(value, dict):
        return None
    try:
        canonical = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        return None
    if len(canonical) > _MAX_ARGUMENT_CHARS:
        return None
    return value, canonical


def _request_floor(store: Store, sid: str, session: dict, scan_limit: int) -> int:
    pointer = session.get("request_message_id")
    if pointer is not None:
        return int(pointer)
    # Store.latest_request migrates legacy histories. Recall must never modify the
    # journal, and must not deserialize old private reasoning just to find a user.
    oldest = store.db.execute(
        "SELECT id FROM messages WHERE session_id=? ORDER BY id DESC LIMIT 1 OFFSET ?",
        (sid, scan_limit - 1),
    ).fetchone()
    rows = store.db.execute(
        """SELECT id, substr(json_extract(body, '$.content'), 1, 256) AS content
           FROM messages WHERE session_id=? AND id>=? AND json_extract(body, '$.role')='user'
               AND json_type(body, '$.content')='text'
           ORDER BY id DESC LIMIT ?""",
        (sid, oldest["id"] if oldest else 0, scan_limit),
    )
    for row in rows:
        if not row["content"].startswith(RUNTIME_MESSAGES):
            return row["id"]
    return 0


def _window_end(store: Store, sid: str, after: int, through: int, scan_limit: int) -> int:
    # The (session_id, id) index bounds a window without reading message bodies.
    # A LIMIT on matching JSON rows alone would still parse an unbounded history.
    row = store.db.execute(
        """SELECT id FROM messages WHERE session_id=? AND id>? AND id<=?
           ORDER BY id ASC LIMIT 1 OFFSET ?""",
        (sid, after, through, scan_limit - 1),
    ).fetchone()
    return row["id"] if row else through


def _saved_check_launches(store: Store, sid: str, request_id: int, call_ids: set[str], scan_limit: int):
    """Resolve historical aliases only from their bounded, recorded launches.

    A declaration may have changed in a later request. The original invocation
    is durable execution provenance; today's declaration is never a substitute.
    Modern saved-check calls always have a request marker. Missing or ambiguous
    provenance conservatively yields no historical match.
    """
    boundary = store.db.execute(
        """SELECT id, json_extract(body, '$.message_id') AS request_id FROM events
           WHERE session_id=? AND kind='request_started' ORDER BY id DESC LIMIT 1""",
        (sid,),
    ).fetchone()
    if boundary is None or boundary["request_id"] != request_id:
        return {}
    # Bound by actual launch rows before filtering JSON, so a missing match
    # cannot scan an arbitrarily long event history or load unrelated bodies.
    rows = store.db.execute(
        """WITH launches AS MATERIALIZED (
               SELECT body FROM events
               WHERE session_id=? AND kind='tool_started' AND id>?
               ORDER BY id ASC LIMIT ?
           )
           SELECT json_extract(body, '$.call_id') AS call_id,
               json_extract(body, '$.name') AS name,
               substr(json_extract(body, '$.arguments'), 1, ?) AS arguments
           FROM launches
           WHERE json_extract(body, '$.call_id') IN (SELECT value FROM json_each(?))""",
        (sid, boundary["id"], scan_limit, _MAX_ARGUMENT_CHARS + 1, json.dumps(sorted(call_ids))),
    )
    found, ambiguous = {}, set()
    for row in rows:
        call_id = row["call_id"]
        if call_id in found or call_id in ambiguous:
            found.pop(call_id, None)
            ambiguous.add(call_id)
        else:
            found[call_id] = row
    return {key: row for key, row in found.items() if row["name"] in _PROCESS_TOOLS}


def _results(
    store: Store,
    sid: str,
    after: int,
    through: int,
    *,
    call_id: str | None = None,
    process_id: str | None = None,
    max_chars: int = 1800,
    scan_limit: int = 2000,
):
    # Extract status independently of the excerpt: output can be very large and
    # an error/exit status may occur after the excerpt's character limit.
    selector = (
        "json_extract(body, '$.tool_call_id')=?"
        if call_id is not None
        else ("json_extract(payload, '$.process_id')=?")
    )
    through = _window_end(store, sid, after, through, scan_limit)
    # Direct results follow their call; do not scan every later reasoning blob
    # backwards to reach that adjacent row. Process observations use the latest
    # status in the bounded window to avoid reviving an earlier running status.
    order = "ASC" if call_id is not None else "DESC"
    return store.db.execute(
        f"""WITH observations AS (
                SELECT id, body,
                    CASE WHEN json_valid(json_extract(body, '$.content'))
                        THEN json_extract(body, '$.content') ELSE '{{}}' END AS payload
                FROM messages
                WHERE session_id=? AND id>? AND id<=?
                    AND json_extract(body, '$.role')='tool'
            )
            SELECT id, json_extract(body, '$.tool_call_id') AS call_id,
                json_type(payload, '$.ok') AS ok_type,
                json_extract(payload, '$.running') AS running,
                json_type(payload, '$.running') AS running_type,
                json_extract(payload, '$.exit_code') AS exit_code,
                json_type(payload, '$.exit_code') AS exit_type,
                json_extract(payload, '$.status') AS status,
                COALESCE(json_extract(payload, '$.error') NOT IN ('', 0), 0) AS error,
                json_extract(payload, '$.timed_out') AS timed_out,
                json_extract(payload, '$.stopped') AS stopped,
                json_extract(payload, '$.process_id') AS process_id,
                substr(payload, 1, ?) AS excerpt,
                length(payload)>? AS truncated
            FROM observations WHERE {selector} ORDER BY id {order} LIMIT 1""",
        (sid, after, through, max_chars, max_chars, call_id if call_id is not None else process_id),
    ).fetchone()


def _succeeded(row, *, process: bool) -> bool:
    if row is None or row["ok_type"] != "true":
        return False
    if row["running"] or row["error"] or row["timed_out"] or row["stopped"]:
        return False
    if row["status"] not in {None, "succeeded", "completed", "success"}:
        return False
    if row["exit_type"] not in {None, "null"} and (row["exit_type"] != "integer" or row["exit_code"] != 0):
        return False
    # Launch acknowledgement and an unknown exit must not become completed work.
    return not process or (
        row["running_type"] == "false" and row["exit_type"] == "integer" and row["exit_code"] == 0
    )


def find_prior_actions(
    store: Store,
    sid: str,
    calls: list[dict],
    *,
    max_results: int = 3,
    max_chars: int = 1800,
    scan_limit: int = 2000,
) -> list[dict]:
    """Find exact, successfully finished actions from this compacted request.

    ``scan_limit`` bounds candidate tool calls and each journal scan by actual
    message rows (or launch events for saved-check aliases), not bytes of reasoning. SQLite
    projects call metadata so a huge later reasoning message cannot hide an early
    action behind a transcript-character cutoff. Each outcome is at most
    ``max_chars`` characters. Results identify the original action and the durable
    terminal observation, which may be a later process poll. No live state is read.

    Failed, interrupted, unobserved, and still-running actions are omitted. The
    helper intentionally makes no equivalence guesses about shell commands,
    paths, omitted defaults, or changed arguments.
    """
    if max_results <= 0 or max_chars <= 0 or scan_limit <= 0:
        return []
    proposed = {}
    for call in calls:
        function = call.get("function", {})
        name = function.get("name")
        args = _arguments(function.get("arguments"))
        if name in SUPPORTED_TOOLS and args is not None:
            proposed.setdefault((name, args[1]), (call.get("id"), args[0]))
    if not proposed:
        return []
    session = store.session(sid)
    through = session["summary_through"]
    if not through:
        return []
    floor = _request_floor(store, sid, session, scan_limit)
    if not floor or floor >= through:
        return []
    candidate_through = _window_end(store, sid, floor, through, scan_limit)
    names = {key[0] for key in proposed}
    if names & _PROCESS_TOOLS:
        names.add("run_required_check")
    names = sorted(names)
    placeholders = ",".join("?" for _ in names)
    rows = store.db.execute(
        f"""SELECT m.id AS assistant_message_id,
                json_extract(c.value, '$.id') AS call_id,
                json_extract(c.value, '$.function.name') AS name,
                substr(json_extract(c.value, '$.function.arguments'), 1, ?) AS arguments
            FROM messages AS m, json_each(
                CASE WHEN json_type(m.body, '$.tool_calls')='array'
                    THEN json_extract(m.body, '$.tool_calls') ELSE '[]' END
            ) AS c
            WHERE m.session_id=? AND m.id>? AND m.id<=?
                AND json_extract(m.body, '$.role')='assistant'
                AND json_extract(c.value, '$.function.name') IN ({placeholders})
            ORDER BY m.id ASC, c.key ASC LIMIT ?""",
        (_MAX_ARGUMENT_CHARS + 1, sid, floor, candidate_through, *names, scan_limit),
    ).fetchall()
    helper_ids = {
        row["call_id"]
        for row in rows
        if row["name"] == "run_required_check" and isinstance(row["call_id"], str) and row["call_id"]
    }
    helper_launches = _saved_check_launches(store, sid, floor, helper_ids, scan_limit) if helper_ids else {}
    found = []
    for row in rows:
        source = helper_launches.get(row["call_id"]) if row["name"] == "run_required_check" else row
        if source is None:
            continue
        name = source["name"]
        args = _arguments(source["arguments"])
        if args is None or not row["call_id"]:
            continue
        key = (name, args[1])
        if key not in proposed:
            continue
        outcome = _results(
            store,
            sid,
            row["assistant_message_id"],
            through,
            call_id=row["call_id"],
            max_chars=max_chars,
            scan_limit=scan_limit,
        )
        process = name in _PROCESS_TOOLS
        if (
            process
            and outcome is not None
            and outcome["ok_type"] == "true"
            and outcome["running_type"] == "true"
            and outcome["process_id"]
            and outcome["status"] in {None, "running"}
            and not (outcome["error"] or outcome["timed_out"] or outcome["stopped"])
        ):
            outcome = _results(
                store,
                sid,
                outcome["id"],
                through,
                process_id=outcome["process_id"],
                max_chars=max_chars,
                scan_limit=scan_limit,
            )
        if not _succeeded(outcome, process=process):
            continue
        proposed_call_id, arguments = proposed.pop(key)
        found.append(
            {
                "proposed_call_id": proposed_call_id,
                "call_id": row["call_id"],
                "name": name,
                "arguments": arguments,
                "assistant_message_id": row["assistant_message_id"],
                "result_message_id": outcome["id"],
                "result_call_id": outcome["call_id"],
                "observed_status": "succeeded",
                "outcome": outcome["excerpt"],
                "outcome_truncated": bool(outcome["truncated"]),
            }
        )
        if len(found) >= max_results or not proposed:
            break
    return found
