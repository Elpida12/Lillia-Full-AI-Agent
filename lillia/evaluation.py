"""Measure delegation over an entire exported run, without model grading.

Usage: python -m lillia.evaluation export.json [another-export.json ...]
Metrics describe observed behavior; they do not establish task correctness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import posixpath
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from .progress import READ_TOOLS

COORDINATION_TOOLS = frozenset(
    {
        "spawn_subagent",
        "wait_subagent",
        "cancel_subagent",
        "list_subagents",
        "update_work_plan",
        "work_plan_status",
    }
)


def recovery_metrics(export):
    """Describe execution and retained evidence without grading a task or a recovery."""
    journals = [(export["session"]["id"], export.get("events", []))]
    children = export.get("subagents", [])
    journals += [
        (child.get("child_id", f"child-{index}"), child.get("events", []))
        for index, child in enumerate(children)
    ]
    processes, findings = {}, {}
    launch_attempts = launch_errors = duplicate_terminals = finding_updates = 0
    reviews, decisions = Counter(), Counter()
    completion_tokens = completion_known = completion_missing = 0
    launch_tools = {"execute_program", "shell"}
    process_tools = launch_tools | {"poll_process", "write_process", "stop_process"}
    terminal_states = {"succeeded", "failed", "timed_out", "stopped"}
    for owner, events in journals:
        starts = {}
        for event in sorted(events, key=lambda e: e["id"]):
            kind, body = event["kind"], event["body"]
            name, call_id = body.get("name"), body.get("call_id")
            if kind == "tool_started":
                starts[call_id] = body.get("arguments", {})
                launch_attempts += name in launch_tools
            elif kind == "tool_error" and name in launch_tools:
                launch_errors += 1
            elif kind == "tool_finished" and name in process_tools:
                result = body.get("result", {})
                pid = result.get("process_id") or starts.get(call_id, {}).get("process_id")
                # Only process observations establish execution. Permission, schema and
                # missing-handle errors must not masquerade as failed experiments.
                if result.get("timed_out"):
                    status = "timed_out"
                elif result.get("stopped"):
                    status = "stopped"
                elif result.get("running"):
                    status = "running"
                elif type(result.get("exit_code")) is int:
                    status = "succeeded" if result["exit_code"] == 0 and result.get("ok") else "failed"
                elif result.get("status") in terminal_states | {"running"}:
                    status = result["status"]
                else:
                    continue
                key = (owner, "process", pid) if pid else (owner, "call", call_id or event["id"])
                previous = processes.get(key)
                if previous and previous["status"] in terminal_states:
                    duplicate_terminals += status in terminal_states
                    continue
                mode = result.get("execution_mode") or (previous or {}).get("mode")
                mode = mode or {"shell": "shell", "execute_program": "argv"}.get(name, "unknown")
                processes[key] = {"status": status, "mode": mode}
            elif kind in {"recovery_review_attempted", "recovery_review", "recovery_review_rejected"}:
                reviews[kind] += 1
                if kind == "recovery_review":
                    decisions[body.get("decision", {}).get("status", "unknown")] += 1
            elif kind == "finding_record":
                finding_updates += 1
                # Findings are request scoped: a reused name in another request is distinct.
                if body.get("finding_id"):
                    findings[(owner, body.get("request_id"), body["finding_id"])] = body.get(
                        "status", "unknown"
                    )
            elif kind == "model_usage":
                value = (body.get("usage") or {}).get("completion_tokens")
                if type(value) is int and value >= 0:
                    # Provider reasoning_tokens is a subset, never add it again.
                    completion_tokens += value
                    completion_known += 1
                else:
                    completion_missing += 1

    submitted = retained = interrupted_without = interrupted_retained = 0
    interrupted_statuses = {"cancelled", "interrupted", "failed", "error"}
    for child in children:
        result = child.get("result") or {}
        submissions = [e["body"] for e in child.get("events", []) if e["kind"] == "child_submission"]
        if result.get("submitted"):
            submissions.append(result)
        has_submission = bool(submissions)
        has_report = any(isinstance(s.get("report"), str) and s["report"].strip() for s in submissions)
        submitted += has_submission
        retained += has_report
        interrupted = child.get("status") in interrupted_statuses or any(
            e["kind"] in {"turn_interrupted", "turn_error"} for e in child.get("events", [])
        )
        interrupted_without += interrupted and not has_submission
        interrupted_retained += interrupted and has_report

    by_mode = {}
    for process in processes.values():
        by_mode.setdefault(process["mode"], Counter())[process["status"]] += 1
    return {
        "execution_launch_attempts": launch_attempts,
        "execution_launch_errors": launch_errors,
        "execution_processes_observed": len(processes),
        "execution_outcomes_reported": dict(Counter(p["status"] for p in processes.values())),
        "execution_outcomes_by_mode": {mode: dict(counts) for mode, counts in by_mode.items()},
        "execution_duplicate_terminal_observations": duplicate_terminals,
        "recovery_review_attempts": reviews["recovery_review_attempted"],
        "recovery_reviews_completed": reviews["recovery_review"],
        "recovery_review_rejections": reviews["recovery_review_rejected"],
        "recovery_review_decisions": dict(decisions),
        "child_reports_submitted": submitted,
        "child_submitted_reports_retained": retained,
        "child_submitted_reports_missing": submitted - retained,
        "children_interrupted_without_submission": interrupted_without,
        "children_interrupted_with_retained_submission": interrupted_retained,
        "child_submitted_reports_on_interrupted_parent": (
            retained if export["session"].get("status") in interrupted_statuses else 0
        ),
        "finding_updates": finding_updates,
        "finding_statuses_reported_by_model": dict(Counter(findings.values())),
        "reported_completion_tokens": completion_tokens if completion_known else None,
        "completion_usage_records_reported": completion_known,
        "completion_usage_records_missing": completion_missing,
    }


def read_context_metrics(export):
    """Measure journaled read outputs and exact shared observations, without judging their purpose."""
    parent = export["session"]["id"]
    cwd = export["session"].get("cwd", "")
    journals = [(parent, export["events"])]
    journals += [
        (child.get("child_id", f"child-{index}"), child.get("events", []))
        for index, child in enumerate(export.get("subagents", []))
    ]
    chars, calls, file_chars = Counter(), Counter(), Counter()
    observations = []
    for owner, events in journals:
        starts = {
            e["body"].get("call_id"): e["body"].get("arguments", {})
            for e in events
            if e["kind"] == "tool_started"
        }
        for event in events:
            body = event["body"]
            result = body.get("result", {})
            if event["kind"] != "tool_finished" or body.get("name") not in READ_TOOLS or not result.get("ok"):
                continue
            # Agent.run serializes the entire result with these options, including metadata.
            chars[owner] += len(json.dumps(result, ensure_ascii=False))
            calls[owner] += 1
            if body["name"] != "read_file" or not isinstance(result.get("content"), str):
                continue
            content = result["content"]
            file_chars[owner] += len(content)
            args = starts.get(body.get("call_id"), {})
            path = result.get("path") or args.get("path")
            if not path or not content:
                continue
            # Prefer the tool's absolute path; never consult the possibly changed local filesystem.
            path = posixpath.normpath(path if posixpath.isabs(path) else posixpath.join(cwd, path))
            lines = re.findall(r"(?m)^(\d+): ", content)
            observed_range = (
                (int(lines[0]), int(lines[-1]))
                if lines
                else (args.get("start_line", 1), args.get("max_lines", 200))
            )
            digest = hashlib.sha256(content.encode()).hexdigest()
            key = (path, observed_range, digest)
            observations.append((event["created"], event["id"], owner, key, len(content)))
    seen = {}
    duplicated_reads = duplicated_chars = parent_repeats = 0
    for _, _, owner, key, count in sorted(observations):
        readers = seen.setdefault(key, set())
        if readers - {owner}:
            duplicated_reads += 1
            duplicated_chars += count
            parent_repeats += owner == parent
        readers.add(owner)
    return {
        "parent_read_output_chars": chars[parent],
        "child_read_output_chars": sum(value for owner, value in chars.items() if owner != parent),
        "read_output_chars_by_agent": dict(chars),
        "parent_read_calls": calls[parent],
        "child_read_calls": sum(value for owner, value in calls.items() if owner != parent),
        "parent_file_read_content_chars": file_chars[parent],
        "child_file_read_content_chars": sum(value for owner, value in file_chars.items() if owner != parent),
        "cross_agent_identical_file_range_reads": duplicated_reads,
        "cross_agent_identical_file_range_chars": duplicated_chars,
        "cross_agent_identical_file_ranges": sum(len(readers) > 1 for readers in seen.values()),
        "parent_repeated_child_file_range_reads": parent_repeats,
    }


def delegation_metrics(export):
    events = export["events"]
    children = export.get("subagents", [])
    finishes = [
        e for e in events if e["kind"] == "subagent_finished" and e["body"].get("started") is not False
    ]
    first_finish = min((e["created"] for e in finishes), default=None)
    last_finish = max((e["id"] for e in finishes), default=None)
    starts = {e["body"]["call_id"]: e["body"] for e in events if e["kind"] == "tool_started"}
    seen, repeated_reads = set(), 0
    for e in events:
        body = e["body"]
        if e["kind"] != "tool_finished" or body.get("name") not in READ_TOOLS:
            continue
        result = body["result"]
        if not result.get("ok"):
            continue
        observation = {
            k: v
            for k, v in result.items()
            if k not in {"evidence_id", "saved_output", "evidence_ref", "evidence_passages", "passage_count"}
        }
        args = starts.get(body["call_id"], {}).get("arguments", {})
        digest = hashlib.sha256(
            json.dumps([body["name"], args, observation], sort_keys=True).encode()
        ).hexdigest()
        repeated_reads += digest in seen
        seen.add(digest)
    plans = [e["body"] for e in events if e["kind"] == "work_plan_state"]
    current_plan = plans[-1] if plans else {}
    archived_items = {
        key
        for e in events
        if e["kind"] == "work_items_archived"
        and current_plan.get("archive_floor", 0) < e["id"] <= current_plan.get("archive_through", 0)
        for key in e["body"]["items"]
    }
    usage = [e["body"] for e in events if e["kind"] == "model_usage"]
    usage += [e["body"] for c in children for e in c.get("events", []) if e["kind"] == "model_usage"]
    created = export["session"]["created"]
    end = max((e["created"] for e in events), default=created)
    spawn_attempts = [
        e for e in events if e["kind"] == "tool_started" and e["body"].get("name") == "spawn_subagent"
    ]
    queue_events = [e for e in events if e["kind"] == "subagent_queued"]
    dispatches = [e for e in events if e["kind"] == "subagent_dispatched"]
    # Older journals have no start events. Their rows were created only on actual spawning.
    queued_ids = {e["body"].get("agent_id") for e in queue_events}
    starts_at = [e["created"] for e in dispatches]
    dispatched_ids = {e["body"].get("agent_id") for e in dispatches}
    starts_at += [
        c["created"]
        for c in children
        if c.get("child_id") not in queued_ids | dispatched_ids
        and c["status"] != "queued"
        and (c.get("result") or {}).get("started") is not False
    ]
    first_start = min(starts_at, default=None)
    coordination_errors = Counter()
    for event in events:
        body = event["body"]
        if event["kind"] == "tool_error" and body.get("name") in COORDINATION_TOOLS:
            coordination_errors[body["name"]] += 1
    assessment_events = [e for e in events if e["kind"].startswith("delegation_assessment")]
    assessments = [e for e in assessment_events if e["kind"] == "delegation_assessment"]
    preflights = [e["body"] for e in events if e["kind"] == "delegation_preflight"]
    return {
        "session_id": export["session"]["id"],
        "status": export["session"]["status"],
        "children_spawned": len(starts_at),
        "spawn_tool_attempts": len(spawn_attempts),
        "assignments_accepted": len(children),
        "assignments_queued": len(queue_events),
        "children_started": len(starts_at),
        "time_to_first_spawn_seconds": (
            datetime.fromisoformat(first_start) - datetime.fromisoformat(created)
        ).total_seconds()
        if first_start
        else None,
        "children_started_after_first_report": sum(
            bool(first_finish and started > first_finish) for started in starts_at
        ),
        "children_spawned_after_first_report": sum(
            bool(first_finish and started > first_finish) for started in starts_at
        ),
        "child_outcomes": dict(Counter(c["status"] for c in children)),
        "parent_tools_after_last_report": sum(
            e["kind"] == "tool_started" and e["id"] > last_finish for e in events
        )
        if last_finish is not None
        else None,
        "repeated_identical_reads": repeated_reads,
        **read_context_metrics(export),
        **recovery_metrics(export),
        "context_compactions": sum(e["kind"] == "context_compacted" for e in events),
        "checkpoints": sum(e["kind"] == "orchestration_checkpoint" for e in events),
        "decisions": sum(e["kind"] == "orchestration_decision" for e in events),
        "delegation_recoveries": dict(
            Counter(e["body"]["action"] for e in events if e["kind"] == "delegation_recovery")
        ),
        "delegation_replan_rejections": sum(
            e["kind"] == "tool_error"
            and e["body"].get("result", {}).get("error") == "DelegationReplanRequired"
            for e in events
        ),
        "assessment_events": dict(Counter(e["kind"] for e in assessment_events)),
        "assessment_decisions": dict(Counter(str(e["body"].get("decision", "unknown")) for e in assessments)),
        "assessment_triggers": dict(Counter(str(e["body"].get("trigger", "unknown")) for e in assessments)),
        "assessment_assignment_attempts": sum(len(e["body"].get("spawn_outcomes", [])) for e in assessments),
        "assessment_assignment_rejections": sum(len(e["body"].get("rejections", [])) for e in assessments),
        "delegation_preflight_count": len(preflights),
        "delegation_preflight_deferred_read_calls": sum(
            len(p.get("deferred_call_ids", [])) for p in preflights
        ),
        "delegation_preflight_estimated_read_chars": sum(
            p.get("estimated_read_chars", 0) for p in preflights
        ),
        "delegation_preflight_peak_input_fraction": max(
            (p.get("input_chars", 0) / p["input_limit"] for p in preflights if p.get("input_limit", 0) > 0),
            default=None,
        ),
        "model_request_purposes": dict(
            Counter(e["body"].get("purpose", "unknown") for e in events if e["kind"] == "model_request")
        ),
        "coordination_errors": dict(coordination_errors),
        "parent_outcome_reported_by_model": next(
            (e["body"].get("outcome") for e in reversed(events) if e["kind"] == "verification_state"), None
        ),
        "children_with_verified_outcomes": sum(
            bool(c.get("result", {}).get("outcome_verified")) for c in children if c.get("result")
        ),
        "work_item_statuses_reported_by_model": dict(
            Counter(i["status"] for i in plans[-1]["items"].values())
        )
        if plans
        else {},
        "archived_work_items_reported_by_model": len(archived_items),
        "accounted_tokens": sum(u["tokens"] for u in usage),
        "token_sources": dict(Counter(u["source"] for u in usage)),
        "session_span_seconds": (
            datetime.fromisoformat(end) - datetime.fromisoformat(created)
        ).total_seconds(),
        "note": "Session aggregates may include several user turns and idle time. More spawns are not proof of quality. "
        "Inspect outcome evidence; repeated reads count identical tool/arguments/results. Read output chars "
        "count successful built-in read results serialized to the model, excluding shell output. Cross-agent "
        "duplicates require the same file path, observed range, and content hash; targeted verification may "
        "legitimately duplicate reads. Preflight estimates are not measured context savings. "
        "Execution outcomes are last observed process states, not task correctness; shell success only "
        "describes the shell exit. Recovery decisions and finding statuses are model interpretations. "
        "Retained reports remain recoverable in the export and do not prove a recovery succeeded. "
        "Reported completion tokens exclude missing usage records and already include reasoning tokens.",
    }


def aggregate_evaluations(reports, min_pass_rate=None):
    """Summarize matched evaluation runs; a configured gate applies independently to every strategy."""

    def aggregate(rows, **labels):
        count = len(rows)
        completed = sum(bool(row.get("completed_with_correct_outcome")) for row in rows)
        tokens = sum(row["accounted_tokens"] for row in rows)
        elapsed = sum(row["elapsed_seconds"] for row in rows)
        sources = Counter()
        for row in rows:
            sources.update(row.get("token_sources", {}))
        return {
            **labels,
            "runs": count,
            "completed_correct": completed,
            "completion_rate": completed / count,
            "accounted_tokens_total": tokens,
            "accounted_tokens_mean": tokens / count,
            "token_sources": dict(sources),
            "elapsed_seconds_total": round(elapsed, 2),
            "elapsed_seconds_mean": round(elapsed / count, 2),
        }

    strategies = sorted({row["strategy"] for row in reports})
    by_strategy = [
        aggregate([r for r in reports if r["strategy"] == strategy], strategy=strategy)
        for strategy in strategies
    ]
    groups = sorted({(r["strategy"], r["case"], r["prompt_mode"]) for r in reports})
    by_case = [
        aggregate(
            [r for r in reports if (r["strategy"], r["case"], r["prompt_mode"]) == group],
            strategy=group[0],
            case=group[1],
            prompt_mode=group[2],
        )
        for group in groups
    ]
    failed = [
        r["strategy"]
        for r in by_strategy
        if min_pass_rate is not None and r["completion_rate"] < min_pass_rate
    ]
    return {
        "total_runs": len(reports),
        "by_strategy": by_strategy,
        "by_case": by_case,
        "gate": {
            "enabled": min_pass_rate is not None,
            "min_pass_rate": min_pass_rate,
            "passed": bool(by_strategy) and not failed if min_pass_rate is not None else None,
            "failed_strategies": failed,
        },
        "note": "Completion requires a correct checked outcome and an answered session. Costs include failed "
        "attempts; token_sources identifies estimated or unmetered usage. Small samples are descriptive, "
        "not proof of a strategy advantage. Skipped cohorts are excluded from every strategy equally.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exports", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps([delegation_metrics(json.loads(p.read_text())) for p in args.exports], indent=2))


if __name__ == "__main__":
    main()
