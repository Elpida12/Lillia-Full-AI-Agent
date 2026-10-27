"""Immutable, domain-independent execution requirements for task criteria.

Command identity and execution success are enforced here. The agent still has to
choose adequate assertions and justify what their results establish.
"""

from __future__ import annotations

import hashlib
import json


class VerificationBlocked(ValueError):
    """A rejected operation with durable, controller-readable recovery context."""

    def __init__(self, message, blocker):
        super().__init__(message)
        self.blocker = blocker


class RequiredChecks:
    def __init__(self, verification):
        self.verification = verification

    @staticmethod
    def schema():
        from .tools import string

        # Grammar-constrained backends may compile union branches independently,
        # ignoring sibling properties/required. Each branch must therefore be a
        # complete contract, or generation can be restricted to {"tool": "shell"}
        # even while our JSON Schema validator requires command and expectation.
        branches = []
        for tool, key, command_schema in (
            ("execute_program", "argv", {
                "type": "array", "minItems": 1, "maxItems": 256,
                "items": string("Program or literal argument", 64000),
            }),
            ("shell", "command", string("Exact shell command; prefer argv for direct execution", 64000)),
        ):
            branches.append({
                "type": "object",
                "properties": {
                    "tool": {"type": "string", "const": tool},
                    key: command_schema,
                    "cwd": string("Working directory; defaults to this session's directory", 4096),
                    "expectation": string("Which assertions this execution must check, and its limits", 1500),
                },
                "required": ["tool", key, "expectation"],
                "additionalProperties": False,
            })
        return {
            "description": "Immutable command/suite; declare before execution. Supply tool, command (shell) "
            "or argv (execute_program), and expectation; cwd is optional. expectation describes assertions, "
            "not an enforced test. Normal permissions apply.",
            "oneOf": branches,
        }

    def prepare(self, criteria, *, replacing=()):
        v = self.verification
        available = {schema["function"]["name"] for schema in v.registry.schemas()}
        latest = v.store.latest_event(v.sid)
        additions = {}
        for criterion in criteria:
            if "required_check" not in criterion:
                continue
            cid = criterion["id"]
            definition = dict(criterion["required_check"])
            if not definition["expectation"].strip():
                raise ValueError("A required check must describe its expected assertions")
            if definition["tool"] == "execute_program":
                argv = definition["argv"]
                if not argv[0] or any("\0" in arg for arg in argv) or sum(
                    len(arg.encode("utf-8")) for arg in argv
                ) > 64000:
                    raise ValueError("Required argv needs a program and NUL-free arguments within 64000 bytes")
            elif not definition["command"].strip() or "\0" in definition["command"]:
                raise ValueError("Required shell command must be nonempty and NUL-free")
            definition["cwd"] = str(v.registry.resolve(definition.get("cwd") or str(v.registry.cwd)))
            existing = v.state["required_checks"].get(cid)
            if existing and cid not in replacing:
                if definition != {k: value for k, value in existing.items() if k != "registered_event_id"}:
                    row = self.status()[cid]
                    amendable = v.state.get("authority") and v.criterion_authority(cid)["origin"] in {
                        "working", "runtime",
                    }
                    raise VerificationBlocked(
                        "Use revise_plan with observed evidence to amend this working check"
                        if amendable
                        else "Required checks cannot be rewritten; add a criterion or report a limitation",
                        {
                            **v.required_check_blocker(cid, row),
                            "cause": "immutable_check_rewrite",
                            "reason": (
                                "plan_task cannot rewrite a saved working check. Use revise_plan with fresh "
                                "anchored evidence; a new criterion does not retire this check."
                                if amendable
                                else "The saved required command is immutable. A new criterion does not "
                                "retire it; report partial/blocked if that declaration cannot succeed."
                            ),
                        },
                    )
                continue
            if definition["tool"] not in available:
                raise ValueError(
                    f"Required check for {cid} needs unavailable tool {definition['tool']}; "
                    "no criteria were added. Report the capability limitation."
                )
            definition["registered_event_id"] = latest["id"] if latest else 0
            additions[cid] = definition
        return additions

    @staticmethod
    def fingerprint(definition):
        argv = definition.get("argv")
        if definition["tool"] == "shell":
            argv = ["/bin/bash", "-o", "pipefail", "-c", definition["command"]]
        return hashlib.sha256(
            json.dumps({"argv": argv, "cwd": definition["cwd"]}, sort_keys=True).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def outcome_fingerprint(result):
        """Execution content, not a new event/process ID or elapsed time, is progress.

        Do not infer semantic equivalence of different output. Runtime bookkeeping,
        passage labels and process provenance are deliberately outside this digest.
        """
        fields = ("ok", "running", "exit_code", "timed_out", "stopped", "error", "output")
        payload = {key: result.get(key) for key in fields}
        payload["output_sha256"] = result.get("output_sha256")
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def execution_outcomes(self, call, result, *, definitions=None):
        """Match a terminal execution to its exact, already registered check."""
        if result.get("running") or not result.get("command_fingerprint"):
            return []
        v = self.verification
        event = v.evidence().get(result.get("evidence_id"))
        if event is None:
            return []
        function = call.get("function", {})
        name = function.get("name")
        succeeded = bool(
            result.get("ok")
            and result.get("exit_code") == 0
            and not result.get("timed_out")
            and not result.get("stopped")
        )
        if name == "poll_process":
            return [
                {
                    **v.required_check_blocker(cid, row),
                    "execution_succeeded": succeeded,
                    "cause": "required_execution_failed",
                    "reason": "The exact required execution failed; inspect its terminal output.",
                }
                for cid, row in self.status(through=event["id"], definitions=definitions).items()
                if row.get("evidence_id") == result.get("evidence_id")
            ]
        if name not in {"shell", "execute_program"}:
            return []  # Polling one failed execution must not count as a new attempt.
        args = json.loads(function["arguments"])
        blockers = []
        definitions = v.state["required_checks"] if definitions is None else definitions
        for cid, definition in definitions.items():
            key = "argv" if name == "execute_program" else "command"
            fingerprint = self.fingerprint(definition)
            if (
                definition["tool"] != name
                or args.get(key) != definition.get(key)
                or result["command_fingerprint"] != fingerprint
                or event["id"] <= definition["registered_event_id"]
            ):
                continue
            blockers.append(
                {
                    **v.required_check_blocker(
                        cid,
                        {
                            "check_fingerprint": fingerprint,
                            "progress_event_id": event["id"],
                            "outcome_fingerprint": self.outcome_fingerprint(result),
                            "evidence_id": result.get("evidence_id"),
                        },
                    ),
                    "execution_succeeded": succeeded,
                    "cause": "required_execution_failed",
                    "reason": "The exact required execution failed; inspect its output. Repeating the "
                    "same failure does not resolve the check. Preserve findings and report partial/blocked "
                    "if the declaration cannot succeed.",
                }
            )
        return blockers

    def invocation(self, criterion_id, timeout_seconds, wait_ms=1000, verification_effect="mutation"):
        """Resolve an explicit request to run the saved command, without executing it.

        Only process lifetime, waiting and the caller-declared effect can vary. In
        particular, no alternative command or argument list can replace a check.
        ToolRegistry still validates and authorizes the concrete invocation.
        """
        definition = self.verification.state["required_checks"].get(criterion_id)
        if definition is None:
            raise ValueError(
                "No declared required_check for this criterion. Use task_status and plan_task "
                "to declare the relevant check before running it."
            )
        name = definition["tool"]
        key = "argv" if name == "execute_program" else "command"
        command = definition[key]
        return name, {
            key: list(command) if key == "argv" else command,
            "cwd": definition["cwd"],
            "timeout_seconds": timeout_seconds,
            "wait_ms": wait_ms,
            "verification_effect": verification_effect,
        }

    def status(self, *, through=None, definitions=None):
        v = self.verification
        definitions = v.state["required_checks"] if definitions is None else definitions
        result = {}
        if (
            v.state["execution_contract"] == "declared"
            and "coding" in v.state["profiles"]
            and "code_behavior" in v.state["criteria"]
            and "code_behavior" not in definitions
        ):
            result["code_behavior"] = {
                "satisfied": False, "evidence_id": None,
                "check_fingerprint": "undeclared", "progress_event_id": 0,
                "outcome_fingerprint": "not_declared",
                "terminal": False,
                "next_action": "plan_task",
                "reason": "Declare required_check for code_behavior with plan_task (id and required_check; "
                "omit description to retain the existing criterion) before executing "
                "the relevant assertion suite. An arbitrary successful command cannot replace it.",
            }
        if not definitions:
            return result
        # Reconstruct from the durable journal; polls never invent a new launch.
        events = v.store.events(
            v.sid, after=v.state["evidence_start"],
            kinds=("tool_started", "tool_finished", "tool_error"),
        )
        starts, launches, results = {}, {}, {}
        for event in events:
            if through is not None and event["id"] > through:
                break
            body = event["body"]
            call_id = body["call_id"]
            if event["kind"] == "tool_started":
                if body["name"] in {"shell", "execute_program"}:
                    starts[call_id] = event
                continue
            observed = body.get("result", {})
            pid = observed.get("process_id")
            if body["name"] in {"shell", "execute_program"}:
                start = starts.get(call_id)
                if pid and start:
                    launches[pid] = start
            elif body["name"] == "poll_process":
                start = launches.get(pid)
            else:
                continue
            if not start:
                continue
            previous = results.get(start["id"])
            # Keep the first terminal result of a launch. Polling that same
            # finished handle again must not change a recorded check's identity.
            if previous is None or previous["body"]["result"].get("running"):
                results[start["id"]] = event
        for cid, definition in definitions.items():
            fingerprint = self.fingerprint(definition)
            matched_start, event = None, None
            for start in starts.values():
                if start["id"] <= definition["registered_event_id"]:
                    continue
                if start["body"]["name"] != definition["tool"]:
                    continue
                args = start["body"]["arguments"]
                key = "argv" if definition["tool"] == "execute_program" else "command"
                if args.get(key) != definition[key]:
                    continue
                candidate = results.get(start["id"])
                observed = candidate["body"]["result"] if candidate else {}
                # Actual runtime cwd is part of the fingerprint. Failed launches may
                # have no fingerprint; they must still supersede an earlier success.
                actual = observed.get("command_fingerprint")
                if actual:
                    if actual != fingerprint:
                        continue
                elif str(v.registry.resolve(args.get("cwd") or str(v.registry.cwd))) != definition["cwd"]:
                    continue
                matched_start, event = start, candidate
            reason = (
                "The latest declared command has no terminal result; collect its result or rerun after interruption"
                if matched_start else "The declared command has not run since registration"
            )
            action = "inspect_interrupted_check" if matched_start else "run_check"
            process_id = None
            if event:
                reason = (
                    "The declared command predates the latest possible change; rerun it"
                    if event["id"] < v.state["floor"] else v.support_error(event)
                )
                observed = event["body"]["result"]
                process_id = observed.get("process_id")
                if not reason and (observed.get("exit_code") != 0 or not observed.get("command_fingerprint")):
                    reason = "The declared check needs a successful terminal execution with runtime provenance"
                if observed.get("running"):
                    action = "poll_process"
                elif reason is None:
                    action = "record_verification"
                elif (
                    observed.get("ok") and observed.get("exit_code") == 0
                    and not observed.get("timed_out") and not observed.get("stopped")
                    and observed.get("command_fingerprint")
                ):
                    action = "run_check"
                else:
                    action = "inspect_failed_check"
            result[cid] = {
                "satisfied": reason is None,
                "check_fingerprint": fingerprint,
                "progress_event_id": (
                    event["id"] if event and not event["body"]["result"].get("running")
                    else matched_start["id"] if matched_start else 0
                ),
                "reason": f"Required check for {cid}: {reason}" if reason else None,
                "evidence_id": event["body"]["call_id"] if event else None,
                "next_action": action,
                "process_id": process_id,
                "outcome_fingerprint": self.outcome_fingerprint(event["body"]["result"])
                if event
                else "no_terminal_result",
                "terminal": bool(event and not event["body"]["result"].get("running")),
            }
        return result

    def required_reruns(self, calls):
        """Identify exact rechecks the runtime itself requires after a possible change.

        A failure, lost process, different command/cwd, or already-current success
        still needs ordinary inspection; nothing here grants execution permission.
        """
        if not self.verification.state["required_checks"]:
            return set()
        status = self.status()
        reruns = set()
        for call in calls:
            name = call["function"]["name"]
            args = json.loads(call["function"]["arguments"])
            for cid, definition in self.verification.state["required_checks"].items():
                row = status[cid]
                if row["next_action"] != "run_check" or not row["evidence_id"]:
                    continue
                key = "argv" if name == "execute_program" else "command"
                if (
                    name == definition["tool"] and args.get(key) == definition.get(key)
                    and str(self.verification.registry.resolve(args.get("cwd") or str(
                        self.verification.registry.cwd
                    ))) == definition["cwd"]
                ):
                    reruns.add(call["id"])
        return reruns
