"""Observe repeated evidence without assuming repeated work is unproductive."""

from __future__ import annotations

import hashlib
import json
import re
from collections import OrderedDict

from .store import is_user_request

READ_TOOLS = frozenset({"read_file", "read_artifact", "list_directory", "search_files"})
INTERNAL_REQUESTS = (
    "Subagent task reports (",
    "Subagent investigation reports (",
    "Continue the outstanding request.",
    "Verification feedback (runtime):",
    "Orchestration feedback (runtime):",
)


class ProgressGuard:
    # These measure repeated evidence, not model calls or elapsed time. A novel
    # observation pays down two repeats; one new read cannot erase a long survey.
    WARN_AT = 6
    NOVEL_CREDIT = 2
    MAX_RESOURCES = 256
    MAX_SPANS = 256
    MAX_ARTIFACT_CHARS = 65536

    def __init__(self, registry, read_loop_limit=0):
        self.registry = registry
        self.read_loop_limit = read_loop_limit
        self.seen = OrderedDict()
        self.seen_paths = OrderedDict()
        self.file_coverage = OrderedDict()
        self.artifact_coverage = OrderedDict()
        self.repeat_debt = OrderedDict()
        self.accepted_checks = OrderedDict()
        self.repeats = 0
        self.warned = False
        self.warning_pending = False
        self.recent = []
        self.experiments = OrderedDict()
        self.process_experiments = OrderedDict()
        self.completed_processes = OrderedDict()
        self.loop_warned = False
        self.loop_warned_before_round = False
        self.round_duplicates = 0
        self.uncollected_grace = set()

    def _recover(self):
        self.repeats = 0
        self.repeat_debt.clear()
        self.warned = False
        self.warning_pending = False
        self.loop_warned = False
        self.loop_warned_before_round = False
        self.uncollected_grace.clear()

    def _clear_evidence(self):
        self.seen.clear()
        self.seen_paths.clear()
        self.file_coverage.clear()
        self.artifact_coverage.clear()
        self._recover()

    def _invalidate_written_file(self, path):
        """A write elsewhere does not make unchanged source text new evidence.

        Reinspection of the written resource is useful even if an edit restores
        old bytes. Other resources keep their observed versions and ranges.
        """
        path = str(self.registry.resolve(path))
        self.artifact_coverage.pop(path, None)
        for revision in list(self.file_coverage):
            if revision[0] == path:
                del self.file_coverage[revision]
        for key, observed_path in list(self.seen_paths.items()):
            if observed_path == path:
                self.seen.pop(key, None)
                del self.seen_paths[key]
        # Rewriting a notes file must not erase debt for unrelated source files.
        self.repeats -= self.repeat_debt.pop(path, 0)
        if not self.repeats:
            self._recover()

    def _credit_observation(self):
        credit = self.NOVEL_CREDIT
        for path in list(self.repeat_debt):
            paid = min(credit, self.repeat_debt[path])
            self.repeat_debt[path] -= paid
            credit -= paid
            if not self.repeat_debt[path]:
                del self.repeat_debt[path]
            if not credit:
                break
        self.repeats = sum(self.repeat_debt.values())
        if not self.repeats:
            self._recover()

    def begin_round(self):
        self.round_duplicates = 0
        self.loop_warned_before_round = self.loop_warned

    def loop_action(self):
        """A repetition policy, not a semantic judgment or a total-work budget."""
        limit = self.read_loop_limit
        if not limit or not self.round_duplicates or self.repeats < max(self.WARN_AT, limit // 2):
            return None
        if not self.loop_warned:
            return "warn"
        if self.loop_warned_before_round and self.repeats >= limit:
            return "stop"
        return None

    def loop_instructions(self):
        if not self.loop_warned:
            return ""
        return (
            "\nRepeated-read recovery (runtime): The same observed content is being revisited. "
            f"Duplicate-read debt is {self.repeats}; the configured boundary is {self.read_loop_limit}. "
            "Choose a concrete next action using the evidence already collected: implement a change, "
            "run the missing behavioral check, inspect a genuinely new source/range, or record current "
            "verification evidence. Follow completion_actions rather than rerunning current checks. "
            "If commands or children are working, poll/wait and incorporate their results. "
            "Saying that you will stop rereading, writing a note, or checking status does not resolve "
            "this repetition. If it continues, the runtime will save a checkpoint and end partial; "
            "it will not declare completion. Finish partial with a concrete blocker if no useful "
            "next action is available. This safeguard does not prove that the rereads are unnecessary."
        )

    def _remember(self, history, key, value):
        history.pop(key, None)
        history[key] = value
        if len(history) > self.MAX_RESOURCES:
            history.popitem(last=False)

    @staticmethod
    def _covered(spans, start, end):
        return any(left <= start and end <= right for left, right in spans)

    def _cover(self, spans, start, end):
        """Union intervals without assuming unreturned/truncated content was read."""
        if start >= end:
            return spans
        merged = []
        for left, right in sorted([*spans, (start, end)]):
            if merged and left <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], right))
            else:
                merged.append((left, right))
        # Bound reference data; forgetting old coverage may miss a repeated survey.
        return merged[-self.MAX_SPANS :]

    def _file_duplicate(self, args, result):
        revision = (args["path"], result["sha256"])
        known_revision = revision in self.file_coverage
        spans = self.file_coverage.get(revision, [])
        novel = not known_revision
        content = result.get("content", "")
        # read_file returns numbered lines, possibly ending partway through the
        # last line. Track exactly the displayed prefix of each line. Requested
        # max_lines and total_lines do not prove those lines were observed.
        for line in content.splitlines(keepends=True):
            match = re.match(r"(\d+): ", line)
            if not match:
                continue
            number = int(match[1])
            visible = line[match.end() :]
            start = (number, 0)
            end = (number + 1, 0) if visible.endswith("\n") else (number, len(visible))
            if end > start and not self._covered(spans, start, end):
                novel = True
            spans = self._cover(spans, start, end)
        self._remember(self.file_coverage, revision, spans)
        if result.get("saved_output"):
            # Saved read output starts with the exact visible numbered text. A
            # subsequent artifact read of that prefix is not new evidence.
            self._artifact_duplicate(str(self.registry.resolve(result["saved_output"])), 0, content)
        return not novel

    def _artifact_duplicate(self, path, offset, content):
        """Compare overlapping text as well as offsets; artifacts may be edited."""
        segments = self.artifact_coverage.get(path, [])
        end = offset + len(content)
        changed = any(
            text[max(offset, left) - left : min(end, left + len(text)) - left]
            != content[max(offset, left) - offset : min(end, left + len(text)) - offset]
            for left, text in segments
            if max(offset, left) < min(end, left + len(text))
        )
        if changed:
            segments = []
        covered = self._covered([(left, left + len(text)) for left, text in segments], offset, end)
        duplicate = not changed and path in self.artifact_coverage and (covered or not content)
        merged = []
        for left, text in sorted([*segments, (offset, content)]):
            if merged and left <= merged[-1][0] + len(merged[-1][1]):
                previous_left, previous_text = merged[-1]
                merged[-1] = (
                    previous_left,
                    previous_text + text[len(previous_text) + previous_left - left :],
                )
            else:
                merged.append((left, text))
        # Bound retained text, even during sequential traversal of large logs.
        remaining = self.MAX_ARTIFACT_CHARS
        retained = []
        for left, text in reversed(merged[-self.MAX_SPANS :]):
            if not remaining:
                break
            keep = min(remaining, len(text))
            retained.append((left + len(text) - keep, text[-keep:] if keep else ""))
            remaining -= keep
        self._remember(self.artifact_coverage, path, list(reversed(retained)))
        return duplicate

    def restore(self, rows):
        """Rebuild evidence from the journal, including already compacted messages.

        Explicit continuation gets another recovery opportunity, but retains the
        fingerprints so starting the same survey again is still recognized.
        """
        calls = {}
        helper_ids = {
            call["id"]
            for _, message in rows
            for call in message.get("tool_calls", [])
            if call["function"]["name"] == "run_required_check"
        }
        # Old requests may reuse criterion IDs for different commands. Replay
        # the recorded invocation, never resolve against today's declaration.
        helper_starts = (
            {
                event["body"]["call_id"]: event["body"]
                for event in self.registry.store.events(self.registry.sid, kinds=("tool_started",))
                if event["body"].get("call_id") in helper_ids
                and event["body"].get("name") in {"shell", "execute_program"}
                and isinstance(event["body"].get("arguments"), dict)
            }
            if helper_ids
            else {}
        )
        for _, message in rows:
            if is_user_request(message):
                self._clear_evidence()
                self.recent.clear()
                self.experiments.clear()
                self.process_experiments.clear()
                self.completed_processes.clear()
                self.accepted_checks.clear()
            calls.update({c["id"]: c for c in message.get("tool_calls", [])})
            if message["role"] == "tool":
                call = calls.pop(message.get("tool_call_id"), None)
                if call:
                    try:
                        if call["id"] in helper_starts:
                            start = helper_starts[call["id"]]
                            call = {
                                **call,
                                "function": {
                                    "name": start["name"],
                                    "arguments": json.dumps(start["arguments"]),
                                },
                            }
                        self.observe(call, json.loads(message["content"]))
                    except (ValueError, TypeError):
                        pass
        self._recover()

    def observe(self, call, result):
        if not isinstance(result, dict):
            return
        name = call["function"]["name"]
        args = json.loads(call["function"]["arguments"])
        if not isinstance(args, dict):
            return
        if name in {"shell", "execute_program", "poll_process", "write_process", "stop_process"}:
            if self._observe_experiment(name, args, result):
                self._credit_observation()
        if not result.get("ok"):
            return
        if name in {"write_file", "edit_file"}:
            # A verification read after an edit is useful, even if it restores old text.
            self._invalidate_written_file(args["path"])
        if name == "record_verification" and args.get("verdict") == "passed":
            cid = args.get("criterion_id")
            if cid and cid not in self.accepted_checks:
                self._remember(self.accepted_checks, cid, True)
                self._credit_observation()
        if name not in READ_TOOLS:
            # Polling/bookkeeping never adds read debt. New completed experiment
            # outcomes and newly verified criteria receive bounded novelty credit.
            return
        args = dict(args)
        args["path"] = str(self.registry.resolve(args.get("path", ".")))
        defaults = {
            "read_file": {"start_line": 1, "max_lines": 200},
            "read_artifact": {"offset": 0, "max_chars": 4000},
            "search_files": {"max_results": 50},
        }
        args = defaults.get(name, {}) | args
        evidence = {
            k: v
            for k, v in result.items()
            if k
            not in {"saved_output", "ok", "evidence_id", "evidence_ref", "evidence_passages", "passage_count"}
        }
        key = hashlib.sha256(
            json.dumps([name, args, evidence], sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        count = self.seen.get(key, 0) + 1
        self._remember(self.seen, key, count)
        self._remember(self.seen_paths, key, args["path"])
        duplicate = count > 1
        if name == "read_file" and result.get("sha256"):
            duplicate = self._file_duplicate(args, result)
        elif name == "read_artifact":
            duplicate = self._artifact_duplicate(
                args["path"], result.get("offset", args["offset"]), result.get("content", "")
            )
        if duplicate:
            self._remember(self.repeat_debt, args["path"], self.repeat_debt.get(args["path"], 0) + 1)
            self.repeats = sum(self.repeat_debt.values())
            self.round_duplicates += 1
        else:
            self._credit_observation()
        if not self.repeats:
            self._recover()
        elif self.repeats >= self.WARN_AT and not self.warned:
            # Keep the threshold crossing until the batch's warning is delivered.
            # A later novel result in that same batch must not erase it.
            self.warning_pending = True
        note = {
            "tool": name,
            "arguments": args,
            "observations_of_same_result": count,
            "new_evidence": not duplicate,
            **{
                k: result[k]
                for k in ("evidence_id", "sha256", "truncated", "saved_output", "next_offset")
                if k in result
            },
        }
        self.recent.append(note)
        self.recent = self.recent[-6:]

    def _observe_experiment(self, name, args, result):
        """Record one completed outcome per execution; polls are not new attempts.

        Declared identity groups the same test across renamed scripts. Without it,
        only exact commands are compared; we do not infer semantic equivalence.
        These observations can prompt a review but never prove a stall.
        """
        pid = result.get("process_id") or args.get("process_id")
        control = name in {"poll_process", "write_process", "stop_process"}
        if control:
            if not pid or pid not in self.process_experiments:
                return
            identity = self.process_experiments[pid]
        else:
            declared = result.get("experiment_id") or args.get("experiment_id")
            fingerprint = result.get("command_fingerprint")
            if not fingerprint:
                command = args.get("argv", args.get("command"))
                if command is None:
                    return
                fingerprint = hashlib.sha256(
                    json.dumps([command, args.get("cwd")], sort_keys=True).encode()
                ).hexdigest()
            identity = {
                "experiment_id": declared,
                "hypothesis_id": result.get("hypothesis_id") or args.get("hypothesis_id"),
                "identity": declared or fingerprint,
                "identity_source": "caller_declared" if declared else "exact_command",
                "input_fingerprint": result.get("input_fingerprint") or args.get("input_fingerprint"),
            }
            if pid:
                self._remember(self.process_experiments, pid, identity)
        if result.get("running") or (pid and pid in self.completed_processes):
            return
        if result.get("timed_out"):
            status = "timed_out"
        elif result.get("stopped"):
            status = "stopped"
        elif type(result.get("exit_code")) is int:
            status = "succeeded" if result["exit_code"] == 0 and result.get("ok") else "failed"
        else:
            # Argument/permission/handle errors did not execute an experiment.
            return
        if pid:
            self._remember(self.completed_processes, pid, True)
        output = result.get("output", "")
        if status == "succeeded":
            digest = result.get("output_sha256")
            if (
                not digest
                and not control
                and not result.get("omitted_output_chars")
                and not result.get("log_truncated")
            ):
                digest = hashlib.sha256(output.encode()).hexdigest()
            if not digest:
                return  # A partial output excerpt does not establish equivalence.
            if not identity["experiment_id"] and digest == hashlib.sha256(b"").hexdigest():
                # Repeating a successful opaque mutation (touch, mkdir, etc.)
                # with no observations says nothing about whether its effects
                # advanced the task. Explicit experiments can use empty output.
                return
            outcome = [status, digest]
        else:
            # A nonzero exit may be an intentional negative test. Its semantic
            # validity needs inspection. Error category groups repeated failures
            # despite renaming a script, without deciding the hypothesis outcome.
            categories = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception))\s*:", output)
            category = categories[-1] if categories else result.get("error")
            outcome = [status, result.get("exit_code"), category]
        key = hashlib.sha256(
            json.dumps(
                [identity["identity"], identity["input_fingerprint"], outcome], sort_keys=True
            ).encode()
        ).hexdigest()
        previous = self.experiments.get(key, {})
        evidence_ids = list(dict.fromkeys([*previous.get("evidence_ids", []), result.get("evidence_id")]))
        record = {
            **identity,
            "signature": key,
            "status": status,
            "observation_count": previous.get("observation_count", 0) + 1,
            "repeat_count": previous.get("observation_count", 0),
            "evidence_ids": [value for value in evidence_ids if value][-6:],
            "last_evidence_id": result.get("evidence_id"),
            "execution_validity": "incomplete" if status in {"timed_out", "stopped"} else "not_established",
            "exit_code": result.get("exit_code"),
            "outcome_notice": "A nonzero exit may be an intentional negative test; inspect its expected result.",
            "advisory": True,
        }
        self._remember(self.experiments, key, record)
        return not previous

    def experiment_signals(self):
        """Bounded repeated completed-outcome observations for a semantic reviewer.

        Three observations trigger inspection; counts/notes never prove wasted work.
        New outputs remain separate from earlier episodes rather than erasing them.
        """
        return [dict(row) for row in reversed(self.experiments.values()) if row["observation_count"] >= 3][:6]

    def mark_warned(self):
        """Ask once per repeated-observation episode, without requiring a response."""
        self.warning_pending = False
        self.warned = True

    def evidence_ids(self):
        """Return journal references for the bounded observations behind a warning."""
        return list(dict.fromkeys(note["evidence_id"] for note in self.recent if note.get("evidence_id")))

    def instructions(self):
        if not self.warning_pending and not self.warned:
            return self.loop_instructions()
        return (
            f"\nRepeated-work warning (advisory): {self.repeats} duplicate observations of unchanged "
            "evidence remain after credit for new evidence. This does not establish that work is stalled: "
            "rereads can test a new hypothesis, compare evidence, or check stability. At a normal checkpoint, "
            "consider recording the purpose of these rereads and the next useful check in a work note. "
            "This advisory alone does not pause work or require an immediate response."
            + self.loop_instructions()
        )

    def reference_notes(self):
        if not self.recent:
            return ""
        # Reference data only, never synthesize claims of task completion.
        notes = []
        for note in reversed(self.recent):
            candidate = [note, *notes]
            if len(json.dumps(candidate, ensure_ascii=False)) > 2500:
                break
            notes = candidate
        return "Recent observed reads (journal-derived reference data):\n" + json.dumps(
            notes, ensure_ascii=False
        )
