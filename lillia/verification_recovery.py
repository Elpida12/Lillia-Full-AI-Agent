"""Bound repeated attempts to pass the same unresolved execution contract."""

from __future__ import annotations

import json


class VerificationRecovery:
    WARN_AT = 2
    STOP_AT = 3

    def __init__(self, registry):
        self.registry = registry
        self.blockers = {}
        self.round_blockers = set()
        self.warned_before_round = set()
        self.uncollected_grace = set()
        self.round_successes = set()
        self.cycle_counted = set()
        self.plan_revision_after = 0
        self.replay_definitions = None

    @staticmethod
    def _key(blocker):
        if not isinstance(blocker, dict) or blocker.get("kind") not in {"required_check", "stale_evidence"}:
            return None
        cid = blocker.get("criterion_id")
        fingerprint = blocker.get("check_fingerprint")
        progress = blocker.get("progress_event_id")
        if (
            not isinstance(cid, str)
            or not cid
            or not isinstance(fingerprint, str)
            or not fingerprint
            or type(progress) is not int
            or progress < 0
        ):
            return None
        outcome = blocker.get("outcome_fingerprint")
        if outcome is not None and (not isinstance(outcome, str) or not outcome):
            return None
        # Older saved journals have no content fingerprint. Retain their legacy
        # identity without treating event IDs as progress for new observations.
        return blocker["kind"], cid, fingerprint, outcome or progress

    @staticmethod
    def _slot(blocker):
        cid = blocker["criterion_id"]
        return cid if blocker["kind"] == "required_check" else (blocker["kind"], cid)

    def _clear(self, cid):
        self.blockers.pop(cid, None)
        self.round_blockers.discard(cid)
        self.warned_before_round.discard(cid)
        if not self.blockers:
            self.uncollected_grace.clear()

    def _plan_revision(self, body):
        """Discard obsolete causal episodes, never debt for mere rewording.

        A changed expectation/description does not change what a command does or
        make stale observations current. Retirement or a different executable
        method does obsolete that criterion's old episode. Other criteria keep
        their debt; no revision constitutes a successful verification.
        """
        cid = body["criterion_id"]
        prior = body["previous"].get("required_check")
        replacement = body.get("replacement")
        current = replacement.get("required_check") if replacement else None
        fingerprint = self.registry.verification.required_checks.fingerprint
        changed = bool(prior) != bool(current) or (bool(prior) and fingerprint(prior) != fingerprint(current))
        if body["action"] == "retire" or changed:
            self._clear(cid)
            self._clear(("stale_evidence", cid))

    def _sync_plan_revisions(self):
        verification = self.registry.verification
        state = getattr(verification, "state", {})
        committed = state.get("plan_revision_event_id") or 0
        if self.replay_definitions is not None or committed <= self.plan_revision_after:
            return
        committed_ids = self._committed_plan_revision_ids()
        for event in self.registry.store.events(
            self.registry.sid,
            after=max(state["evidence_start"], self.plan_revision_after),
            kinds=("plan_revision",),
        ):
            if event["id"] > committed:
                break
            if event["id"] in committed_ids:
                self._plan_revision(event["body"])
            self.plan_revision_after = event["id"]

    def _committed_plan_revision_ids(self):
        state = self.registry.verification.state
        committed = state.get("plan_revision_event_id") or 0
        if not committed:
            return set()
        # The full audit is written before the ledger commit. A crash can leave
        # an orphan audit record; a later revision must not make it effective.
        return {
            row["revision_id"]
            for row in self.registry.store.db.execute(
                "SELECT DISTINCT json_extract(body,'$.plan_revision_event_id') AS revision_id "
                "FROM events WHERE session_id=? AND kind='verification_state' AND id>?",
                (self.registry.sid, state["evidence_start"]),
            )
            if row["revision_id"] is not None and row["revision_id"] <= committed
        }

    def begin_round(self):
        self.sync()
        self.round_blockers.clear()
        self.round_successes.clear()
        self.cycle_counted.clear()
        self.warned_before_round = {cid for cid, row in self.blockers.items() if row["warned"]}

    @staticmethod
    def _debt(row):
        # Successful self-invalidating cycles are not rejected tool calls.
        return max(row["rejections"], 1 + row.get("repeat_cycles", 0))

    def _cycle_checks(self, through=None):
        verification = self.registry.verification
        if not getattr(self.registry, "store", None):
            return {}
        evidence = verification.evidence()
        definitions = (
            verification.state["required_checks"]
            if self.replay_definitions is None
            else self.replay_definitions
        )
        rows = verification.required_checks.status(through=through, definitions=definitions)
        checks = {}
        for cid, definition in definitions.items():
            if through is not None and definition["registered_event_id"] > through:
                continue
            row = rows[cid]
            event = evidence.get(row.get("evidence_id"))
            result = event["body"]["result"] if event else {}
            varied = any(
                result.get(key) is not None
                for key in (
                    "experiment_id",
                    "hypothesis_id",
                    "input_fingerprint",
                )
            )
            checks[cid] = {
                "signature": [row["check_fingerprint"], row["outcome_fingerprint"], varied],
                "event_id": row["progress_event_id"],
                "evidence_id": row.get("evidence_id"),
                "terminal": row["terminal"],
                "eligible": bool(
                    row["terminal"]
                    and result.get("ok")
                    and result.get("exit_code") == 0
                    and not result.get("timed_out")
                    and not result.get("stopped")
                    and result.get("verification_effect") == "mutation"
                    and not varied
                ),
            }
        return checks

    def _count_cycles(self):
        """Recognize an unchanged registered check invalidated again at a boundary.

        This is deliberately narrower than detecting repeated successful commands:
        an actual stale rejection enrolls the episode, both causal executions must
        be registered checks, and explicit experiments remain advisory.
        """
        if not self.round_successes or not any(
            row["kind"] == "stale_evidence" for row in self.blockers.values()
        ):
            return
        checks = self._cycle_checks()
        signature = {cid: row["signature"] for cid, row in checks.items()}
        floor = self.registry.verification.state["floor"]
        invalidators = {
            cid
            for cid, row in checks.items()
            if row["eligible"] and row["event_id"] == floor and floor in self.round_successes
        }
        for slot, blocker in self.blockers.items():
            if (
                blocker["kind"] != "stale_evidence"
                or "cycle_signature" not in blocker
                or slot in self.cycle_counted
            ):
                continue
            update = None
            changed = signature != blocker["cycle_signature"]
            foreign_change = self._other_mutation(blocker["cycle_after_event_id"], floor)
            if changed and not foreign_change and not all(check["terminal"] for check in checks.values()):
                # Re-registration or an in-flight rerun is not changed terminal
                # evidence. Wait for the cohort before forgiving a known cycle.
                continue
            if changed or foreign_change:
                # A changed method/cohort, input identity or terminal observation
                # gets a fresh opportunity; it never becomes a verification pass.
                update = {
                    "cycle_signature": signature,
                    "repeat_cycles": 0,
                    "cycle_after_event_id": max([floor, *self.round_successes]),
                    "warned": False,
                    "cycle_reset_reason": "changed check observation/cohort" if changed else "other mutation",
                }
            else:
                row = checks.get(blocker["criterion_id"])
                if (
                    blocker.get("dormant")
                    or not row
                    or not row["eligible"]
                    or row["event_id"] <= blocker["cycle_after_event_id"]
                    or row["event_id"] >= floor
                    or not (invalidators - {blocker["criterion_id"]})
                    or not all(check["eligible"] for check in checks.values())
                ):
                    continue
                update = {
                    "repeat_cycles": blocker.get("repeat_cycles", 0) + 1,
                    "cycle_after_event_id": floor,
                    "cycle_evidence_id": row["evidence_id"],
                    "cycle_invalidator_event_id": floor,
                    "cycle_invalidator_criteria": sorted(invalidators - {blocker["criterion_id"]}),
                }
                self.round_blockers.add(slot)
            blocker.update(update)
            self.cycle_counted.add(slot)
            self.registry.store.event(
                self.registry.sid,
                "verification_recovery_cycle_state",
                {
                    "blocker": {
                        key: blocker[key]
                        for key in (
                            "kind",
                            "criterion_id",
                            "check_fingerprint",
                            "outcome_fingerprint",
                            "progress_event_id",
                        )
                    },
                    "state": update,
                },
            )

    def _other_mutation(self, after, floor):
        """Do not mistake actual intervening work for a check-only cycle."""
        definitions = self.registry.verification.state["required_checks"].values()
        fingerprints = {
            self.registry.verification.required_checks.fingerprint(definition) for definition in definitions
        }
        events = self.registry.store.db.execute(
            "SELECT DISTINCT e.id,e.session_id,e.kind,e.body FROM events v JOIN events e "
            "ON e.id=json_extract(v.body,'$.floor') WHERE v.session_id=? "
            "AND v.kind='verification_state' AND e.id>? AND e.id<=?",
            (self.registry.sid, after, floor),
        )
        for event in events:
            if event["session_id"] != self.registry.sid:
                return True
            body = json.loads(event["body"])
            if event["kind"] == "tool_started":
                args = body.get("arguments", {})
                if any(
                    body.get("name") == definition["tool"]
                    and args.get("argv" if definition["tool"] == "execute_program" else "command")
                    == definition.get("argv" if definition["tool"] == "execute_program" else "command")
                    and str(self.registry.resolve(args.get("cwd") or str(self.registry.cwd)))
                    == definition["cwd"]
                    for definition in definitions
                ):
                    continue
            elif (
                event["kind"] == "tool_finished"
                and body.get("name") in {"shell", "execute_program", "poll_process"}
                and body.get("result", {}).get("command_fingerprint") in fingerprints
            ):
                continue
            return True
        return False

    def observe(self, call, result):
        if not isinstance(result, dict):
            return
        function = call.get("function", {})
        name = function.get("name")
        try:
            args = json.loads(function.get("arguments", "{}"))
        except (ValueError, TypeError):
            return
        if not isinstance(args, dict):
            return
        if name == "revise_plan" and result.get("ok"):
            self._sync_plan_revisions()
            return
        if name == "record_verification" and args.get("verdict") == "passed":
            if result.get("ok"):
                return
            blockers = [result.get("verification_blocker")]
        elif name == "plan_task":
            blockers = [result.get("verification_blocker")]
        elif name in {"shell", "execute_program", "poll_process"}:
            checks = self.registry.verification.required_checks
            outcomes = (
                checks.execution_outcomes(call, result, definitions=self.replay_definitions)
                if hasattr(checks, "execution_outcomes")
                else []
            )
            blockers = []
            for outcome in outcomes:
                if outcome.pop("execution_succeeded", False):
                    # A successful exact execution resolves its failed-execution
                    # episode, including on journal replay. It does not erase
                    # stale-evidence/cohort debt; that waits for reconciliation.
                    self._clear(outcome["criterion_id"])
                    self.round_successes.add(outcome["progress_event_id"])
                else:
                    blockers.append(outcome)
        elif name in {"finish_task", "submit_result"} and args.get("outcome") == "complete":
            blockers = result.get("verification_blockers", [])
        else:
            return
        if result.get("ok") is not False or not isinstance(blockers, list):
            return
        seen = set()
        for blocker in blockers:
            key = self._key(blocker)
            if key is None or key in seen:
                continue
            seen.add(key)
            cid = blocker["criterion_id"]
            if name == "record_verification" and cid != args.get("criterion_id"):
                continue
            slot = self._slot(blocker)
            row = self.blockers.get(slot)
            if row is None or self._key(row) != key:
                self._clear(slot)
                row = self.blockers[slot] = {
                    **blocker,
                    "rejections": 0,
                    "warned": False,
                }
                if blocker["kind"] == "stale_evidence" and getattr(self.registry, "store", None):
                    after = blocker.get("invalidator_event_id", blocker["progress_event_id"])
                    row.update(
                        cycle_signature={
                            cid: check["signature"]
                            for cid, check in self._cycle_checks(through=after).items()
                        },
                        repeat_cycles=0,
                        cycle_after_event_id=after,
                    )
            else:
                row.update(blocker)  # Latest references, while preserving episode debt.
            row["rejections"] += 1
            row["last_call_id"] = call.get("id")
            self.round_blockers.add(slot)

    def sync(self):
        """Only progress on the blocked contract clears its rejection history.

        Reconcile the final batch state, never a transient mid-batch success.
        Stale-evidence debt stays dormant until the recorded task cohort is
        coherent, so passing A then mutating B cannot erase an A/B stale cycle.
        """
        self._sync_plan_revisions()
        if not self.blockers:
            return
        status = self.registry.verification.required_checks.status()
        verification = self.registry.verification
        state = getattr(verification, "state", {})
        coherent = bool(state.get("criteria")) and not verification.gaps() if state else False
        resolved = []
        for slot, blocker in list(self.blockers.items()):
            cid = blocker["criterion_id"]
            if blocker["kind"] == "stale_evidence":
                if coherent:
                    resolved.append(
                        {
                            key: blocker[key]
                            for key in ("kind", "criterion_id", "check_fingerprint", "outcome_fingerprint")
                        }
                    )
                    self._clear(slot)
                else:
                    check = state.get("checks", {}).get(cid, {})
                    blocker["dormant"] = bool(
                        check.get("verdict") == "passed"
                        and check.get("revision") == state.get("revision")
                        and cid not in verification.gaps()
                    )
                    if blocker["dormant"]:
                        self.round_blockers.discard(slot)
                continue
            row = status.get(cid)
            if (
                row is None
                or row.get("satisfied")
                or row.get("check_fingerprint") != blocker["check_fingerprint"]
                or (
                    row.get("terminal")
                    and row.get("outcome_fingerprint") != blocker.get("outcome_fingerprint")
                    if blocker.get("outcome_fingerprint") is not None
                    else row.get("progress_event_id") != blocker["progress_event_id"]
                )
            ):
                self._clear(slot)
        if resolved and getattr(self.registry, "store", None) is not None:
            # A repaired cohort is real progress even if later work invalidates
            # it again. Preserve this boundary across interrupted/resumed runs.
            self.registry.store.event(
                self.registry.sid, "verification_recovery_resolved", {"blockers": resolved}
            )

    def action(self):
        self.sync()
        self._count_cycles()
        for cid in self.round_blockers:
            row = self.blockers[cid]
            if self._debt(row) >= self.STOP_AT and cid in self.warned_before_round and row["warned"]:
                return "stop"
        if any(
            self._debt(self.blockers[cid]) >= self.WARN_AT and not self.blockers[cid]["warned"]
            for cid in self.round_blockers
        ):
            return "warn"
        return None

    def mark_warned(self):
        for cid in self.round_blockers:
            row = self.blockers[cid]
            if self._debt(row) >= self.WARN_AT:
                row["warned"] = True

    def snapshot(self):
        return {
            "warn_at": self.WARN_AT,
            "stop_at": self.STOP_AT,
            "blockers": [dict(row) for row in self.blockers.values()],
        }

    def context(self):
        active = [
            {
                "criterion_id": row["criterion_id"],
                "rejections": row["rejections"],
                "repeat_cycles": row.get("repeat_cycles", 0),
            }
            for row in self.blockers.values()
            if row["warned"] and not row.get("dormant")
        ]
        if not active:
            return ""
        return (
            "\nVerification-loop recovery (runtime): The same verification blocker remains "
            "unresolved after failed executions, rejected verification, or unchanged successful "
            "checks that invalidate one another: "
            + json.dumps(active, ensure_ascii=False)
            + ". Consult task_status completion_actions and take the exact next action for that "
            "criterion. Successful substitute commands, unrelated reads, and new notes do not "
            "satisfy the declared check. Inspect failures and the actions invalidating evidence. "
            "Do not rewrite immutable checks or weaken acceptance requirements. If a check cannot succeed, "
            "finish partial with the concrete blocker. Another rejection without progress on "
            "that check, or another unchanged self-invalidating check cycle after this recovery "
            "opportunity, will save a checkpoint and end partial; "
            "it will not declare completion."
        )

    def restore(self):
        """Restore current-request debt; a saved stop permits a fresh continuation.

        Rebuild from durable tool events, including compacted history. An interrupted
        run receives warning context before its first new recovery round, even if
        interruption prevented the original warning from being delivered.
        """
        self.blockers.clear()
        self.round_blockers.clear()
        self.warned_before_round.clear()
        self.uncollected_grace.clear()
        self.round_successes.clear()
        self.cycle_counted.clear()
        self.plan_revision_after = 0
        verification = self.registry.verification
        events = self.registry.store.events(
            self.registry.sid,
            after=verification.state["evidence_start"],
            kinds=(
                "tool_started",
                "tool_finished",
                "tool_error",
                "verification_loop_stopped",
                "verification_recovery_resolved",
                "verification_recovery_cycle_state",
                "verification_loop_recovery",
                "plan_revision",
            ),
        )
        # Current definitions cannot match failures before re-registration. Walk
        # committed amendments backwards, then replay their definitions forward
        # without changing the actual verification ledger or its freshness gates.
        committed_ids = self._committed_plan_revision_ids()
        revisions = [
            event for event in events if event["kind"] == "plan_revision" and event["id"] in committed_ids
        ]
        self.replay_definitions = dict(verification.state["required_checks"])
        for event in reversed(revisions):
            body = event["body"]
            prior = body["previous"].get("required_check")
            if prior is None:
                self.replay_definitions.pop(body["criterion_id"], None)
            else:
                self.replay_definitions[body["criterion_id"]] = prior
        calls = {}
        warning_resets = set()
        for event in events:
            body = event["body"]
            if event["kind"] == "plan_revision":
                if event["id"] not in committed_ids:
                    continue
                self._plan_revision(body)
                cid = body["criterion_id"]
                replacement = body.get("replacement")
                definition = replacement.get("required_check") if replacement else None
                if definition is None:
                    self.replay_definitions.pop(cid, None)
                else:
                    self.replay_definitions[cid] = {**definition, "registered_event_id": event["id"]}
                self.plan_revision_after = event["id"]
                warning_resets &= {self._key(row) for row in self.blockers.values()}
            elif event["kind"] == "verification_loop_stopped":
                self.blockers.clear()
                calls.clear()
                warning_resets.clear()
            elif event["kind"] == "verification_recovery_resolved":
                for blocker in body.get("blockers", []):
                    if isinstance(blocker, dict) and blocker.get("kind") == "stale_evidence":
                        self._clear(self._slot(blocker))
                        warning_resets = {
                            key
                            for key in warning_resets
                            if key[:2] != (blocker["kind"], blocker["criterion_id"])
                        }
            elif event["kind"] == "verification_recovery_cycle_state":
                blocker = body.get("blocker", {})
                if self._key(blocker) is not None:
                    row = self.blockers.get(self._slot(blocker))
                    if row is not None and self._key(row) == self._key(blocker):
                        row.update(body["state"])
                        if body["state"].get("warned") is False:
                            warning_resets.add(self._key(row))
            elif event["kind"] == "verification_loop_recovery":
                for blocker in body.get("blockers", []):
                    key = self._key(blocker)
                    if key is not None:
                        warning_resets.discard(key)
            elif event["kind"] == "tool_started":
                if body.get("name") not in {
                    "record_verification",
                    "finish_task",
                    "submit_result",
                    "plan_task",
                    "shell",
                    "execute_program",
                    "poll_process",
                }:
                    continue
                if not isinstance(body.get("arguments"), dict) or not isinstance(body.get("call_id"), str):
                    continue
                calls[body["call_id"]] = {
                    "id": body["call_id"],
                    "function": {"name": body["name"], "arguments": json.dumps(body["arguments"])},
                }
            elif call := calls.pop(body.get("call_id"), None):
                self.observe(call, body.get("result"))
        self.replay_definitions = None
        self.sync()
        for row in self.blockers.values():
            # A durable progress reset requires a new warning opportunity just
            # as in an uninterrupted run, even with older rejection debt.
            if self._key(row) not in warning_resets:
                row["warned"] = self._debt(row) >= self.WARN_AT
        self.round_blockers.clear()
        self.round_successes.clear()
        self.warned_before_round.clear()
