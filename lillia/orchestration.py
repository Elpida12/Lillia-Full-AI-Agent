"""Durable work ownership and event-driven delegation reviews.

The ledger records the model's plan, not proof of success. Verification remains
responsible for accepting the requested outcome.
"""

from __future__ import annotations

import copy
import json
import uuid

from .progress import READ_TOOLS
from .tools import string

INSTRUCTIONS = """
Use update_work_plan for substantial work: bounded tasks, dependencies, ownership and evidence.
The ledger survives compaction and /continue. plan_task sets acceptance criteria, not work ownership.
Future children: owner=unassigned/status=pending. Your ready work: owner=parent/status=active.
Unfinished dependencies stay pending/blocked. Give reasons. spawn_subagent(work_item_id=...) assigns
ready pending work and sets its child owner/status. Review child evidence before marking work done;
reports alone are not completion. Keep integration and final verification explicit.
Completed items may be archived to make room for later work; their IDs remain valid completed
dependencies. Use work_plan_status(item_id=...) to inspect an archived item. Never reuse an archived ID.
Omit running child items from partial updates; their assignments stay unchanged. A written file does
not mean its child has finished. Use wait_subagent to obtain a terminal report, then inspect evidence.
Before spawning dependent work, reconcile its prerequisites using their existing work item IDs.
At a normal checkpoint, consider updating assignments and the explanation of current local work.
Unfinished parent work may remain active; never invent completion evidence to reconcile.
items=[] records a decision without changing ownership. Rejected updates are not applied, but ordinary
tools remain available. Review questions and tool counts are advisory; they do not establish a stall.
Refill available child slots when useful; do not invent assignments or exceed capacity. Prefer small
interface milestones to one large parent core prerequisite. Independent read-only review can start early.
The optional recovery field records a structured handoff or sequential-work explanation; it never
releases or grants tool access. Usually omit it and use decision or a work note to explain the causal
connection between an action and the user's objective. Explicit evidence IDs must identify journal
observations, not hashes. Finished failed or timed-out commands may justify repair but cannot verify success.
If remaining work becomes serial, explicitly assign all waiting items to parent and explain why;
preserve unfinished statuses/dependencies and promised handoffs. Planning does not expand permissions
or scope. Resource ownership is coordination, not a filesystem sandbox. Report incomplete work honestly.
"""

MANAGED_INSTRUCTIONS = """
A focused delegation assessment runs after initial discovery and at later milestones.
Selected assignments start automatically or queue for available child capacity. Continue different useful
work while children run. You can also call spawn_subagent(name, task, context, resources, read_only) directly;
the runtime creates its work item. Use work_item_id to assign an existing ready pending item.
Use update_work_plan for dependencies and ownership when useful. When a child's report is delivered,
the runtime marks its work item done with the report path; a delivered report is not proof of success,
so verify important claims before relying on them. Items whose child failed, was cancelled or ended
partial stay in review until you retry, reassign, finish or mark them; blocked items record a concrete
limitation to report. Existing items accept partial updates. Keep integration and final verification
explicit. Never mark work done to unlock a child. Queued and running children own their assigned resources; cancel before reassigning them.
Only require prerequisites the child actually needs. Requirements review and tests against a stable
interface can start before implementation or all tests finish. Preserve the user's scope and opt-outs.
Completed items are archived when the active ledger fills; their IDs remain completed dependencies.
Use work_plan_status(item_id=...) for historical details and new IDs for new assignments.
"""

COORDINATION = frozenset(
    {
        "update_work_plan",
        "work_plan_status",
        "spawn_subagent",
        "wait_subagent",
        "list_subagents",
        "cancel_subagent",
        "task_status",
        "inspect_evidence",
        "list_playbooks",
        "read_playbook",
        "forget_playbook",
        "set_playbook_capture",
        "plan_task",
        "record_verification",
        "finish_task",
        "record_work_note",
        "inspect_progress",
        "record_finding",
        "list_findings",
        "submit_result",
        "poll_process",
        "stop_process",
    }
)


class DelegationReplanRequired(ValueError):
    """A supplied optional recovery record is invalid; tools remain available."""


class WorkPlan:
    REVIEW_AFTER = 8  # Executed local tools while child capacity is available.
    REPLAN_AFTER = 2 * REVIEW_AFTER
    REVIEW_READS = 4  # Legacy journal metadata; this no longer limits observations.

    def __init__(self, manager):
        self.manager = manager
        self.parent = manager.parent
        self.store, self.sid = manager.store, manager.sid
        self.review_generation = 0
        event = self.store.latest_event(self.sid, "work_plan_state")
        self.state = event["body"] if event else self.empty()
        # Additive migration: old sessions can continue with their existing ledger.
        self.state.setdefault("review_reads", 0)
        self.state.setdefault("archive_floor", event["id"] if event else 0)
        self.state.setdefault("archive_through", self.state["archive_floor"])
        self.archived = {}
        if self.state["archive_through"] > self.state["archive_floor"]:
            for archived in self.store.events(
                self.sid, after=self.state["archive_floor"], kinds=("work_items_archived",)
            ):
                if archived["id"] <= self.state["archive_through"]:
                    self.archived.update(archived["body"]["items"])
        if "handoff" not in self.state:
            self.state["handoff"] = self.empty_handoff()
            self.state["handoff"]["idle_tools"] = self.legacy_idle_tools()

    def legacy_idle_tools(self):
        """Recover accumulated work across old reaffirmations, without rewriting history."""
        idle, previous = 0, 0
        owners = set()
        for event in self.store.events(self.sid, kinds=("work_plan_state",)):
            state = event["body"]
            items = state["items"].values()
            assigned = {i["owner"] for i in items if i["owner"] not in {"parent", "unassigned"}}
            waiting = any(i["owner"] == "unassigned" and i["status"] in {"pending", "blocked"} for i in items)
            if not waiting or assigned - owners:
                idle = 0
            if waiting:
                idle += max(0, state["local_tools"] - previous)
            previous, owners = state["local_tools"], assigned
        return idle

    @staticmethod
    def empty_handoff():
        return {
            "idle_tools": 0,
            "grace_tools": 0,
            "assignment": None,
            "milestone": None,
            "used_milestones": [],
            "evidence_floor": 0,
        }

    @staticmethod
    def empty():
        return {
            "items": {},
            "decision": "",
            "pending_review": [],
            "local_tools": 0,
            "handoff": WorkPlan.empty_handoff(),
            "review_reads": 0,
            "archive_floor": 0,
            "archive_through": 0,
        }

    def save(self, archived=None):
        if not archived:
            self.store.event(self.sid, "work_plan_state", self.state)
            return
        # Archive and the snapshot that removes these items are one durable change.
        # Full archived records are written once, not copied into every later snapshot.
        with self.store.db:
            cursor = self.store.db.execute(
                "INSERT INTO events(session_id,kind,body) VALUES(?,?,?)",
                (self.sid, "work_items_archived", json.dumps({"items": archived})),
            )
            self.state["archive_through"] = cursor.lastrowid
            self.store.db.execute(
                "INSERT INTO events(session_id,kind,body) VALUES(?,?,?)",
                (self.sid, "work_plan_state", json.dumps(self.state)),
            )
        self.archived.update(copy.deepcopy(archived))

    def compact_items(self, proposed):
        """Reclaim only completed records; unresolved work and dependency IDs survive."""
        proposed = dict(proposed)
        archived = {}

        def archive_completed(key):
            if key not in proposed:
                return
            item = proposed[key]
            if item["status"] != "done":
                raise ValueError("Cannot archive unfinished dependencies; reconcile completion first")
            # Freeze the completed dependency closure too: reopening a prerequisite
            # must never make an archived completion claim silently stale.
            for dependency in item["dependencies"]:
                archive_completed(dependency)
            archived[key] = proposed.pop(key)

        for key, item in list(proposed.items()):
            if len(proposed) <= 20 and len(json.dumps(proposed, ensure_ascii=False)) <= 16000:
                break
            if key in proposed and item["status"] == "done":
                archive_completed(key)
        # Keep compacting completed records, but an unlimited child pool must not be
        # limited indirectly by the number or combined size of unfinished assignments.
        if self.manager.config.max_active and len(proposed) > 20:
            raise ValueError("At most 20 unfinished work items; finish existing work before adding more")
        if self.manager.config.max_active and len(json.dumps(proposed, ensure_ascii=False)) > 16000:
            raise ValueError(
                "Work plan exceeds 16000 characters; shorten unfinished descriptions and evidence"
            )
        return proposed, archived

    def begin(self, fresh):
        if fresh:
            self.state = self.empty()
            self.archived = {}
            latest = self.store.latest_event(self.sid)
            self.state["archive_floor"] = self.state["archive_through"] = latest["id"] if latest else 0
            self.save()
        elif self.state["items"]:
            self.checkpoint("continued task; reconcile unfinished ownership before proceeding")

    def install(self, registry):
        item = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "id": string("Stable work item ID", 60),
                "task": string("Bounded outcome and expected checks", 500),
                "dependencies": {
                    "type": "array",
                    "items": string("Work item ID", 60),
                    "uniqueItems": True,
                    "maxItems": 20,
                },
                "resources": {
                    "type": "array",
                    "items": string("Owned path/resource; shared reads are OK", 300),
                    "uniqueItems": True,
                    "maxItems": 12,
                },
                "owner": string(
                    "parent for local work; unassigned for pending/blocked work; "
                    "child IDs are assigned only by spawn_subagent(work_item_id=...)",
                    100,
                ),
                "status": {
                    "enum": ["pending", "active", "review", "done", "blocked"],
                    "description": "Unassigned work stays pending until spawned, or blocked. "
                    "Active work needs an owner and completed dependencies. "
                    "Done work needs completed dependencies and observed evidence.",
                },
                "reason": string("Why this owner/status; concrete blocker if blocked", 500),
                "evidence": string("Actual evidence IDs/paths and observed results for completed work", 800),
            },
            "required": ["id"],
        }
        registry.add(
            "update_work_plan",
            "Add/update durable work items and record the next delegation decision. "
            "Partial updates preserve other items. Unfinished parent work may stay active. "
            "Checkpoints are advisory. A decision-only update may use items=[]. "
            "The optional recovery record never controls whether ordinary tools may execute.",
            {
                "items": {"type": "array", "items": item, "maxItems": 20},
                "decision": string(
                    "Next assignments and parent work, or concrete reason to stay local", 1000
                ),
                "recovery": {
                    "type": "object",
                    "additionalProperties": False,
                    "description": "Optional structured plan record: delegate requires assignment_id; "
                    "milestone requires assignment_id and milestone_id; sequential "
                    "uses reason (or decision) and fresh evidence_ids (or the latest "
                    "eligible observed tool result when omitted).",
                    "properties": {
                        "action": {
                            "enum": ["delegate", "milestone", "sequential"],
                            "description": "delegate: assignment_id; milestone: assignment_id and milestone_id; "
                            "sequential: reason defaults to decision; omitted evidence_ids "
                            "selects the latest eligible observed tool result.",
                        },
                        "assignment_id": string(
                            "Ready child item, or item unlocked directly by the milestone", 60
                        ),
                        "milestone_id": string(
                            "NEW ready parent milestone directly unlocking the assignment", 60
                        ),
                        "reason": string(
                            "Concrete serial constraint/release condition; defaults to decision", 1000
                        ),
                        "evidence_ids": {
                            "type": "array",
                            "items": string("Fresh finished tool evidence_id, not a file hash", 100),
                            "description": "Omit to select the latest eligible observation. Finished failed "
                            "or timed-out commands may justify repair, not a passing check.",
                            "minItems": 1,
                            "maxItems": 4,
                            "uniqueItems": True,
                        },
                    },
                    "required": ["action"],
                },
            },
            ["items", "decision"],
            "read",
            self.update,
        )
        registry.add(
            "work_plan_status",
            "Read current work and ownership, or retrieve one current/archived item by ID.",
            {"item_id": string("Optional current or archived work item ID", 60)},
            [],
            "read",
            self.status_tool,
        )

    def available(self) -> int | None:
        """Remaining child capacity; None means unlimited, zero means unavailable."""
        cap = self.manager.config.max_spawn_per_turn
        active_cap = self.manager.config.max_active
        slots = max(0, active_cap - self.manager.active_count()) if active_cap else None
        if cap:
            remaining = max(0, cap - self.manager.spawned)
            slots = remaining if slots is None else min(slots, remaining)
        budget = self.manager.config.max_model_calls
        if budget and self.parent.scheduler.calls >= budget:
            return 0
        return slots

    def enabled(self):
        names = {s["function"]["name"] for s in self.parent.registry.schemas()}
        return {"spawn_subagent", "update_work_plan", "work_plan_status"} <= names

    def managed(self):
        return self.manager.config.assessment_enabled and self.manager.config.strategy == "aggressive"

    def assessed(self, reason):
        """A controller decision acknowledges review without claiming any work is complete."""
        self.state.update(
            decision=reason[:1000],
            pending_review=[],
            local_tools=0,
            review_reads=0,
            handoff=self.empty_handoff(),
        )
        self.save()

    def prepare_assignment(self, name, task, resources=None, work_item_id=None):
        """Bind an existing item or supply routine ledger fields for one ready assignment."""
        if work_item_id is not None:
            self.assignment(work_item_id)
            return work_item_id
        if not self.managed():
            if self.enabled() and self.state["items"]:
                raise ValueError(
                    "Use work_item_id to bind this child to a pending item in the existing work plan"
                )
            return None
        key = "delegate-" + uuid.uuid4().hex[:12]
        item = dict(
            id=key,
            task=task[:500],
            dependencies=[],
            resources=list(resources or []),
            owner="unassigned",
            status="pending",
            reason=f"Delegate {name}"[:500],
            evidence="",
        )
        proposed, archived = self.compact_items({**self.state["items"], key: item})
        self.state["items"] = proposed
        self.save(archived)
        return key

    def checkpoint(self, reason):
        if not self.enabled() or reason in self.state["pending_review"]:
            return
        if not self.requires_review():
            self.state["review_reads"] = 0
        self.state["pending_review"].append(reason)
        self.save()
        self.store.event(
            self.sid, "orchestration_checkpoint", {"reason": reason, "available": self.available()}
        )
        if not self.managed():
            self.parent.registry.work_notes.review(
                status="unknown",
                reason=reason,
                question="At a normal checkpoint, explain how current work supports the task and "
                "whether independent work would help. Continue useful work while considering this question.",
                reviewer="work-plan monitor",
            )

    def observe(self, name, result):
        if name in COORDINATION or result.get("error") in {
            "read_batch_deferred",
            "orchestration_review_required",
            "delegated_read_pending",
        }:
            return
        review_read = name in READ_TOOLS and self.requires_review()
        if review_read:
            # Retain diagnostic counts for old journals; these never restrict observations.
            self.state["review_reads"] += 1
        if self.available() != 0:
            self.state["local_tools"] += 1
            handoff = self.state["handoff"]
            if self.waiting() or handoff["assignment"]:
                handoff["idle_tools"] += 1
                handoff["grace_tools"] = max(0, handoff["grace_tools"] - 1)
            if self.replan_needed():
                self.checkpoint(
                    "extended local work while assignments remain; causal context may need review"
                )
            if self.state["local_tools"] >= self.REVIEW_AFTER:
                self.checkpoint("extended local work with available child capacity")
            self.save()
        elif review_read:
            self.save()

    @staticmethod
    def _complete_from_child(item, row):
        result = row.get("result") or {}
        references = " ".join(
            f"{key}={result[key]}" for key in ("report_output", "saved_output") if result.get(key)
        )
        item["status"] = "done"
        item["reason"] = f"Child {row['status']}; its report was delivered to the parent."[:500]
        item["evidence"] = (
            f"Delivered report of child {row['child_id']}" + (f": {references}" if references else "")
        )[:800]
        # One-shot: a worker that later reopens this item (status review/pending) keeps its verdict.
        item["auto_completed"] = row["child_id"]

    def sync(self):
        changed = False
        rows = {row["child_id"]: row for row in self.manager.rows()}
        for item in self.state["items"].values():
            row = rows.get(item["owner"])
            if row is None or row["status"] in {"running", "queued"}:
                continue
            result = row.get("result") or {}
            # A delivered report completes the ledger item; the parent reviews claims through
            # verification, not by hand-marking every finished assignment.
            delivered = bool(row.get("collected")) and row["status"] in {"reported", "completed"}
            never_started = result.get("started") is False
            disposition = result.get("disposition") if never_started else None
            if item["status"] == "active":
                if delivered:
                    self._complete_from_child(item, row)
                elif disposition == "blocked":
                    # Superseded by the parent's final answer: a limitation to report, not open work.
                    item["status"] = "blocked"
                    item["reason"] = ("Never started: " + str(result.get("detail") or row["status"]))[:500]
                elif disposition == "pending":
                    # The turn, capacity or capability ended before it ran: back to the ready pool.
                    item.update(owner="unassigned", status="pending", evidence="")
                    item["reason"] = (
                        "Not started: " + str(result.get("detail") or row["status"]) + " Ready to retry or reassign."
                    )[:500]
                else:
                    item["status"] = "review"
                    item["reason"] = (
                        f"Child {row['status']}; inspect report and partial effects before completing or retrying."
                    )
                changed = True
            elif (
                item["status"] == "review"
                and delivered
                and item.get("auto_completed") != row["child_id"]
            ):
                # The runtime's own interim review marking; a worker's reopened verdict is kept.
                self._complete_from_child(item, row)
                changed = True
        if changed:
            # Actual child progress creates new evidence to reconcile, even when
            # another checkpoint is already pending. Repeated syncs do not refill retries.
            self.review_generation += 1
            self.checkpoint("assigned child finished; review evidence and remaining work")
            self.save()

    def ready(self, item, items=None):
        items = self.state["items"] if items is None else items
        return all(
            items.get(dep, self.archived.get(dep, {})).get("status") == "done" for dep in item["dependencies"]
        )

    def waiting(self):
        """Unassigned work, excluding work necessarily awaiting a running child."""
        items = self.state["items"]
        running = {r["child_id"] for r in self.manager.rows() if r["status"] in {"running", "queued"}}
        blocked = {}

        def waits_for_child(key):
            if key in self.archived:
                return False
            if key in blocked:
                return blocked[key]
            item = items[key]
            if item["status"] == "done":
                return False
            blocked[key] = item["owner"] in running or any(
                waits_for_child(dep) for dep in item["dependencies"]
            )
            return blocked[key]

        return [
            key
            for key, item in items.items()
            if item["owner"] == "unassigned"
            and item["status"] in {"pending", "blocked"}
            and not waits_for_child(key)
        ]

    def promised_assignment_ready(self):
        handoff = self.state["handoff"]
        if not handoff["assignment"]:
            return False
        milestone = handoff["milestone"]
        return (
            milestone is None
            or self.state["items"].get(milestone, self.archived.get(milestone, {})).get("status") == "done"
        )

    def replan_needed(self):
        if self.managed():
            return False
        handoff = self.state["handoff"]
        return bool(
            self.available() != 0
            and (
                self.promised_assignment_ready()
                or (
                    (handoff["idle_tools"] >= self.REPLAN_AFTER or handoff["assignment"])
                    and not handoff["grace_tools"]
                    and (self.waiting() or handoff["assignment"])
                )
            )
        )

    def recovery_evidence_problem(self, event):
        """Diagnostic observations can justify more work without proving success."""
        if event is None:
            return (
                "is not an observation evidence_id from the current task (file hashes are not evidence IDs)"
            )
        if event["id"] <= self.state["handoff"]["evidence_floor"]:
            return "is stale: it predates the previous sequential recovery; use fresh evidence"
        if event["kind"] != "tool_finished":
            return "records a tool error, not a finished observation"
        result = event["body"]["result"]
        if result.get("running"):
            return "belongs to a command that is still running; poll it to completion"
        if result.get("stopped"):
            return "belongs to a stopped command"
        if result.get("error"):
            return "records a tool or process error"
        if event["body"]["name"] in {"shell", "execute_program", "poll_process"}:
            if result.get("running") is not False or type(result.get("exit_code")) is not int:
                return "has no finished command result with a terminal exit_code"
        elif not result.get("ok") or result.get("timed_out") or result.get("exit_code", 0) != 0:
            return "is not a successful finished observation"
        return ""

    def recovery_evidence(self, evidence=None):
        evidence = self.parent.registry.verification.evidence() if evidence is None else evidence
        candidates = []
        for cid, event in sorted(evidence.items(), key=lambda entry: entry[1]["id"]):
            if self.recovery_evidence_problem(event):
                continue
            result = event["body"]["result"]
            observation = {
                "evidence_id": cid,
                "tool": event["body"]["name"],
                "status": "timed_out"
                if result.get("timed_out")
                else "failed"
                if result.get("exit_code", 0) != 0
                else "succeeded",
            }
            if "exit_code" in result:
                observation["exit_code"] = result["exit_code"]
            path = result.get("path") or result.get("log_path")
            if path:
                observation["path"] = str(path)[:300]
            candidates.append(observation)
        return candidates[-4:]

    def recovery_state(self, proposed, recovery):
        handoff = copy.deepcopy(self.state["handoff"])
        action = recovery.get("action")
        if action in {"delegate", "milestone"}:
            key = recovery.get("assignment_id")
            target = proposed.get(key)
            if not target or target["owner"] != "unassigned" or target["status"] != "pending":
                candidates = [
                    item_id
                    for item_id, item in proposed.items()
                    if item["owner"] == "unassigned" and item["status"] == "pending"
                ]
                problem = (
                    "is missing" if not key else f"={key!r} must name a pending unassigned child work item"
                )
                raise DelegationReplanRequired(
                    f"recovery.assignment_id {problem} for action={action!r}. "
                    f"Pending unassigned IDs: {json.dumps(candidates)}. "
                    "Use a ready item for delegate, or one unlocked by milestone_id for milestone. "
                    "If replan_required is false, omit recovery and record a concrete decision; "
                    "unfinished parent work can remain active."
                )
            milestone = recovery.get("milestone_id") if action == "milestone" else None
            if action == "delegate":
                if not self.ready(target, proposed):
                    raise DelegationReplanRequired("The promised child assignment must be ready now")
            else:
                step = proposed.get(milestone)
                if (
                    not step
                    or milestone in self.state["items"]
                    or milestone in self.archived
                    or milestone in handoff["used_milestones"]
                    or step["owner"] != "parent"
                    or step["status"] != "active"
                    or not self.ready(step, proposed)
                ):
                    raise DelegationReplanRequired(
                        "Create a NEW ready active parent milestone, not the existing large prerequisite"
                    )
                unfinished = {
                    dep
                    for dep in target["dependencies"]
                    if proposed.get(dep, self.archived.get(dep, {})).get("status") != "done"
                }
                if unfinished != {milestone}:
                    raise DelegationReplanRequired(
                        "The child must depend only on the new milestone and already completed work"
                    )
                handoff["used_milestones"].append(milestone)
            handoff.update(
                assignment=key, milestone=milestone, grace_tools=self.REVIEW_AFTER if milestone else 0
            )
        elif action == "sequential":
            if not recovery.get("reason", "").strip():
                raise DelegationReplanRequired(
                    "Sequential recovery needs a concrete reason/release condition; "
                    "omit reason to use decision"
                )
            evidence = self.parent.registry.verification.evidence()
            candidates = self.recovery_evidence(evidence)
            guidance = (
                " Recent eligible evidence_ids: "
                + json.dumps([candidate["evidence_id"] for candidate in candidates])
                + ". Omit evidence_ids to select the latest eligible observation, or use an available "
                "built-in read/poll_process to observe missing evidence."
            )
            if "evidence_ids" not in recovery:
                if not candidates:
                    raise DelegationReplanRequired(
                        "Sequential recovery needs fresh finished tool evidence." + guidance
                    )
                recovery["evidence_ids"] = [candidates[-1]["evidence_id"]]
            evidence_ids = recovery["evidence_ids"]
            if not evidence_ids:
                raise DelegationReplanRequired(
                    "Sequential recovery evidence_ids must not be empty." + guidance
                )
            for cid in evidence_ids:
                problem = self.recovery_evidence_problem(evidence.get(cid))
                if problem:
                    raise DelegationReplanRequired(
                        f"Sequential recovery evidence_id {cid!r} {problem}." + guidance
                    )
            latest = self.store.latest_event(self.sid)
            handoff.update(
                assignment=None,
                milestone=None,
                grace_tools=self.REVIEW_AFTER,
                evidence_floor=latest["id"] if latest else 0,
            )
        else:
            raise DelegationReplanRequired("Recovery action must be delegate, milestone, or sequential")
        return handoff

    async def update(self, items, decision, recovery=None):
        if not decision.strip():
            raise ValueError("Record a concrete delegation/local-work decision")
        self.sync()
        proposed = copy.deepcopy(self.state["items"])
        if len({i["id"] for i in items}) != len(items):
            raise ValueError("Work item IDs must be unique")
        children = {r["child_id"]: r for r in self.manager.rows()}
        for patch in items:
            if patch["id"] in self.archived:
                raise ValueError("Archived work item IDs are immutable; create a new ID for new work")
            old = proposed.get(patch["id"])
            defaults = dict(
                task="",
                dependencies=[],
                resources=[],
                owner="unassigned",
                status="pending",
                reason="Planned assignment",
                evidence="",
            )
            item = {**(old or defaults), **patch}
            if any(not item[key].strip() for key in ("id", "task", "reason")):
                raise ValueError("Work items require nonempty ID, task and reason")
            if old and old["owner"] in children and children[old["owner"]]["status"] in {"running", "queued"}:
                if item != old:
                    raise ValueError(
                        f"Work item {item['id']!r} is owned by running child {old['owner']!r}; "
                        "its assignment cannot change while the child is running. "
                        "This update was not applied. Omit this item and retry only the other changes "
                        "(items=[] is valid for a decision-only update). "
                        f"Use wait_subagent(agent_id={old['owner']!r}) to await its terminal report, "
                        "then inspect evidence before marking it done. A written file is not child completion. "
                        "Cancel only if you intend to stop that child's work."
                    )
            if item["owner"] not in {"parent", "unassigned"}:
                if not old or old["owner"] != item["owner"]:
                    raise ValueError(
                        "Assign a child through spawn_subagent(work_item_id=...), not by editing owner"
                    )
                if item["status"] in {"active", "pending"} and children[item["owner"]]["status"] not in {
                    "running",
                    "queued",
                }:
                    raise ValueError(
                        "Finished children cannot own active or pending work; choose parent or unassigned to retry"
                    )
            if item["status"] == "done" and not item["evidence"].strip():
                raise ValueError("Completed work needs observed completion evidence")
            if item["status"] == "active" and item["owner"] == "unassigned":
                raise ValueError(
                    f"Active work needs an owner: work item {item['id']!r} has owner='unassigned'. "
                    "For local work, set owner='parent'. For a child assignment, keep owner='unassigned' "
                    f"and set status='pending', then use spawn_subagent(work_item_id={item['id']!r}). "
                    "This update was not applied; correct it and retry update_work_plan. "
                    "Other useful work may continue under existing ownership."
                )
            proposed[item["id"]] = copy.deepcopy(item)
        visiting, visited = set(), set()

        def visit(key):
            if key in self.archived:
                return
            if key not in proposed:
                raise ValueError("Unknown work dependency")
            if key in visiting:
                raise ValueError("Work dependencies must not contain cycles")
            if key in visited:
                return
            visiting.add(key)
            for dep in proposed[key]["dependencies"]:
                visit(dep)
            visiting.remove(key)
            visited.add(key)

        for key, item in proposed.items():
            visit(key)
            if item["status"] in {"active", "done"} and not self.ready(item, proposed):
                raise ValueError("Active/completed work requires completed dependencies")
        parent_ids = {key for key, item in self.state["items"].items() if item["owner"] == "parent"}
        child_claims = [
            (path, owner) for path, owner in self.manager.owned_resources() if owner not in parent_ids
        ]
        for key, item in proposed.items():
            if item["owner"] != "parent" or item["status"] != "active":
                continue
            for resource in item["resources"]:
                path = self.parent.registry.resolve(resource)
                for claimed, owner in child_claims:
                    if path == claimed or path in claimed.parents or claimed in path.parents:
                        raise ValueError(
                            f"Parent work item {key!r} overlaps active ownership {owner!r}; "
                            "wait for or cancel the editing child before assigning these resources"
                        )
        recovery = copy.deepcopy(recovery) if recovery else None
        if recovery and recovery.get("action") == "sequential":
            recovery.setdefault("reason", decision)
        handoff = (
            self.recovery_state(proposed, recovery) if recovery else copy.deepcopy(self.state["handoff"])
        )
        if not any(item["owner"] == "unassigned" and item["status"] != "done" for item in proposed.values()):
            handoff.update(idle_tools=0, grace_tools=0, assignment=None, milestone=None)
        if handoff["assignment"] and (
            proposed[handoff["assignment"]]["status"] == "done"
            or proposed[handoff["assignment"]]["owner"] != "unassigned"
        ):
            handoff.update(assignment=None, milestone=None, grace_tools=0)
        previous_ready = self.promised_assignment_ready()
        proposed, archived = self.compact_items(proposed)
        # A completed handoff milestone may move to the archive in this update.
        # Publish the archive with the new state before calculating its readiness.
        self.state.update(
            items=proposed, decision=decision, pending_review=[], local_tools=0, handoff=handoff
        )
        if archived:
            self.save(archived)
        new_ready_assignment = self.promised_assignment_ready() and not previous_ready
        if not self.requires_review() or new_ready_assignment:
            self.state["review_reads"] = 0
        self.save()
        if recovery:
            self.store.event(self.sid, "delegation_recovery", recovery)
        self.store.event(
            self.sid,
            "orchestration_decision",
            {"decision": decision, "available": self.available(), "ready": self.status()["ready"]},
        )
        result = self.status()
        if recovery:
            result["recovery_applied"] = recovery
        return result

    def assignment(self, key):
        self.sync()
        item = self.state["items"].get(key)
        if not item or item["status"] != "pending" or item["owner"] not in {"parent", "unassigned"}:
            raise ValueError("Spawn requires a pending, unassigned/parent-owned work item")
        if not self.ready(item):
            blockers = [
                {
                    "id": dep,
                    "status": self.state["items"][dep]["status"],
                    "owner": self.state["items"][dep]["owner"],
                }
                for dep in item["dependencies"]
                if dep not in self.archived and self.state["items"][dep]["status"] != "done"
            ]
            raise ValueError(
                f"Work item dependencies are not complete for {key!r}: {json.dumps(blockers)}. "
                "No child was spawned. Reconcile these existing work item IDs with update_work_plan "
                "after inspecting completion evidence; files on disk do not update the ledger. "
                "Keep unfinished prerequisites active/pending and continue or wait for their work. "
                "Do not mark them done just to unblock spawning."
            )
        return item

    def assigned(self, key, child_id):
        self.state["items"][key].update(owner=child_id, status="active", evidence="")
        handoff = self.state["handoff"]
        handoff["idle_tools"] = 0
        if handoff["assignment"] == key:
            handoff.update(assignment=None, milestone=None, grace_tools=0)
        self.save()

    def requires_review(self):
        """Legacy diagnostic name; a suggested review never restricts execution."""
        if self.managed():
            return False
        substantial = self.state["items"] or self.state["local_tools"] >= self.REVIEW_AFTER
        return bool(
            self.enabled()
            and self.manager.config.strategy == "aggressive"
            and (self.state["pending_review"] or self.replan_needed())
            and substantial
        )

    def defer(self, name):
        # A missing explanation or an unresolved advisory is not an execution veto.
        return False

    def review_reads_remaining(self):
        return None

    def next_action(self):
        if not self.requires_review():
            return "Continue the planned work; update ownership and evidence as work progresses."
        if self.promised_assignment_ready():
            key = self.state["handoff"]["assignment"]
            return f"Consider the planned assignment {key!r} at a normal checkpoint; tools remain available."
        if self.replan_needed():
            return (
                "Independent assignments remain while local work continues. This does not establish a stall. "
                "At a normal checkpoint, consider explaining the causal plan in a work note or updating "
                "assignments. Ordinary tools remain available without a response or recovery record."
            )
        if not self.state["items"]:
            return (
                "Consider recording bounded work items when useful. Keep unfinished local work active "
                "and future child work pending. This suggestion does not block ordinary tools."
            )
        return (
            "At a normal checkpoint, consider recording the next decision or revising assignments. "
            "Keep unfinished parent work active; never mark it done merely to acknowledge a suggestion. "
            "Useful work may continue while this advisory remains unanswered."
        )

    def status(self):
        return {
            **self.state,
            "archived_count": len(self.archived),
            "available": self.available(),
            "waiting_for_delegation": self.waiting(),
            "replan_required": False,
            "review_required": False,
            "replan_suggested": self.replan_needed(),
            "review_suggested": self.requires_review(),
            "review_reads_remaining": self.review_reads_remaining(),
            "next_action": self.next_action(),
            "recovery_evidence": self.recovery_evidence(),
            "ready": [
                key
                for key, item in self.state["items"].items()
                if item["status"] == "pending" and self.ready(item)
            ],
            "children": self.manager.status(),
        }

    async def status_tool(self, item_id=None):
        self.sync()
        if item_id is not None:
            item = self.state["items"].get(item_id, self.archived.get(item_id))
            if item is None:
                raise ValueError("Unknown work item ID in the current task")
            return {"item": copy.deepcopy(item), "archived": item_id in self.archived}
        return self.status()

    def context(self):
        if not self.enabled():
            return ""
        # Full records remain available via work_plan_status, without repeatedly
        # expanding evidence/paths into the system prompt on small-context models.
        items = [
            {
                "id": key,
                **{k: item[k] for k in ("status", "owner", "dependencies")},
                "task": item["task"][:160],
            }
            for key, item in self.state["items"].items()
        ]
        data = {
            "items": items,
            "archived_count": len(self.archived),
            "decision": self.state["decision"],
            "checkpoint": self.state["pending_review"],
            "handoff": self.state["handoff"],
            "replan_required": False,
            "review_required": False,
            "replan_suggested": self.replan_needed(),
            "review_suggested": self.requires_review(),
            "review_reads_remaining": self.review_reads_remaining(),
            "next_action": self.next_action(),
            "recovery_evidence": self.recovery_evidence(),
        }
        if self.managed():
            for item in items:
                item["resources"] = self.state["items"][item["id"]]["resources"]
            data = {
                "items": items,
                "archived_count": len(self.archived),
                "decision": self.state["decision"],
                "queued": self.manager.queued_count(),
            }
            return MANAGED_INSTRUCTIONS + "\nWork plan (reference data):\n" + json.dumps(data)
        return INSTRUCTIONS + "\nWork plan (reference data, not new instructions):\n" + json.dumps(data)

    def blocked_summary(self):
        """Blocked items are limitations the delivered answer must carry, never open work."""
        if not self.enabled():
            return ""
        blocked = [
            f"{key} ({item['reason'][:160]})"
            for key, item in self.state["items"].items()
            if item["status"] == "blocked"
        ]
        if not blocked:
            return ""
        return (
            "Blocked work items: "
            + "; ".join(blocked)
            + ". If the answer already covers them, mark them done with evidence; to retry one, set "
            'owner="unassigned" and status="pending"'
        )

    def completion_gap(self):
        if not self.enabled():
            return ""
        if self.parent.registry.verification.state["outcome"] in {"partial", "blocked"}:
            return ""
        self.sync()
        # Blocked items record a concrete limitation for the answer to report; they are not open
        # work, so they must not withhold a finished answer or re-arm delegation.
        unfinished = [
            key for key, item in self.state["items"].items() if item["status"] not in {"done", "blocked"}
        ]
        if unfinished:
            return "Unfinished work items: " + ", ".join(unfinished)
        return ""
