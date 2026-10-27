"""Task-specific completion evidence, independent of lossy conversation summaries.

Adapted conceptually from Lillia's verification ladder and demonstrations: an
executed command is evidence, not a proof of conformance. No coding-only runner
or permission bypass is imported here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .evidence import EvidenceReferences
from .planning_authority import OUTCOME, PlanningAuthority
from .verification_checks import RequiredChecks, VerificationBlocked

PROFILES = {
    "general": "Check the requested outcome directly; scale checks to the task. No coding tests required.",
    "research": "Check sources and exploratory reproductions for claims, including paths and relevant "
    "source passages for code investigations. Check URLs, dates and freshness when relevant.",
    "system": "Inspect before/after state; confirm the requested system change and relevant health checks.",
    "coding": "Verify requested behavior, component integration and relevant edge cases through execution. "
    "A successful import or build alone does not demonstrate features. Check persistence and "
    "long-running behavior when relevant. Review advertised features against implementation; "
    "use an independent review for substantial changes when useful. Do not weaken acceptance "
    "criteria or tests merely to obtain a pass.",
}
INSTRUCTIONS = """
Verification changes no permissions. Before substantial work, plan_task observable criteria with profiles:
general, research, system, coding, or a combination. Use coding for software changes (including via shell)
or executable acceptance checks; use research for investigation/reproductions, system for
maintenance. Coding needs execute_program or shell; report missing capabilities instead of accepting
an impossible plan. Simple conversation needs no plan; small tasks need proportionate checks.
Preregister immutable required_check for executable acceptance validation.
Exploratory commands need no declaration. Record predicted causes/counts/results with record_finding.
plan_task {id,finding_id} investigates without prescribing the result. Supported/refuted need current anchored finding evidence;
inconclusive remains unresolved. __requested_outcome__ still requires delivery; strict criteria stay required.
Inspect interfaces before assertions; never freeze guessed names or values.
Prefer a test-file/suite to long inline code. Repair test mistakes against the user's
requirements without weakening assertions. Run a saved declaration with run_required_check(criterion_id,
timeout_seconds); choose a timeout explicitly. This uses the exact stored command and normal permissions.
Rewritten commands or helper scripts cannot substitute for a different registered command. If an immutable
inline check is wrong or impossible, finish partial with the specific declaration error and available results;
do not change correct application behavior just to satisfy a mistaken test or repeatedly claim equivalence.
New coding tasks require one for code_behavior. Use full checks from task instructions/configuration
and behavioral assertions; a partial command is insufficient. Results attach automatically; judge adequacy.
New strict criteria need description. To attach required_check, use an existing id and omit its immutable description.
For mutating checks, use one suite that fails if any check fails, or follow mutations with genuinely
observational checks. These checks work across domains.
record_verification accepts evidence_selections: {evidence_ref, passage_id, reasoning}. Select the short
reference and exact labeled passage from an observation; the runtime fills its canonical ID and quote.
Use inspect_evidence(evidence_ref, page) for more passages and task_status to recover references.
Selections still require criterion-specific reasoning; selecting a passage does not prove its relevance.
With evidence_support, omit evidence_ids to infer its IDs; explicit lists must cover them. Use observation
IDs, not process_id (even with call_). Error evidence_id identifies the rejected call. Inspection and
selection never refresh evidence.
Keep explanation to a brief outcome and limitations (at most 1500 characters); put the full report in
the final answer. Every fresh cited result needs a selection or
evidence_support: an exact contiguous excerpt and criterion-specific reasoning. Preserve whitespace and
read_file line numbers. For separate passages, use separate entries with the same ID; never join skipped
or reordered lines. Keep interpretation in reasoning. Repair rejected quotes from the original result or
error sample. Do not invent quotes or use unrelated fresh evidence to support older claims.
For silent commands, exit_code=0 establishes execution status only: explain the assertions actually run.
You judge whether evidence supports the claim; source anchoring checks quotations, not semantic relevance.
For each claim, establish that the observed source applies to the requested entity, time, environment,
and scope. A recent read of historical material does not establish current truth. Source titles, shared
names, matching words and the existence of a quotation are insufficient. Inspect the source's status,
dependencies and conflicting evidence when relevant. Cite enough passages to support all claimed details;
do not describe a short selected passage as containing unselected content. If applicability or coverage
is unestablished, record inconclusive and qualify the answer. Do not turn missing evidence into a pass.
Tests must assert behavior, not merely process survival or any response. Comparisons need the same inputs
and assertions on both implementations, with differences explained. Distinguish unique cases from repeats.
Use a failing negative control when assertion adequacy is uncertain; counts alone do not prove coverage.
Record failed/inconclusive checks; rerun after repairs. Commands default to
verification_effect=mutation. Observation leaves workspace/shared/system/external state unchanged;
disposable process-local objects and normal tool output do not count. Temporary shared-state changes
do. Mutations invalidate evidence. Use mutation if unsure. No permission is granted or read-only
execution enforced.
Saved-memory writes do not change workspace/system evidence. Checks citing recall must be recorded again
if the saved memories change. For memory tasks, use recall to verify the saved state directly.
Consult completion_actions before rerunning checks: record_verification means the required execution
already succeeded and is current. Read its saved result and record the criterion; do not rerun it just
to obtain a new evidence ID. Only run_check requires another execution. Genuine changes still need checks.
finish_task(complete) requires all criteria passed and an evidence-grounded answer. For partial/blocked
work give a concrete reason; do not pretend success or retry forever. Planning and verdict tools only
bookkeep; run_required_check executes the declared command.
"""
AUTHORITY_INSTRUCTIONS = """
Verification grants no permissions. Actual user requests/updates are acceptance authority.
Before substantial work, plan_task observable working criteria: general, research, system or coding.
Conversation needs no plan. Research covers source investigations/reproductions; coding covers software
changes/executable acceptance and needs shell/execute_program. Report missing capabilities.
Runtime outcome checks the actual request; host checks/runtime obligations cannot be weakened.
Model descriptions, predicted counts/results and required_check declarations are working hypotheses,
not requirements. Inspect interfaces first. plan_task adds but cannot rewrite.
Use revise_plan to replace/retire a mistaken working criterion with fresh exact evidence and a reason
preserving the actual request. It archives prior definitions/verdicts and clears affected/outcome passes;
it never certifies correctness. Omitted replacement fields stay. A completed failed command may explain
a method error, not prove a factual claim. Legacy ledgers remain immutable; report their limitations.
record_finding + plan_task {id,finding_id} tracks inquiries. Supported/refuted need current anchored
finding evidence; inconclusive remains unresolved and refutation alone does not establish delivery.
Preregister executable acceptance methods (code_behavior for coding), then run_required_check with an
explicit timeout. It executes the exact saved command through normal permissions. Exploratory commands
need no declaration. Use behavioral/integration/edge assertions, not mere imports/survival/counts.
Preserve user checks. Revised methods need new executions, never old polls. Suites must fail on any
failed assertion. Combine mutating checks or follow mutations with observational checks.
record_verification needs current successful observations and any saved required_check. Use short
evidence_selections {evidence_ref,passage_id,reasoning} from tool results/inspect_evidence, or
evidence_support {evidence_id,excerpt,reasoning}. Omitted evidence_ids are inferred; explicit lists must
cover support. Use observation IDs, never process IDs. Quote exact contiguous passages including
whitespace/line numbers; separate skipped passages. Each fresh citation needs criterion-specific support.
Inspection/selection never refreshes evidence. Old observations are context only. A quote establishes
attribution, not relevance: check source applicability, entity/time/environment/scope, dependencies,
conflicts and coverage. Missing or conflicting support means inconclusive, not passed. Silent exit_code=0
proves execution status only; explain assertions and limits briefly, then report fully in the answer.
Commands default to verification_effect=mutation. Observation leaves workspace/shared/system/external
state unchanged; disposable process-local objects and normal output do not count, temporary shared
changes do. Use mutation if unsure. Mutations invalidate evidence; labels do not enforce read-only execution.
Saved memory changes reopen checks citing recall; memory tasks verify with fresh recall. Record failures.
Follow completion_actions: record_verification means the required execution already succeeded/current;
read its saved result rather than rerunning for a new ID. Only run_check needs execution.
finish_task(complete) requires all active criteria and actual outcome passed with an evidence-grounded
answer. Partial/blocked needs a specific limitation; do not pretend success or retry forever.
"""
REQUESTED_OUTCOME_ID = "__requested_outcome__"
REQUESTED_OUTCOME = (
    "Verify and deliver the actual user-requested outcome, preserving all requirements and constraints. "
    "Resolving an investigation alone does not establish this outcome."
)
CODE_BEHAVIOR = "Demonstrate the requested software behavior with relevant executable checks."
MUTATIONS = {
    "write_file",
    "edit_file",
    "shell",
    "execute_program",
    "write_process",
    "stop_process",
}
MEMORY_WRITES = {"remember", "forget"}
EXECUTABLE_TOOLS = {"shell", "execute_program", "poll_process"}
EVIDENCE_TOOLS = {
    "read_file",
    "read_artifact",
    "list_directory",
    "search_files",
    "fetch_url",
    "shell",
    "execute_program",
    "poll_process",
    "recall",
}
SOURCE_SUFFIXES = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".go",
    ".rs",
    ".c",
    ".h",
    ".cpp",
    ".java",
    ".cs",
    ".rb",
    ".php",
    ".swift",
    ".kt",
    ".vue",
    ".svelte",
    ".sh",
    ".html",
    ".css",
    ".scss",
    ".sql",
    ".bash",
    ".ps1",
}


class Verification:
    def __init__(self, registry):
        self.registry = registry
        self.store, self.sid = registry.store, registry.sid
        self.state = self.empty()
        event = self.store.latest_event(self.sid, "verification_state")
        if event:
            self.state.update(event["body"])
            # Existing sessions retain their contract; begin() opts new tasks into anchoring.
            self.state["evidence_contract"] = event["body"].get("evidence_contract", "legacy")
            self.state["execution_contract"] = event["body"].get("execution_contract", "legacy")
            # Add memory dependencies to old records without refreshing their workspace revision
            # or changing their evidence/execution contract. Recall snapshots are durable evidence.
            evidence = None
            for check in self.state["checks"].values():
                if "memory_fingerprint" not in check:
                    if evidence is None:
                        evidence = self.evidence()
                    check["memory_fingerprint"] = self.recalled_memory(check["evidence_ids"], evidence)
        self.required_checks = RequiredChecks(self)
        self.references = EvidenceReferences(self)
        self.planning = PlanningAuthority(self)
        if any(pid not in registry.processes.jobs for pid in self.state["pending"]):
            latest = self.store.latest_event(self.sid)
            self.invalidate(latest["id"] + 1, "managed processes were lost on restart")
            self.save()

    @staticmethod
    def empty():
        return {
            "profiles": [],
            "criteria": {},
            "inquiries": {},
            "inquiry_outcome": False,
            "checks": {},
            "required": False,
            "revision": 0,
            "floor": 0,
            "floor_reason": "",
            "evidence_start": 0,
            "pending": [],
            "outcome": None,
            "reason": "",
            "children": {},
            "process_revisions": {},
            "process_effects": {},
            "evidence_contract": "anchored",
            "execution_contract": "declared",
            "required_checks": {},
            "final_answer": None,
            "pending_answer": None,
            "authority": None,
            "criterion_origins": {},
            "plan_history": [],
            "plan_history_omitted": 0,
            "retired_criteria": {},
            "plan_revision_event_id": None,
            "criterion_revision_events": {},
        }

    def save(self):
        self.store.event(self.sid, "verification_state", self.state)

    def begin(self, *, authority_request=False, acceptance_checks=None):
        records = self.validate_acceptance_checks(acceptance_checks)
        if records and not authority_request:
            raise ValueError("Host checks require an actual request authority contract")
        if authority_request:
            self.planning.request()
        self.state = self.empty()
        event = self.store.latest_event(self.sid)
        self.state["evidence_start"] = event["id"] if event else 0
        # Existing children belong to prior requests, not this task.
        self.state["children"] = {r["child_id"]: r["status"] for r in self.store.subagents(self.sid)}
        if authority_request:
            self.planning.bind(records)
        self.planning.install()
        self.save()

    def validate_acceptance_checks(self, records):
        return self.planning.validate_host(records)

    def sync_authority(self):
        return self.planning.sync()

    def authority_status(self, *, include_request=False):
        return self.planning.status(include_request=include_request)

    def criterion_authority(self, criterion_id):
        return self.planning.criterion(criterion_id)

    def instructions(self):
        if not self.state.get("authority"):
            return INSTRUCTIONS
        return AUTHORITY_INSTRUCTIONS

    def install(self):
        from .tools import string

        self.registry.add(
            "plan_task",
            "Add criteria (id, description, optional required_check) or inquiries (id, finding_id). "
            "New request plans are working proposals; use revise_plan for changes. Legacy criteria "
            "stay immutable. Attaching a check may omit description.",
            {
                "profiles": {
                    "type": "array",
                    "items": {"enum": list(PROFILES)},
                    "minItems": 1,
                    "maxItems": 4,
                    "uniqueItems": True,
                },
                "criteria": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 20,
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": string("Stable criterion ID", 60),
                            "description": string(
                                "Observable required outcome; required for new criteria. Omit for an existing "
                                "criterion to preserve its immutable description",
                                500,
                            ),
                            "required_check": RequiredChecks.schema(),
                            "finding_id": string(
                                "Current-request Finding; no other criterion fields",
                                80,
                            ),
                        },
                        "required": ["id"],
                        "additionalProperties": False,
                    },
                },
            },
            ["profiles", "criteria"],
            "read",
            self.plan,
        )
        self.registry.add(
            "record_verification",
            "Record a criterion's verdict using evidence_selections (reference, passage, reasoning) or "
            "evidence_support. Omitted evidence_ids are inferred; explicit IDs must cover support. Pass "
            "needs fresh successful evidence and the declared required_check. Older IDs are context only. "
            "Runtime checks attribution and command identity; you judge semantic relevance.",
            {
                "criterion_id": string("Criterion ID from plan_task or task_status", 60),
                "verdict": {"enum": ["passed", "failed", "inconclusive"]},
                "evidence_ids": {
                    "type": "array",
                    "items": string("Exact evidence_id from an observation result; never process_id", 100),
                    "maxItems": 20,
                    "uniqueItems": True,
                },
                "resolved_process_ids": {
                    "type": "array",
                    "items": string("Lost handle inspected after restart", 100),
                    "maxItems": 8,
                    "uniqueItems": True,
                },
                "explanation": string(
                    "Brief observed outcome and limits, at most 1500 characters. Put the full report in "
                    "the final answer",
                    1500,
                ),
                "evidence_support": {
                    "type": "array",
                    "maxItems": 20,
                    "items": {
                        "type": "object",
                        "properties": {
                            "evidence_id": string("Fresh evidence ID cited by this check", 100),
                            "excerpt": string(
                                "Exact contiguous quote, including whitespace/line numbers. Use separate "
                                "entries for separate passages. exit_code=0 establishes execution status only",
                                1000,
                            ),
                            "reasoning": string(
                                "Why this specific observation establishes this criterion, and its limits",
                                1500,
                            ),
                        },
                        "required": ["evidence_id", "excerpt", "reasoning"],
                        "additionalProperties": False,
                    },
                },
                "evidence_selections": {
                    "type": "array",
                    "maxItems": 20,
                    "items": {
                        "type": "object",
                        "properties": {
                            "evidence_ref": string("Short observation reference, such as E17", 40),
                            "passage_id": string("Exact displayed passage label, such as P2", 40),
                            "reasoning": string(
                                "Why this passage establishes this criterion, and its limits", 1500
                            ),
                        },
                        "required": ["evidence_ref", "passage_id", "reasoning"],
                        "additionalProperties": False,
                    },
                },
            },
            ["criterion_id", "verdict", "explanation"],
            "read",
            self.record,
        )
        self.registry.add(
            "task_status",
            "Read durable criteria, checks, evidence references, eligibility and completion gaps.",
            {},
            [],
            "read",
            self.status_tool,
        )
        self.registry.add(
            "inspect_evidence",
            "Read a page of exact labeled passages from a current task observation. "
            "Use its evidence_ref and passage_id in record_verification.evidence_selections. "
            "This reads the original journal; it never refreshes evidence or counts as a new observation.",
            {
                "evidence_ref": string("Short reference from an observation or task_status", 40),
                "page": {"type": "integer", "minimum": 1},
            },
            ["evidence_ref"],
            "read",
            self.inspect_evidence,
        )
        self.registry.add(
            "finish_task",
            "Propose complete, partial or blocked based on recorded checks. "
            "Completion is invalidated by later changes. If a prior answer was withheld, complete "
            "requires answer containing the current substantive deliverable; the runtime delivers it. "
            "Running subagents are collected first; queued ones that have not started are cancelled as "
            "superseded, so wait for queued work the answer depends on before finishing.",
            {
                "outcome": {"enum": ["complete", "partial", "blocked"]},
                "reason": string("Required explanation for partial or blocked work", 1500),
                "answer": string(
                    "Current substantive final response; required after a withheld answer", 64000
                ),
            },
            ["outcome"],
            "read",
            self.finish,
        )
        if self.store.session(self.sid)["parent_id"] is not None:
            self.registry.add(
                "submit_result",
                "Finish this child assignment and durably deliver its report in one operation. "
                "Complete requires current passing verification; partial/blocked requires a reason. "
                "The runtime saves read-only reports without workspace write permission. "
                "No additional final answer or tool call is needed after success.",
                {
                    "outcome": {"enum": ["complete", "partial", "blocked"]},
                    "report": string("Concise findings, changes, verification and remaining work", 16000),
                    "evidence_ids": {
                        "type": "array",
                        "items": string("Actual local observation evidence_id", 100),
                        "maxItems": 40,
                        "uniqueItems": True,
                    },
                    "uncertainties": {
                        "type": "array",
                        "items": string("Uncertainty, limitation or unresolved question", 1000),
                        "maxItems": 12,
                    },
                    "reason": string("Required concrete explanation for partial or blocked work", 1500),
                },
                ["outcome", "report", "evidence_ids", "uncertainties"],
                "read",
                self.submit_result,
            )
        self.planning.install()

    def require(self):
        self.state["required"] = True
        if self.state.get("authority"):
            self.state["criteria"].setdefault("outcome", OUTCOME)
            self.state["criterion_origins"]["outcome"] = "request"
        elif not self.state["criteria"]:
            self.state["criteria"]["outcome"] = (
                "Verify the requested result against observed evidence. This is an acceptance criterion; "
                "recording it does not declare completion. Call finish_task afterwards."
            )
        if not self.state["profiles"]:
            self.state["profiles"] = ["general"]

    def arm_plan(self):
        """A permitted planning attempt is substantial work, even if its plan is invalid.

        Install only the actual-request obligation before validating model proposals.
        Rejected amendments cannot reopen existing passes or change plan definitions.
        """
        if not self.state.get("authority") or self.state["required"]:
            return
        self.sync_authority()
        self.require()
        self.save()

    def invalidate(self, floor, reason="a shell command or write ran"):
        self.require()
        self.state["revision"] += 1
        self.state["floor"] = floor
        # Named in stale-evidence rejections so the model knows what moved the floor.
        self.state["floor_reason"] = reason
        self.state["outcome"] = None
        self.state["reason"] = ""
        self.state["final_answer"] = None

    def mutates(self, name, args):
        if name not in MUTATIONS:
            return False
        if name in {"shell", "execute_program"}:
            return args.get("verification_effect", "mutation") != "observation"
        if name in {"stop_process", "write_process"}:
            job = self.registry.processes.jobs.get(args.get("process_id"))
            # A finished/unknown handle cannot affect a process, or refresh its provenance.
            return bool(job and not job.waiter.done())
        return True

    def started(self, name, args, event_id):
        # Persist before awaiting a mutation: cancellation may prevent any result.
        if self.mutates(name, args):
            self.invalidate(event_id)
            if name in {"write_process", "stop_process"}:
                self.state["process_effects"][args["process_id"]] = "mutation"
            if name in {"write_file", "edit_file"} and Path(args.get("path", "")).suffix in SOURCE_SUFFIXES:
                if "coding" not in self.state["profiles"]:
                    self.state["profiles"].append("coding")
                self.state["criteria"].setdefault(
                    "code_behavior",
                    CODE_BEHAVIOR,
                )
                if self.state.get("authority"):
                    self.state["criterion_origins"]["code_behavior"] = "runtime"
            self.save()
        elif name in MEMORY_WRITES:
            # A memory-only task still needs verification, but incidental memory saving
            # must not erase an existing task's passing workspace/system/source checks.
            if not self.state["required"]:
                self.require()
                self.save()
        elif name in {"fetch_url", "shell", "execute_program"}:
            self.require()
            self.state["outcome"] = None
            self.save()
        return self.state["revision"]

    def observe(self, name, result, event_id, args=None, started_revision=None):
        # Rejected calls do not mutate; failed executed commands may have partial effects.
        changed = False
        pid = result.get("process_id")
        process_revisions = self.state.setdefault("process_revisions", {})
        process_effects = self.state.setdefault("process_effects", {})
        current_process = pid and process_revisions.get(pid) == self.state["revision"]
        args = args or {}
        effect = args.get("verification_effect", "mutation")
        launch = name in {"shell", "execute_program"}
        if name in MEMORY_WRITES and any(
            self.memory_changed(check) for check in self.state["checks"].values()
        ):
            self.state["outcome"] = None
            self.state["reason"] = ""
            self.state["final_answer"] = None
            changed = True
        # Launch provenance is fixed here and never re-anchored by stop/write on an old handle.
        # A concurrent mutation during the awaited launch also makes that launch stale.
        uninterrupted = started_revision == self.state["revision"]
        if (
            name in MUTATIONS
            and (not launch or effect == "mutation")
            and name not in {"stop_process", "write_process"}
        ):
            self.invalidate(event_id)
            changed = True
        if launch and pid:
            process_revisions[pid] = self.state["revision"] if uninterrupted else started_revision
            process_effects[pid] = effect
            current_process = uninterrupted
            changed = True
        if pid and result.get("running") and pid not in self.state["pending"]:
            self.state["pending"].append(pid)
            changed = True
        elif pid and result.get("running") is False and pid in self.state["pending"]:
            self.state["pending"].remove(pid)
            if process_effects.get(pid, "mutation") != "observation":
                self.invalidate(event_id, "a managed process finished")
                if current_process:
                    process_revisions[pid] = self.state["revision"]
            changed = True
        if changed:
            self.save()

    def sync_children(self):
        changed = False
        for row in self.store.subagents(self.sid):
            cid, status = row["child_id"], row["status"]
            if self.state["children"].get(cid) == status:
                continue
            self.state["children"][cid] = status
            result = row.get("result") or {}
            child_verification = result.get("verification")
            if child_verification is None:
                event = self.store.latest_event(cid, "verification_state")
                child_verification = event["body"] if event else {}
            # Observe-mode children cannot change the workspace. Their own verification
            # requirements do not make the parent's earlier observations stale.
            read_only = self.store.session(cid)["mode"] == "observe"
            # The parent must verify integration after delegated changes, including partial changes.
            # A known zero revision has no environment mutations (it may only have saved memories).
            # Legacy reports without a revision remain conservative.
            if (
                status not in {"running", "queued"}
                and not read_only
                and child_verification.get("required")
                and child_verification.get("revision", 1) > 0
            ):
                event = self.store.latest_event(self.sid)
                self.invalidate(
                    event["id"] + 1 if event else 0, "a delegated child that could make changes finished"
                )
            changed = True
        if changed:
            self.save()

    def evidence(self):
        return {
            e["body"]["call_id"]: e
            for e in self.store.events(
                self.sid, after=self.state["evidence_start"], kinds=("tool_finished", "tool_error")
            )
            if e["body"].get("name") in EVIDENCE_TOOLS
        }

    def annotate_result(self, name, result, event_id):
        """Add bounded exact passages after journaling the original observation payload."""
        if name in EVIDENCE_TOOLS:
            event = {"id": event_id, "body": {"name": name, "result": result}}
            result.update(self.references.summary(event))

    async def inspect_evidence(self, evidence_ref, page=1):
        self.sync_children()
        return self.references.inspect(evidence_ref, page)

    @staticmethod
    def memory_fingerprint(memories):
        return hashlib.sha256(json.dumps(memories, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def recalled_memory(self, evidence_ids, evidence):
        snapshots = [
            evidence[cid]["body"]["result"].get("memories", {})
            for cid in evidence_ids
            if cid in evidence and evidence[cid]["body"]["name"] == "recall"
        ]
        return self.memory_fingerprint(snapshots[-1]) if snapshots else None

    def memory_changed(self, check):
        fingerprint = check.get("memory_fingerprint")
        metadata = check.get("memory_metadata_fingerprint")
        return (
            fingerprint is not None and fingerprint != self.memory_fingerprint(self.store.memories())
        ) or (metadata is not None and metadata != self.memory_fingerprint(self.store.memory_records()))

    def recalled_memory_metadata(self, evidence_ids, evidence):
        snapshots = [
            evidence[cid]["body"]["result"]["metadata"]
            for cid in evidence_ids
            if evidence[cid]["body"]["name"] == "recall" and "metadata" in evidence[cid]["body"]["result"]
        ]
        return self.memory_fingerprint(snapshots[-1]) if snapshots else None

    def execution_available(self):
        return bool(
            {"shell", "execute_program"} & {schema["function"]["name"] for schema in self.registry.schemas()}
        )

    async def plan(self, profiles, criteria):
        self.arm_plan()
        self.sync_authority()
        if "coding" in profiles and not self.execution_available():
            raise ValueError(
                "The coding profile requires executable checks, but shell is unavailable and execute_program "
                "is unavailable in this "
                "agent's capabilities. Use research or general for static code/source investigation. "
                "For software changes or executable checks, report the capability mismatch to the "
                "parent or finish partial/blocked with the limitation. No proposed criteria were added."
            )
        # Attaching a declaration must not require the caller to reproduce an implicit
        # criterion's exact wording. Normalize a copy; validate the whole plan before
        # committing anything, and never reinterpret an explicit description change.
        normalized = []
        inquiries = {}
        finding_records = (
            self.registry.work_notes.findings.records() if any("finding_id" in c for c in criteria) else {}
        )
        for c in criteria:
            cid = c["id"]
            if self.state.get("authority"):
                origin = self.criterion_authority(cid)["origin"]
                if cid == "code_behavior":
                    if "coding" not in {*profiles, *self.state["profiles"]}:
                        raise ValueError("code_behavior is reserved for the coding profile")
                    if "finding_id" in c or c.get("description", CODE_BEHAVIOR) != CODE_BEHAVIOR:
                        raise ValueError(
                            "code_behavior is a runtime obligation. Omit its description and attach a "
                            "working required_check; use another ID for model predictions."
                        )
                    c = {**c, "description": CODE_BEHAVIOR}
                if cid == "outcome" or origin in {"host", "request"}:
                    existing = self.state["criteria"].get(cid, OUTCOME)
                    if set(c) - {"id", "description"} or c.get("description", existing) != existing:
                        raise ValueError("Actual request and host obligations cannot be changed by plan_task")
                    c = {"id": cid, "description": existing}
                if cid in self.state["retired_criteria"]:
                    raise ValueError("Retired criterion IDs remain archived; use a new working criterion ID")
            if "finding_id" in c:
                if "description" in c or "required_check" in c:
                    raise ValueError("Inquiry links cannot include description or required_check")
                if cid in {REQUESTED_OUTCOME_ID, "code_behavior", "outcome"}:
                    raise ValueError("Requested outcomes and code behavior cannot become inquiries")
                previous = self.state["inquiries"].get(cid)
                if cid in self.state["criteria"] and previous is None:
                    raise ValueError("Existing strict criteria cannot become inquiries")
                finding = finding_records.get(c["finding_id"])
                if finding is None:
                    raise ValueError("Inquiry needs an existing current-request finding_id")
                link = {key: finding[key] for key in ("request_id", "finding_id", "claim")}
                if previous is not None and previous != link:
                    raise ValueError("An inquiry's finding identity cannot change")
                inquiries[cid] = link
                normalized.append(
                    {
                        "id": cid,
                        "description": "Resolve this investigation from applicable evidence, whether supported "
                        "or refuted: " + finding["claim"],
                    }
                )
                continue
            if cid in self.state["inquiries"]:
                raise ValueError(
                    "Existing inquiries retain their finding_id and cannot become strict criteria"
                )
            if cid == REQUESTED_OUTCOME_ID:
                raise ValueError(
                    "The runtime user-outcome obligation cannot be declared or given a required_check"
                )
            if "description" not in c:
                existing = self.state["criteria"].get(c["id"])
                if existing is None:
                    raise ValueError(
                        "New criteria require a description; omit description only for an existing "
                        "criterion ID from task_status. No proposed criteria were added."
                    )
                c = {**c, "description": existing}
            normalized.append(c)
        criteria = normalized
        if any(not c["id"].strip() or not c["description"].strip() for c in criteria):
            raise ValueError("Criteria require nonempty IDs and descriptions")
        if len({c["id"] for c in criteria}) != len(criteria):
            raise ValueError("Criterion IDs must be unique")
        for c in criteria:
            existing = self.state["criteria"].get(c["id"])
            if existing and existing != c["description"]:
                if self.state.get("authority"):
                    raise ValueError(
                        "Use revise_plan with fresh observed evidence to change a working criterion"
                    )
                raise ValueError(
                    "Existing criteria cannot be rewritten. To attach required_check, supply the existing "
                    "criterion id and omit description, or copy its exact description from task_status. "
                    "Add a new criterion for an additional requirement. No proposed criteria were added."
                )
        criterion_ids = set(self.state["criteria"]) | {c["id"] for c in criteria}
        if self.state.get("authority"):
            criterion_ids.add("outcome")
        if inquiries:
            if REQUESTED_OUTCOME_ID in self.state["criteria"] and not self.state["inquiry_outcome"]:
                raise ValueError("Existing requested_outcome ID conflicts with the runtime obligation")
            criterion_ids.add("outcome" if self.state.get("authority") else REQUESTED_OUTCOME_ID)
        if "coding" in profiles:
            criterion_ids.add("code_behavior")
        if len(criterion_ids) > 20:
            raise ValueError("At most 20 criteria per task; group related outcomes")
        # Validate every declaration before committing any part of a plan.
        additions = self.required_checks.prepare(criteria)
        if self.state.get("authority"):
            self.require()
        else:
            self.state["required"] = True
        self.state["profiles"] = sorted(set(self.state["profiles"]) | set(profiles))
        self.state["criteria"].update({c["id"]: c["description"] for c in criteria})
        if self.state.get("authority"):
            for c in criteria:
                self.state["criterion_origins"].setdefault(
                    c["id"], "runtime" if c["id"] == "code_behavior" else "working"
                )
        if inquiries:
            self.state["inquiries"].update(inquiries)
            if not self.state.get("authority"):
                self.state["criteria"].setdefault(REQUESTED_OUTCOME_ID, REQUESTED_OUTCOME)
            self.state["inquiry_outcome"] = True
        if "coding" in profiles:
            self.state["criteria"].setdefault(
                "code_behavior",
                CODE_BEHAVIOR,
            )
            if self.state.get("authority"):
                self.state["criterion_origins"]["code_behavior"] = "runtime"
        for cid, definition in additions.items():
            self.state["required_checks"][cid] = definition
            self.state["checks"].pop(cid, None)
        self.state["outcome"] = None
        self.state["final_answer"] = None
        self.save()
        return self.acknowledgement(criterion_ids=[c["id"] for c in criteria])

    async def record(
        self,
        criterion_id,
        verdict,
        evidence_ids=None,
        explanation="",
        resolved_process_ids=None,
        evidence_support=None,
        evidence_selections=None,
    ):
        self.sync_authority()
        self.sync_children()
        if criterion_id not in self.state["criteria"]:
            raise ValueError("Unknown criterion; use plan_task or task_status")
        if not explanation.strip():
            raise ValueError("Explain the observed outcome and any limitations")
        evidence = self.evidence()
        selected = self.references.resolve(evidence_selections or [], evidence)
        if evidence_ids is None:
            # Literal support already names its observations, just like selections. Preserve
            # explicit lists (including []) so contradictory references still fail below.
            evidence_ids = [support.get("evidence_id") for support in evidence_support or []]
        evidence_ids = list(dict.fromkeys([*(evidence_ids or []), *(s["evidence_id"] for s in selected)]))
        if len(evidence_ids) > 20:
            raise ValueError("At most 20 distinct observations may support a criterion")
        evidence_support = [*(evidence_support or []), *selected]
        self.validate_evidence_ids(evidence_ids, evidence)
        if any(support.get("evidence_id") not in evidence_ids for support in evidence_support or []):
            raise ValueError("evidence_support must refer to evidence_ids cited by this criterion")
        resolved = set(resolved_process_ids or [])
        if resolved and (
            verdict != "passed"
            or not resolved <= set(self.state["pending"])
            or any(pid in self.registry.processes.jobs for pid in resolved)
        ):
            raise ValueError("Only lost handles can be reconciled; poll known handles to completion")
        stale = []
        required_result = None
        inquiry = None
        if verdict == "passed":
            if not evidence_ids or set(self.state["pending"]) - resolved:
                raise ValueError("Pass requires evidence and no unfinished managed commands")
            floor = self.state["floor"]
            executable = bool(
                ("coding" in self.state["profiles"] and criterion_id == "code_behavior") or resolved
            )
            stale = [cid for cid in evidence_ids if evidence[cid]["id"] < floor]
            fresh = [cid for cid in evidence_ids if cid not in stale]
            if not fresh:
                # The floor rises after any command, write, finished child or restart. A pass must
                # cite at least one observation made after it; older citations are context only.
                usable = [
                    cid
                    for cid, e in evidence.items()
                    if e["id"] >= floor
                    and self.support_error(e) is None
                    and (not executable or e["body"]["name"] in EXECUTABLE_TOOLS)
                ][-4:]
                cause = self.state.get("floor_reason") or "a possible change was recorded"
                detail = (
                    f"Evidence predates the latest possible change (journal event {floor}: {cause}): "
                    + ", ".join(stale)
                    + ". Commands default to possible mutations; explicit observations preserve "
                    "freshness only until a mutation, changed child or restart. Rerun the check and cite that "
                    "evidence_id" + (f"; fresh usable evidence: {', '.join(usable)}" if usable else "") + "."
                )
                raise VerificationBlocked(detail, self.stale_evidence_blocker(criterion_id, evidence, stale))
            # Stale ids carry no weight, so only the fresh ids must be sound.
            for cid in fresh:
                problem = self.support_error(evidence[cid])
                if problem:
                    raise ValueError(f"{cid}: {problem}")
            if executable and not any(evidence[cid]["body"]["name"] in EXECUTABLE_TOOLS for cid in fresh):
                raise ValueError("Software behavior / lost-process inspection requires executable evidence")
            if self.state["evidence_contract"] == "anchored":
                self.validate_support(fresh, evidence, evidence_support or [])
            if criterion_id in self.state["inquiries"]:
                problem, inquiry = self.inquiry_resolution(criterion_id)
                if problem:
                    raise ValueError(problem)
                if not set(fresh) & set(inquiry["evidence_ids"]):
                    raise ValueError("Inquiry verification must cite fresh evidence from its linked finding")
                # New inquiry semantics always require anchoring, even in a restored legacy ledger.
                if self.state["evidence_contract"] != "anchored":
                    self.validate_support(fresh, evidence, evidence_support or [])
            required_result = self.required_checks.status().get(criterion_id)
            if required_result and not required_result["satisfied"]:
                return {
                    "ok": False,
                    "error": "ValueError",
                    "detail": required_result["reason"],
                    "verification_blocker": self.required_check_blocker(criterion_id, required_result),
                    "recovery": self.required_check_recovery(),
                    "completion_actions": self.completion_actions([criterion_id]),
                }
        self.state["pending"] = [pid for pid in self.state["pending"] if pid not in resolved]
        check = {
            "verdict": verdict,
            "evidence_ids": [cid for cid in evidence_ids if cid not in stale],
            "explanation": explanation,
            "revision": self.state["revision"],
            "memory_fingerprint": self.recalled_memory(
                [cid for cid in evidence_ids if cid not in stale], evidence
            ),
            "memory_metadata_fingerprint": self.recalled_memory_metadata(
                [cid for cid in evidence_ids if cid not in stale], evidence
            ),
        }
        if evidence_support:
            check["evidence_support"] = evidence_support
        if stale:
            # Context for readers; the pass rests on the fresh evidence above.
            check["stale_evidence_ids"] = stale
        if required_result:
            check["required_check_evidence"] = required_result["evidence_id"]
        if inquiry is not None:
            check["inquiry_resolution"] = self.inquiry_identity(inquiry)
        self.state["checks"][criterion_id] = check
        self.state["outcome"] = None
        self.save()
        return self.acknowledgement(
            checks={
                criterion_id: {
                    key: value
                    for key, value in check.items()
                    if key not in {"explanation", "evidence_support"}
                }
            }
        )

    def validate_evidence_ids(self, evidence_ids, evidence):
        unknown = [cid for cid in evidence_ids if cid not in evidence]
        if not unknown:
            return
        # Diagnose the common process_id/call_id confusion without accepting aliases:
        # several polls can observe one process, and each has its own output and eligibility.
        matches = {}
        for cid in unknown[:4]:
            ids = [
                eid
                for eid, event in evidence.items()
                if (pid := event["body"]["result"].get("process_id")) and cid in {pid, "call_" + pid}
            ]
            if ids:
                matches[cid] = ids[-3:]
        detail = (
            "Evidence must reference this task's actual observation tool results. Unknown evidence_ids: "
            + json.dumps(unknown[:4], ensure_ascii=True)[:500]
            + (f" ({len(unknown)} unknown in total)" if len(unknown) > 4 else "")
            + ". Copy evidence_id from an observation result; process_id is a command handle. "
            "Do not construct evidence IDs by adding call_. "
        )
        if matches:
            detail += (
                "Observations for those process handles (check output and eligibility before choosing): "
                + json.dumps(matches, ensure_ascii=True)[:850]
                + ". "
            )
        detail += (
            "Use task_status for recent evidence IDs, previews and eligibility. "
            "The evidence_id on this error identifies the rejected call, not an observation to cite."
        )
        raise ValueError(detail)

    @staticmethod
    def evidence_source(event):
        """Observable payload for exact source anchoring, excluding bookkeeping/provenance.

        Execution status is intentionally labeled narrowly. A caller can still misinterpret a
        real excerpt; this mechanism establishes source attribution, not semantic entailment.
        """
        result = event["body"]["result"]
        if event["body"]["name"] in EXECUTABLE_TOOLS:
            return str(result.get("output") or "") + (
                f"\nexit_code={result['exit_code']} (execution status only)"
                if result.get("exit_code") is not None
                else ""
            )
        metadata = {
            "ok",
            "evidence_id",
            "evidence_ref",
            "evidence_passages",
            "passage_count",
            "saved_output",
            "source_archive",
            "output_truncated",
            "truncated",
            "sha256",
            "error",
            "status",
            "next_offset",
            "total_chars",
        }
        payload = {key: value for key, value in result.items() if key not in metadata}
        # String values retain literal newlines instead of only their JSON escapes.
        return "\n".join(str(value) for value in payload.values() if isinstance(value, str)) + (
            "\n" + json.dumps(payload, ensure_ascii=False, sort_keys=True)
        )

    @staticmethod
    def observed_text(event):
        """Prefer the displayed text over the metadata included in the full evidence payload."""
        result = event["body"]["result"]
        for key in ("output", "content", "text"):
            if isinstance(result.get(key), str) and result[key]:
                return result[key]
        return Verification.evidence_source(event)

    @staticmethod
    def support_excerpt_sample(event, excerpt):
        """Bounded literal context for repairing a quote, never a substitute for validation."""
        observed = Verification.observed_text(event)
        start = 0
        for line in excerpt.splitlines():
            if line.strip() and (offset := observed.find(line.strip())) >= 0:
                # Show the actual line prefix/indentation even if the proposal dropped it.
                start = observed.rfind("\n", 0, offset) + 1
                break
        # JSON quoting keeps newlines unambiguous; even escaped control characters
        # fit within ToolRegistry's 2,000-character error detail limit.
        return json.dumps(observed[start : start + 160], ensure_ascii=False)

    def validate_support(self, fresh, evidence, supports):
        supported_ids = set()
        for support in supports:
            cid = support.get("evidence_id")
            if cid not in evidence:
                raise ValueError("evidence_support must reference actual observation evidence")
            supported_ids.add(cid)
        missing = [cid for cid in fresh if cid not in supported_ids]
        if missing:
            raise ValueError(
                "Pass requires evidence_support for each fresh ID: " + ", ".join(missing) + ". "
                "Provide {evidence_id, excerpt, reasoning}: quote one contiguous passage exactly and explain "
                "why it establishes this criterion. Use separate support entries with the same evidence_id "
                "for separate passages. For empty command output, exit_code=0 supports "
                "execution status only; explain the actual checked assertions."
            )
        for support in supports:
            cid = support["evidence_id"]
            excerpt, reasoning = support.get("excerpt", ""), support.get("reasoning", "")
            source = self.evidence_source(evidence[cid])
            if not excerpt.strip() or excerpt not in source:
                sample = self.support_excerpt_sample(evidence[cid], excerpt)
                matches = [
                    eid
                    + (
                        " (eligible observation; judge relevance)"
                        if event["id"] >= self.state["floor"] and self.support_error(event) is None
                        else " (ineligible for pass)"
                    )
                    for eid, event in evidence.items()
                    if eid != cid and len(excerpt.strip()) >= 12 and excerpt in self.evidence_source(event)
                ][-3:]
                attribution = (
                    "This exact excerpt occurs in other observations: " + ", ".join(matches)[:300] + ". "
                    if matches
                    else ""
                )
                raise ValueError(
                    f"{cid}: evidence_support excerpt must quote actual observed content. "
                    "Copy one contiguous passage exactly, preserving whitespace and read_file line numbers "
                    "when spanning lines. Do not join skipped or reordered lines: use separate "
                    "evidence_support entries with the same evidence_id for separate passages. "
                    "Put summaries and interpretation in reasoning. Correct the quote using the original "
                    "result or this sample; source matching alone does not establish the criterion. "
                    + attribution
                    + "Observed content sample (JSON string): "
                    + sample
                )
            if not reasoning.strip():
                raise ValueError(f"{cid}: explain how this observation supports this specific criterion")

    def support_error(self, event):
        """Why a fresh observation cannot support a pass, or None when it can."""
        result = event["body"]["result"]
        if event["body"]["name"] == "recall" and (
            result.get("memories") != self.store.memories()
            or ("metadata" in result and result["metadata"] != self.store.memory_records())
        ):
            return (
                "Saved memories changed; use recall for current memory evidence and record this check again"
            )
        if (
            not result.get("ok")
            or result.get("running")
            or result.get("timed_out")
            or result.get("stopped")
            or result.get("exit_code", 0) != 0
        ):
            return "Failed, running, stopped or timed-out results cannot support a pass"
        pid = result.get("process_id")
        if pid and self.state.get("process_revisions", {}).get(pid) != self.state["revision"]:
            return "Polling an older process does not make its execution evidence fresh"
        return None

    @staticmethod
    def inquiry_identity(finding):
        return {key: finding[key] for key in ("request_id", "finding_id", "revision", "status")}

    def inquiry_status(self, criterion_id):
        """Bounded interpretation provenance, never an independent correctness verdict."""
        link = self.state["inquiries"][criterion_id]
        findings = self.registry.work_notes.findings
        record = findings.records().get(link["finding_id"])
        if record is None or any(record[key] != link[key] for key in link):
            return {
                **link,
                "status": "unavailable",
                "revision": None,
                "needs_recheck": True,
                "dependency_problems": ["Finding is unavailable in this request"],
                "evidence_ids": [],
                "test_validity": "unknown",
                "interpretation_verified": False,
            }
        record = findings.snapshot(finding_ids=[link["finding_id"]])["findings"][0]
        result = {
            key: record[key]
            for key in (
                "request_id",
                "finding_id",
                "claim",
                "status",
                "revision",
                "needs_recheck",
                "dependency_problems",
                "evidence_ids",
                "test_validity",
                "interpretation_verified",
            )
        }
        problems = result["dependency_problems"]
        result["dependency_problems"] = [problem[:300] for problem in problems[:8]]
        result["omitted_dependency_problems"] = max(0, len(problems) - 8)
        return result

    def inquiry_resolution(self, criterion_id):
        record = self.inquiry_status(criterion_id)
        if record["status"] == "unavailable":
            return "Inquiry finding is unavailable in this request; do not substitute another claim", record
        if record["status"] not in {"supported", "refuted"} or record["test_validity"] != "valid":
            return (
                "Resolve the inquiry with valid supported/refuted finding evidence; inconclusive is unresolved",
                record,
            )
        if record["needs_recheck"]:
            return "Recheck the inquiry finding's changed or unsupported dependencies", record
        return None, record

    def inquiry_changed(self, criterion_id, check):
        if criterion_id not in self.state["inquiries"]:
            return False
        problem, finding = self.inquiry_resolution(criterion_id)
        return bool(problem) or check.get("inquiry_resolution") != self.inquiry_identity(finding)

    def gaps(self, required=None):
        self.sync_authority()
        gaps = []
        if required is None:
            required = self.required_checks.status()
        for cid in self.state["criteria"]:
            check = self.state["checks"].get(cid)
            if (
                not check
                or check["verdict"] != "passed"
                or check["revision"] != self.state["revision"]
                or self.memory_changed(check)
                or self.inquiry_changed(cid, check)
                or (
                    cid in required
                    and (
                        not required[cid]["satisfied"]
                        or check.get("required_check_evidence") != required[cid]["evidence_id"]
                    )
                )
            ):
                gaps.append(cid)
        if self.state["pending"]:
            gaps.append("unfinished_commands")
        return gaps

    async def finish(self, outcome, reason="", answer=""):
        self.sync_children()
        if outcome == "complete" and self.gaps():
            # Preserve precise next steps in the tool result instead of a generic error
            # that encourages rerunning every check (including already-current ones).
            return {
                "ok": False,
                "error": "ValueError",
                "detail": "Completion requires current passing checks: " + ", ".join(self.gaps()),
                "verification_blockers": self.required_check_blockers(),
                "recovery": self.required_check_recovery(),
                **self.completion_status(),
            }
        if outcome != "complete" and not reason.strip():
            raise ValueError("Partial/blocked outcomes require a concrete reason")
        if outcome == "complete" and self.state.get("pending_answer") and not answer.strip():
            raise ValueError(
                "A substantive answer was withheld before verification completed. Supply answer with "
                "the current full deliverable, updated for these checks; completion bookkeeping alone "
                "does not deliver the answer."
            )
        self.state["outcome"], self.state["reason"] = outcome, reason
        self.state["final_answer"] = answer if answer.strip() else None
        self.save()
        return self.acknowledgement(final_answer_staged=bool(self.state["final_answer"]))

    def submitted_result(self):
        return self.store.submitted_result(self.sid)

    async def submit_result(self, outcome, report, evidence_ids, uncertainties, reason=""):
        if self.store.session(self.sid)["parent_id"] is None:
            raise ValueError("Only child assignments can submit a result")
        if self.submitted_result():
            raise ValueError("This child already submitted its result")
        if not report.strip():
            raise ValueError("Submission requires a nonempty report")
        self.sync_children()
        if outcome == "complete":
            missing_plan = not self.state["required"] or not self.state["criteria"]
            gaps = self.gaps()
            if missing_plan or gaps:
                return {
                    "ok": False,
                    "error": "ValueError",
                    "detail": "Complete submission requires declared criteria and current passing checks: "
                    + ", ".join((["verification plan"] if missing_plan else []) + gaps),
                    "verification_blockers": self.required_check_blockers(),
                    "recovery": (
                        "Static research can cite source observations without executable required_check. "
                        "Alternatively, deliver a plain report with verification limits; that does not "
                        "certify a verified submission."
                        if missing_plan
                        else self.required_check_recovery()
                    ),
                    **self.completion_status(gaps, require_plan=True),
                }
        elif outcome not in {"partial", "blocked"} or not reason.strip():
            raise ValueError("Partial/blocked submissions require a concrete reason")
        evidence = self.evidence()
        if any(cid not in evidence for cid in evidence_ids):
            raise ValueError("Submission evidence must reference this task's actual observation results")
        contract_event = self.store.latest_event(self.sid, "assignment_contract")
        deliverable = contract_event["body"].get("deliverable", {}) if contract_event else {}
        artifacts = []
        if deliverable.get("kind") == "workspace_artifact":
            path = self.registry.resolve(deliverable["path"])
            if outcome == "complete" and not path.is_file():
                raise ValueError("The declared workspace artifact does not exist as a regular file")
            if path.is_file():
                stat = path.stat()
                artifacts.append({"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        state = {**self.state, "outcome": outcome, "reason": reason}
        verification = {**self.status(), "outcome": outcome, "reason": reason}
        parent_id = self.store.session(self.sid)["parent_id"]
        row = next(row for row in self.store.subagents(parent_id) if row["child_id"] == self.sid)
        result = {
            "ok": outcome == "complete",
            "submitted": True,
            "agent_id": self.sid,
            "name": row["name"],
            "status": "completed" if outcome == "complete" else outcome,
            "execution_status": "submitted",
            "outcome": outcome,
            "outcome_verified": outcome == "complete",
            "report": report,
            "report_truncated": False,
            "evidence_ids": list(evidence_ids),
            "uncertainties": list(uncertainties),
            "detail": reason,
            "verification": verification,
            "artifacts": artifacts,
        }
        self.store.submit_subagent_result(self.sid, result, state)
        self.state = state
        return {"ok": True, "submitted": True, "status": result["status"], "outcome": outcome}

    def status(self):
        evidence = self.evidence()
        required = self.required_checks.status()
        gaps = self.gaps(required)
        limitations = []
        if "coding" in self.state["profiles"] and "code_behavior" in gaps and not self.execution_available():
            limitations.append(
                "code_behavior still requires executable evidence, but shell is unavailable and "
                "execute_program is unavailable. "
                "Existing criteria remain in force. If existing evidence cannot satisfy them, "
                "finish partial/blocked and report the missing execution capability to the parent."
            )
        return {
            **self.state,
            # Answers may be large; their text belongs in delivery, not every bookkeeping reply.
            "final_answer": None,
            "final_answer_staged": bool(self.state.get("final_answer")),
            "gaps": gaps,
            **self.completion_status(gaps),
            "capability_limitations": limitations,
            "required_check_status": required,
            "guidance": {p: PROFILES[p] for p in self.state["profiles"]},
            "recent_evidence": [
                {
                    "evidence_id": cid,
                    "evidence_ref": f"E{e['id']}",
                    "tool": e["body"]["name"],
                    "source": {
                        key: e["body"]["result"][key]
                        for key in ("path", "url", "saved_output")
                        if e["body"]["result"].get(key)
                    },
                    "observed_excerpt": self.observed_text(e)[:160],
                    "ok": e["body"]["result"].get("ok"),
                    "fresh": e["id"] >= self.state["floor"],
                    "eligible_for_pass": e["id"] >= self.state["floor"] and self.support_error(e) is None,
                    "ineligibility_reason": (
                        f"Predates event {self.state['floor']}: {self.state['floor_reason']}"
                        if e["id"] < self.state["floor"]
                        else self.support_error(e)
                    ),
                    "process_id": e["body"]["result"].get("process_id"),
                    "verification_effect": self.state["process_effects"].get(
                        e["body"]["result"].get("process_id")
                    ),
                }
                for cid, e in list(evidence.items())[-12:]
            ],
        }

    def completion_status(self, gaps=None, *, require_plan=False):
        if gaps is None:
            gaps = self.gaps()
        # Verified child submissions have a stronger contract than plain final answers.
        # Scope this extra prerequisite to their rejection guidance, not ordinary finish_task.
        missing_plan = require_plan and (not self.state["required"] or not self.state["criteria"])
        actions = self.completion_actions(gaps)
        if missing_plan:
            action = (
                "Use plan_task to declare observable criteria with allowed tools, then record current "
                "evidence before resubmitting a verified complete result. Resolve other completion_actions."
            )
            actions = {
                "verification_plan": {
                    "action": "plan_task",
                    "reason": "Verified child submission requires declared criteria and current passing checks.",
                },
                **actions,
            }
        elif gaps:
            action = (
                "Follow completion_actions for the missing criteria. Reuse current successful execution "
                "results; record their evidence instead of rerunning them. Resolve unfinished commands."
            )
        elif self.state["outcome"] is None:
            action = (
                "Call finish_task to declare completion; passing criteria alone does not finalize the task."
            )
        else:
            action = "Deliver the substantive final response with findings and limitations."
        if self.state.get("pending_answer") and self.state["outcome"] is None:
            action += (
                " The user has not seen a final answer for this request. Earlier assistant proposals "
                "remain withheld even if they say the answer was delivered. Include answer with the "
                "current full deliverable in finish_task; incorporate new evidence and child reports."
            )
        return {
            "completion_ready": not gaps and not missing_plan,
            "completion_declared": self.state["outcome"] is not None,
            "next_action": action,
            "completion_actions": actions,
        }

    @staticmethod
    def required_check_blocker(cid, row):
        return {
            "criterion_id": cid,
            "kind": "required_check",
            "check_fingerprint": row["check_fingerprint"],
            "progress_event_id": row["progress_event_id"],
            "outcome_fingerprint": row["outcome_fingerprint"],
            "evidence_id": row.get("evidence_id"),
            "reason": (row.get("reason") or "The required check remains unresolved")[:700],
        }

    def stale_evidence_blocker(self, cid, evidence, stale):
        # Content signatures survive reruns with new call IDs. Exclude provenance
        # and annotation labels, which change without changing the observation.
        signatures = sorted(
            {hashlib.sha256(self.evidence_source(evidence[eid]).encode()).hexdigest() for eid in stale}
        )
        return {
            "criterion_id": cid,
            "kind": "stale_evidence",
            "check_fingerprint": hashlib.sha256(
                (cid if self.state.get("authority") else self.state["criteria"][cid]).encode()
            ).hexdigest(),
            "outcome_fingerprint": hashlib.sha256(json.dumps(signatures).encode()).hexdigest(),
            "progress_event_id": max(evidence[eid]["id"] for eid in stale),
            "evidence_ids": stale[-4:],
            "invalidator_event_id": self.state["floor"],
            "cause": "stale_evidence",
            "reason": "Cited evidence was invalidated by "
            + self.state.get("floor_reason", "a possible change")
            + ". Inspect the invalidating action and obtain coherent current evidence; do not "
            "reclassify old executions. If checks keep invalidating each other, report partial/blocked.",
        }

    def required_check_blockers(self):
        return [
            self.required_check_blocker(cid, row)
            for cid, row in self.required_checks.status().items()
            if not row["satisfied"]
        ]

    def required_check_recovery(self):
        if self.state.get("authority"):
            return (
                "Follow completion_actions and inspect failed output. A different command cannot satisfy "
                "a saved check. For a mistaken model working definition, use revise_plan with fresh "
                "anchored evidence; then execute and verify the revised method. User/host/runtime "
                "requirements remain binding. Report partial/blocked for unresolved limitations."
            )
        return (
            "Follow completion_actions. run_required_check executes the stored declaration exactly; "
            "choose timeout_seconds explicitly. A different successful command or a rewritten explanation "
            "cannot satisfy it. Inspect a failed check before retrying. If the immutable declaration is "
            "mistaken or impossible, finish partial/blocked with the specific error and available results. "
            "Do not alter correct behavior to satisfy a mistaken assertion."
        )

    def completion_actions(self, gaps):
        """Mechanical next steps only: never certify semantic correctness or run a tool."""
        required = self.required_checks.status() if gaps else {}
        evidence = self.evidence() if required else {}
        actions = {}
        for cid in gaps:
            if cid == "unfinished_commands":
                actions[cid] = {"action": "poll_process", "process_ids": list(self.state["pending"])}
            elif self.memory_changed(self.state["checks"].get(cid, {})):
                actions[cid] = {"action": "recall", "reason": "Saved memory evidence changed"}
            elif cid in self.state["inquiries"]:
                problem, finding = self.inquiry_resolution(cid)
                actions[cid] = {
                    "action": "record_finding" if problem else "record_verification",
                    "finding_id": self.state["inquiries"][cid]["finding_id"],
                    "reason": problem
                    or "Verify this supported/refuted resolution using current anchored "
                    "evidence from the linked finding; the requested outcome remains separate",
                }
            elif cid in required:
                row = required[cid]
                actions[cid] = {
                    "action": row["next_action"],
                    "evidence_id": row["evidence_id"],
                    "reason": row["reason"],
                }
                if row["evidence_id"] in evidence:
                    actions[cid]["evidence_ref"] = f"E{evidence[row['evidence_id']]['id']}"
                if row.get("process_id"):
                    actions[cid]["process_id"] = row["process_id"]
                if cid in self.state["required_checks"] and row["next_action"] in {
                    "run_check",
                    "inspect_failed_check",
                    "inspect_interrupted_check",
                }:
                    actions[cid]["check_runner"] = {
                        "tool": "run_required_check",
                        "arguments": {"criterion_id": cid},
                        "required_arguments": ["timeout_seconds"],
                    }
            else:
                actions[cid] = {
                    "action": "inspect_evidence",
                    "reason": "Record current observations with criterion-specific support; obtain missing evidence",
                }
        return actions

    def acknowledgement(self, **updates):
        """Bound mutation replies; task_status remains the detailed ledger inspection tool."""
        gaps = self.gaps()
        return {
            "revision": self.state["revision"],
            "outcome": self.state["outcome"],
            "reason": self.state["reason"],
            "gaps": gaps,
            **self.completion_status(gaps),
            **updates,
        }

    async def status_tool(self):
        self.sync_children()
        return self.status()

    def context(self):
        if not self.state["required"] and not self.state.get("pending_answer"):
            return ""
        status = self.status()
        encoded = json.dumps(status, ensure_ascii=False)
        if len(encoded) > 6000:
            path = str(self.store.save_artifact(self.sid, encoded))
            encoded = json.dumps(
                {
                    "profiles": self.state["profiles"],
                    "outcome": self.state["outcome"],
                    "gaps": status["gaps"],
                    "capability_limitations": status["capability_limitations"],
                    "required_check_status": status["required_check_status"],
                    **self.completion_status(status["gaps"]),
                    "pending_answer": self.state.get("pending_answer"),
                    "final_answer_staged": bool(self.state.get("final_answer")),
                    "saved_output": path,
                    "detail": "Use task_status or read_artifact for criteria and evidence.",
                }
            )
        return "\nDurable task verification (journal-derived, not proof of correctness):\n" + encoded

    def needs_completion(self):
        return self.state["required"] and (
            self.state["outcome"] is None or (self.state["outcome"] == "complete" and bool(self.gaps()))
        )
