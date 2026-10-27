"""Actual requests and host checks are authority; model plans are audited proposals.

Legacy ledgers retain their immutable contract. No tool argument can promote a
model criterion to trusted authority. Revision evidence establishes attribution,
not semantic permission to disregard any part of the actual request.
"""

from __future__ import annotations

import copy
import hashlib

from jsonschema import Draft202012Validator

from .verification_checks import RequiredChecks

OUTCOME = (
    "Verify and deliver the actual user request and its updates, preserving all requirements and "
    "constraints. Model plan revisions and refuted hypotheses do not discharge this obligation."
)
RESERVED = {"outcome", "code_behavior", "__requested_outcome__"}


class PlanningAuthority:
    def __init__(self, verification):
        self.v = verification

    def request(self):
        row = self.v.store.latest_request(self.v.sid)
        if row is None:
            raise ValueError("Request authority requires an actual user request")
        request_id, message = row
        text = message.get("content", "")
        return request_id, text, hashlib.sha256(text.encode()).hexdigest()

    def validate_host(self, records):
        records = [] if records is None else records
        schema = {
            "type": "array",
            "maxItems": 18,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "minLength": 1, "maxLength": 60},
                    "description": {"type": "string", "minLength": 1, "maxLength": 500},
                    "required_check": RequiredChecks.schema(),
                },
                "required": ["id", "description", "required_check"],
                "additionalProperties": False,
            },
        }
        errors = list(Draft202012Validator(schema).iter_errors(records))
        if errors:
            raise ValueError("Invalid host acceptance_checks: " + errors[0].message)
        ids = [record["id"] for record in records]
        if len(set(ids)) != len(ids) or any(cid in RESERVED or not cid.strip() for cid in ids):
            raise ValueError("Host acceptance check IDs must be unique and nonreserved")
        if any(not record["description"].strip() for record in records):
            raise ValueError("Host acceptance descriptions must be nonempty")
        # Reuse capability/cwd/command checks against an empty prospective ledger.
        # This preflight has no events or writes, even when called before a new request.
        prospective = copy.copy(self.v)
        prospective.state = {**self.v.state, "required_checks": {}}
        RequiredChecks(prospective).prepare(records)
        return copy.deepcopy(records)

    def bind(self, records):
        v = self.v
        request_id, _, digest = self.request()
        v.state["authority"] = {
            "version": 1,
            "request_id": request_id,
            "request_sha256": digest,
            "source": "actual_user_request",
        }
        if records:
            definitions = v.required_checks.prepare(records)
            v.require()
            for record in records:
                cid = record["id"]
                v.state["criteria"][cid] = record["description"]
                v.state["criterion_origins"][cid] = "host"
                v.state["required_checks"][cid] = definitions[cid]

    def sync(self):
        v = self.v
        authority = v.state.get("authority")
        if not authority:
            return False
        request_id, _, digest = self.request()
        if request_id != authority["request_id"]:
            raise ValueError("A new request needs a new verification ledger")
        if digest == authority["request_sha256"]:
            return False
        before = authority["request_sha256"]
        authority["request_sha256"] = digest
        v.state["checks"].pop("outcome", None)
        self.clear_delivery()
        v.store.event(
            v.sid,
            "planning_authority_updated",
            {
                "request_id": request_id,
                "previous_sha256": before,
                "request_sha256": digest,
            },
        )
        v.save()
        return True

    def status(self, *, include_request=False):
        authority = self.v.state.get("authority")
        if not authority:
            return {"version": "legacy"}
        request_id, text, digest = self.request()
        result = {
            **authority,
            "current_request_sha256": digest,
            "request_chars": len(text),
            "changed": request_id != authority["request_id"] or digest != authority["request_sha256"],
        }
        if include_request:
            result["request"] = text
        return result

    def criterion(self, cid):
        v = self.v
        return {
            "origin": v.state.get("criterion_origins", {}).get(cid, "legacy"),
            "revision_event_id": v.state.get("criterion_revision_events", {}).get(cid),
        }

    def clear_delivery(self):
        self.v.state.update(outcome=None, reason="", final_answer=None)

    def install(self):
        v = self.v
        if not v.state.get("authority"):
            v.registry.tools.pop("revise_plan", None)
            return
        from .tools import string

        support = v.registry.tools["record_verification"].properties

        # Other tools already explain these shared shapes; retain all validation.
        def concise(schema):
            if isinstance(schema, dict):
                return {key: concise(value) for key, value in schema.items() if key != "description"}
            if isinstance(schema, list):
                return [concise(value) for value in schema]
            return schema

        v.registry.add(
            "revise_plan",
            "Replace/retire a model working criterion using fresh quoted evidence. Does not pass it or "
            "change user/host requirements; archives prior definition. Omitted replacement fields stay.",
            {
                "criterion_id": string("Working criterion ID", 60),
                "action": {"enum": ["replace", "retire"]},
                "reason": string("Why evidence corrects the plan while preserving the actual request", 1500),
                "description": string("Replacement working criterion", 500),
                "required_check": concise(RequiredChecks.schema()),
                **{
                    key: concise(support[key])
                    for key in ("evidence_ids", "evidence_support", "evidence_selections")
                },
            },
            ["criterion_id", "action", "reason"],
            "read",
            self.revise,
        )

    async def revise(
        self,
        criterion_id,
        action,
        reason,
        evidence_ids=None,
        evidence_support=None,
        evidence_selections=None,
        description=None,
        required_check=None,
    ):
        v = self.v
        self.sync()
        if not v.state.get("authority"):
            raise ValueError("Legacy criteria/checks remain immutable")
        cid = criterion_id
        if cid not in v.state["criteria"]:
            raise ValueError("Unknown active criterion")
        origin = self.criterion(cid)["origin"]
        method_only = origin == "runtime" and cid == "code_behavior" and action == "replace"
        if origin != "working" and not method_only:
            raise ValueError("Actual request, host and runtime obligations cannot be revised or retired")
        if action not in {"replace", "retire"} or not reason.strip():
            raise ValueError("Revision needs replace/retire and a concrete reason")
        if method_only and (description is not None or required_check is None):
            raise ValueError("Runtime code behavior permits only an evidence-backed method replacement")
        if action == "retire" and (description is not None or required_check is not None):
            raise ValueError("Retirement cannot supply replacement fields")
        if cid in v.state["inquiries"] and action == "replace":
            raise ValueError("Finding identities stay fixed; retire this inquiry and add a new one")
        if v.state["pending"]:
            raise ValueError("Settle managed commands before revising their plan")
        evidence = v.evidence()
        selected = v.references.resolve(evidence_selections or [], evidence)
        supports = [*(evidence_support or []), *selected]
        ids = evidence_ids if evidence_ids is not None else [s.get("evidence_id") for s in supports]
        ids = list(dict.fromkeys([*ids, *(s["evidence_id"] for s in selected)]))
        v.validate_evidence_ids(ids, evidence)
        if not ids or len(ids) > 20 or any(s.get("evidence_id") not in ids for s in supports):
            raise ValueError("Revision needs actual cited evidence and matching support")
        fresh = [eid for eid in ids if evidence[eid]["id"] >= v.state["floor"]]
        if not fresh:
            raise ValueError("Revision requires fresh evidence after the latest possible change")
        for eid in fresh:
            event = evidence[eid]
            result = event["body"]["result"]
            pid = result.get("process_id")
            finished_execution = (
                event["body"]["name"] in {"shell", "execute_program", "poll_process"}
                and isinstance(result.get("exit_code"), int)
                and bool(result.get("command_fingerprint"))
            )
            # Failed finished commands can explain a bad method, but errors, incomplete
            # executions and stale process polls cannot manufacture fresh justification.
            if (
                event.get("kind") == "tool_error"
                or (not result.get("ok") and not finished_execution)
                or result.get("running")
                or result.get("timed_out")
                or result.get("stopped")
                or (pid and v.state["process_revisions"].get(pid) != v.state["revision"])
            ):
                raise ValueError("Revision requires a current completed observation")
        v.validate_support(fresh, evidence, supports)
        previous = {
            "description": v.state["criteria"][cid],
            "required_check": copy.deepcopy(v.state["required_checks"].get(cid)),
            "check": copy.deepcopy(v.state["checks"].get(cid)),
            "inquiry": copy.deepcopy(v.state["inquiries"].get(cid)),
        }
        replacement = None
        if action == "replace":
            replacement = {
                "id": cid,
                "description": description if description is not None else previous["description"],
            }
            if not replacement["description"].strip():
                raise ValueError("Replacement description must be nonempty")
            definition = required_check if required_check is not None else previous["required_check"]
            if definition is not None:
                replacement["required_check"] = {
                    k: value for k, value in definition.items() if k != "registered_event_id"
                }
            additions = v.required_checks.prepare([replacement], replacing={cid})
            if cid in additions:
                replacement["required_check"] = {
                    key: value for key, value in additions[cid].items() if key != "registered_event_id"
                }
            comparable = {
                "description": replacement["description"],
                "required_check": replacement.get("required_check"),
            }
            prior = {
                "description": previous["description"],
                "required_check": (
                    {
                        k: value
                        for k, value in previous["required_check"].items()
                        if k != "registered_event_id"
                    }
                    if previous["required_check"]
                    else None
                ),
            }
            if comparable == prior:
                raise ValueError("Revision must change the working definition; retrying is not a revision")
        event_id = v.store.event(
            v.sid,
            "plan_revision",
            {
                "request_id": v.state["authority"]["request_id"],
                "criterion_id": cid,
                "action": action,
                "reason": reason,
                "evidence_ids": ids,
                "evidence_support": supports,
                "previous": previous,
                "replacement": replacement,
            },
        )
        if action == "retire":
            v.state["criteria"].pop(cid)
            v.state["required_checks"].pop(cid, None)
            v.state["inquiries"].pop(cid, None)
            v.state["retired_criteria"][cid] = event_id
        else:
            v.state["criteria"][cid] = replacement["description"]
            if cid in additions:
                v.state["required_checks"][cid] = {**additions[cid], "registered_event_id": event_id}
        v.state["checks"].pop(cid, None)
        v.state["checks"].pop("outcome", None)
        self.clear_delivery()
        v.state["plan_history"].append(
            {
                "criterion_id": cid,
                "action": action,
                "event_id": event_id,
                "reason": reason[:200],
                "evidence_ids": ids,
            }
        )
        v.state["plan_revision_event_id"] = event_id
        v.state["criterion_revision_events"][cid] = event_id
        if len(v.state["plan_history"]) > 20:
            v.state["plan_history_omitted"] += len(v.state["plan_history"]) - 20
            v.state["plan_history"] = v.state["plan_history"][-20:]
        v.save()
        return v.acknowledgement(plan_revision_event_id=event_id, criterion_id=cid, action=action)
