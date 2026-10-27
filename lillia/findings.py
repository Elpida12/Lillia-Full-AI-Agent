"""Request-scoped, evidence-linked findings independent of conversation compaction.

Statuses describe the worker's interpretation, not an independent proof. Journaled
observations establish provenance and test execution; they do not establish relevance.
"""

from __future__ import annotations

import json
import re

STATUSES = {"hypothesis", "supported", "refuted", "superseded", "inconclusive"}
VALIDITIES = {"valid", "invalid", "unknown"}
DISCLAIMER = "Finding statuses are evidence-linked interpretations, not verified task completion."


class Findings:
    MAX_FINDINGS = 100

    def __init__(self, notes):
        self.notes = notes
        self.store, self.sid = notes.store, notes.sid

    def records(self):
        rows = self.store.db.execute(
            "SELECT id,created,body FROM events WHERE session_id=? AND kind='finding_record' "
            "AND json_extract(body,'$.request_id') IS ? ORDER BY id",
            (self.sid, self.notes.request_id()),
        )
        records = {}
        for row in rows:
            body = json.loads(row["body"])
            records[body["finding_id"]] = {**body, "event_id": row["id"], "created": row["created"]}
        return records

    @staticmethod
    def dependency_problems(record, records, visited=None, artifact_versions=None):
        visited = set() if visited is None else visited
        if record["finding_id"] in visited:
            return ["cyclic assumption dependency"]
        visited = visited | {record["finding_id"]}
        problems = []
        for dependency in record.get("artifact_dependencies", []):
            current = (artifact_versions or {}).get(dependency["path"])
            if (
                current
                and current["event_id"] > dependency["event_id"]
                and current["sha256"] != dependency["sha256"]
            ):
                problems.append(
                    f"artifact {dependency['path']} has a newer observed version; recheck this finding"
                )
        for dependency in record.get("dependencies", []):
            current = records.get(dependency["finding_id"])
            if current is None or current["revision"] != dependency["revision"]:
                problems.append(f"assumption {dependency['finding_id']} changed; recheck this finding")
            elif current["status"] != "supported":
                problems.append(f"assumption {dependency['finding_id']} is {current['status']}")
            else:
                problems.extend(Findings.dependency_problems(current, records, visited, artifact_versions))
        return problems

    def latest_artifact_versions(self, records):
        paths = sorted(
            {
                dependency["path"]
                for record in records.values()
                for dependency in record.get("artifact_dependencies", [])
            }
        )
        if not paths:
            return {}
        rows = self.store.db.execute(
            "SELECT id,body FROM events WHERE session_id=? AND id>? AND kind='tool_finished' "
            "AND json_extract(body,'$.result.ok')=1 AND json_extract(body,'$.result.sha256') IS NOT NULL "
            "AND json_extract(body,'$.result.path') IN (" + ",".join("?" for _ in paths) + ") ORDER BY id",
            (self.sid, self.notes.evidence_floor(), *paths),
        )
        versions = {}
        for row in rows:
            result = json.loads(row["body"])["result"]
            versions[result["path"]] = {"sha256": result["sha256"], "event_id": row["id"]}
        return versions

    def snapshot(self, limit=6, finding_ids=None):
        if type(limit) is not int or not 1 <= limit <= 20:
            raise ValueError("Retrieve between one and twenty findings")
        records = self.records()
        artifact_versions = self.latest_artifact_versions(records)
        if finding_ids is not None:
            if (
                not isinstance(finding_ids, list)
                or len(finding_ids) > 20
                or any(not isinstance(key, str) or key not in records for key in finding_ids)
            ):
                raise ValueError("finding_ids must identify at most twenty current-request findings")
            selected = [records[key] for key in dict.fromkeys(finding_ids)][:limit]
        else:
            selected = sorted(records.values(), key=lambda row: row["event_id"], reverse=True)[:limit]
        findings = []
        for record in selected:
            problems = self.dependency_problems(record, records, artifact_versions=artifact_versions)
            findings.append({**record, "needs_recheck": bool(problems), "dependency_problems": problems})
        return {
            "request_id": self.notes.request_id(),
            "findings": findings,
            "total": len(records),
            "latest_event_id": max((row["event_id"] for row in records.values()), default=0),
            "disclaimer": DISCLAIMER,
        }

    async def list(self, finding_ids=None, limit=6):
        return self.snapshot(limit=limit, finding_ids=finding_ids)

    async def record(
        self,
        finding_id,
        claim,
        status,
        explanation,
        evidence_ids=None,
        test_validity="unknown",
        assumption_ids=None,
        superseded_by=None,
        expected_exit_codes=None,
    ):
        if self.notes.request_id() is None:
            raise ValueError("A finding needs a current user request or child assignment")
        if not isinstance(finding_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", finding_id):
            raise ValueError(
                "Use a stable finding_id of at most eighty letters, digits, dots, dashes or underscores"
            )
        if not isinstance(claim, str) or not claim.strip() or len(claim) > 800:
            raise ValueError("A finding needs a nonempty claim of at most 800 characters")
        if not isinstance(explanation, str) or not explanation.strip() or len(explanation) > 1200:
            raise ValueError("Explain test validity, observed outcome and limits in at most 1200 characters")
        if status not in STATUSES or test_validity not in VALIDITIES:
            raise ValueError("Invalid finding status or test_validity")
        evidence_ids = [] if evidence_ids is None else evidence_ids
        evidence = self.notes.evidence(evidence_ids)
        records = self.records()
        previous = records.get(finding_id)
        if previous and previous["claim"] != claim:
            raise ValueError(
                "A stable finding_id keeps its claim; create a new ID and supersede the old claim"
            )
        if previous is None and len(records) >= self.MAX_FINDINGS:
            raise ValueError(
                "Finding limit reached; refine existing findings instead of recording every action"
            )
        expected_exit_codes = (
            ([0] if previous is None else previous.get("expected_exit_codes", [0]))
            if expected_exit_codes is None
            else expected_exit_codes
        )
        if (
            not isinstance(expected_exit_codes, list)
            or not 1 <= len(expected_exit_codes) <= 8
            or any(type(code) is not int or not 0 <= code <= 255 for code in expected_exit_codes)
            or len(set(expected_exit_codes)) != len(expected_exit_codes)
        ):
            raise ValueError("Declare one to eight distinct expected_exit_codes between zero and 255")
        if assumption_ids is None:
            assumption_ids = (
                [dependency["finding_id"] for dependency in previous.get("dependencies", [])]
                if previous
                else []
            )
        if (
            not isinstance(assumption_ids, list)
            or len(assumption_ids) > 8
            or any(
                not isinstance(key, str) or key not in records or key == finding_id for key in assumption_ids
            )
            or len(set(assumption_ids)) != len(assumption_ids)
        ):
            raise ValueError(
                "Use at most eight distinct existing assumption IDs; a finding cannot depend on itself"
            )
        dependencies = [{"finding_id": key, "revision": records[key]["revision"]} for key in assumption_ids]
        artifacts, artifact_dependencies = {}, {}
        for observation in sorted(evidence, key=lambda row: row["event_id"]):
            for version in observation.get("artifact_versions", []):
                artifacts[(version["path"], version["sha256"])] = version
                artifact_dependencies[version["path"]] = {**version, "event_id": observation["event_id"]}
        candidate = {
            "finding_id": finding_id,
            "dependencies": dependencies,
            "artifact_dependencies": list(artifact_dependencies.values()),
        }
        proposed = {**records, finding_id: candidate}

        # Detect cycles even through currently unproven assumptions.
        def reaches(key, target, seen):
            if key == target:
                return True
            if key in seen:
                return False
            return any(
                reaches(dep["finding_id"], target, seen | {key})
                for dep in records[key].get("dependencies", [])
            )

        if any(reaches(key, finding_id, set()) for key in assumption_ids):
            raise ValueError("Assumption dependencies cannot form a cycle")
        if status in {"supported", "refuted"}:
            if test_validity != "valid" or not evidence:
                raise ValueError(
                    "Supported/refuted findings require valid tests and actual observation evidence"
                )
            for observation in evidence:
                expected_execution = (
                    observation["tool"] in {"execute_program", "shell", "poll_process"}
                    and type(observation.get("exit_code")) is int
                    and observation["exit_code"] in expected_exit_codes
                    and not observation.get("error")
                )
                if (
                    (not observation["ok"] and not expected_execution)
                    or observation.get("running")
                    or observation.get("timed_out")
                    or observation.get("stopped")
                    or (type(observation.get("exit_code")) is int and not expected_execution)
                ):
                    raise ValueError(
                        "A failed, unfinished or invalid test is inconclusive, not support or refutation"
                    )
            if previous and not any(
                observation["event_id"] > previous["event_id"] for observation in evidence
            ):
                raise ValueError(
                    "Updating support/refutation requires fresh evidence after the previous finding"
                )
            if self.dependency_problems(
                candidate, proposed, artifact_versions=self.latest_artifact_versions(proposed)
            ):
                raise ValueError("Resolve or recheck assumption dependencies before promoting this finding")
        if status == "superseded":
            if superseded_by not in records or superseded_by == finding_id:
                raise ValueError("Superseded findings must identify another existing finding")
        elif superseded_by is not None:
            raise ValueError("superseded_by applies only to superseded findings")
        body = {
            "request_id": self.notes.request_id(),
            "finding_id": finding_id,
            "claim": claim,
            "status": status,
            "explanation": explanation,
            "test_validity": test_validity,
            "evidence_ids": evidence_ids,
            "evidence_event_ids": [row["event_id"] for row in evidence],
            "artifact_versions": list(artifacts.values()),
            "dependencies": dependencies,
            "artifact_dependencies": list(artifact_dependencies.values()),
            "expected_exit_codes": expected_exit_codes,
            "superseded_by": superseded_by,
            "revision": (previous["revision"] + 1) if previous else 1,
            "interpretation_verified": False,
        }
        event_id = self.store.event(self.sid, "finding_record", body)
        return {
            "finding_id": finding_id,
            "revision": body["revision"],
            "event_id": event_id,
            "status": status,
            "interpretation_verified": False,
            "disclaimer": DISCLAIMER,
        }

    def install(self, registry):
        from .tools import string

        registry.add(
            "record_finding",
            "Preserve a stable hypothesis/finding and evidence-backed status at a useful checkpoint. "
            "A broken test is inconclusive, not a refutation. Support/refutation requires a valid finished "
            "observation; explain its relevance and limitations. This is not task completion verification.",
            {
                "finding_id": string("Stable ID; retain the same ID across tests of the same claim", 80),
                "claim": string("The exact claim; immutable for this ID", 800),
                "status": {"enum": sorted(STATUSES)},
                "explanation": string(
                    "Why the test is valid and what the observations establish or leave uncertain", 1200
                ),
                "test_validity": {"enum": sorted(VALIDITIES)},
                "expected_exit_codes": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 0, "maximum": 255},
                    "minItems": 1,
                    "maxItems": 8,
                    "uniqueItems": True,
                    "description": "Expected valid execution exits, default [0]. Explain intentional negative-test exits; unrelated errors remain invalid.",
                },
                "evidence_ids": {
                    "type": "array",
                    "items": string("Current-request observation ID", 100),
                    "maxItems": 6,
                    "uniqueItems": True,
                },
                "assumption_ids": {
                    "type": "array",
                    "items": string("Existing finding used as an assumption", 80),
                    "maxItems": 8,
                    "uniqueItems": True,
                },
                "superseded_by": string("Replacement finding ID, required for superseded status", 80),
            },
            ["finding_id", "claim", "status", "explanation"],
            "read",
            self.record,
        )
        registry.add(
            "list_findings",
            "Retrieve durable findings, assumptions, artifact versions and rejected hypotheses. "
            "Changed assumptions need rechecking; recorded status is an interpretation, not proof.",
            {
                "finding_ids": {"type": "array", "items": string("Finding ID", 80), "maxItems": 20},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            [],
            "read",
            self.list,
        )
