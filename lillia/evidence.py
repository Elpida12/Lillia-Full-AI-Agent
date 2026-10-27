"""Short journal references and exact, runtime-selected observation passages.

References use globally unique journal event IDs, never an in-memory counter or
process handle. The v1 passage algorithm is intentionally fixed: old references
must select the same bytes after restart, compaction, and new observations.
"""

from __future__ import annotations

import re

PASSAGE_CHARS = 240
PAGE_SIZE = 8


def passages_v1(source: str) -> list[dict]:
    """Split into exact contiguous excerpts, preferring whole lines when possible."""
    passages = []
    start = 0
    while start < len(source):
        end = min(start + PASSAGE_CHARS, len(source))
        if end < len(source):
            newline = source.rfind("\n", start, end)
            if newline >= start:
                end = newline + 1
        excerpt = source[start:end]
        if excerpt.strip():
            passages.append({"passage_id": f"P{len(passages) + 1}", "excerpt": excerpt})
        start = end
    return passages


class EvidenceReferences:
    def __init__(self, verification):
        self.verification = verification

    def passages(self, event):
        # Prefer literal displayed content. evidence_source may also carry a JSON
        # copy of that same text, which would duplicate long file observations.
        return passages_v1(self.verification.observed_text(event))

    def summary(self, event):
        passages = self.passages(event)
        selected = sorted({0, 1, len(passages) - 1} & set(range(len(passages))))
        return {
            "evidence_ref": f"E{event['id']}",
            "evidence_passages": [passages[index] for index in selected],
            "passage_count": len(passages),
        }

    @staticmethod
    def lookup(evidence_ref, evidence):
        if isinstance(evidence_ref, str) and re.fullmatch(r"E[1-9][0-9]*", evidence_ref):
            event_id = int(evidence_ref[1:])
            for event in evidence.values():
                if event["id"] == event_id:
                    return event
        raise ValueError(
            "Unknown evidence_ref for this task. Use task_status for current observation references; "
            "references from another task or session cannot be cited."
        )

    def resolve(self, selections, evidence):
        supports = []
        for selection in selections:
            event = self.lookup(selection.get("evidence_ref"), evidence)
            passage = next(
                (p for p in self.passages(event) if p["passage_id"] == selection.get("passage_id")), None
            )
            if passage is None:
                raise ValueError(
                    f"{selection.get('evidence_ref')}: unknown passage_id. "
                    "Use inspect_evidence to read the labeled passages."
                )
            reasoning = selection.get("reasoning", "")
            if not isinstance(reasoning, str) or not reasoning.strip():
                raise ValueError("Explain why each selected passage supports this specific criterion")
            supports.append(
                {
                    "evidence_id": event["body"]["call_id"],
                    "excerpt": passage["excerpt"],
                    "reasoning": reasoning,
                }
            )
        return supports

    def inspect(self, evidence_ref, page=1):
        verification = self.verification
        event = self.lookup(evidence_ref, verification.evidence())
        passages = self.passages(event)
        pages = max(1, (len(passages) + PAGE_SIZE - 1) // PAGE_SIZE)
        if not isinstance(page, int) or isinstance(page, bool) or not 1 <= page <= pages:
            raise ValueError(f"page must be between 1 and {pages}")
        start = (page - 1) * PAGE_SIZE
        stale = event["id"] < verification.state["floor"]
        reason = (
            f"Predates event {verification.state['floor']}: {verification.state['floor_reason']}"
            if stale
            else verification.support_error(event)
        )
        return {
            "evidence_ref": evidence_ref,
            "source_evidence_id": event["body"]["call_id"],
            "tool": event["body"]["name"],
            "page": page,
            "next_page": page + 1 if page < pages else None,
            "passage_count": len(passages),
            "evidence_passages": passages[start : start + PAGE_SIZE],
            "eligible_for_pass": not stale and reason is None,
            "ineligibility_reason": reason,
            "guidance": "Select exact passage_id values and explain relevance. Inspection does not refresh evidence.",
        }
