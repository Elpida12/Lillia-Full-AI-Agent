"""Conservative, bounded selection of durable memories for a request.

This is a lexical relevance filter, not a semantic oracle. Explicit recall remains
available when a relevant memory uses different words from the request.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

MAX_ENTRIES = 8
MAX_CHARS = 6000
_COMMON = set(
    "a an and are as at be been by can could do does for from had has have how i if in is it its "
    "me my of on or our please should that the their them then there these they this to was we "
    "were what when where which who why will with would you your need want help find get tell "
    "show know think believe forgot remind provide run command used use using launch check "
    "file files project home users server status notes facts preference preferences answers ".split()
)


def terms(text):
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return {
        word
        for word in re.findall(r"[^\W_]+(?:[.-][^\W_]+)*", normalized)
        if len(word) > 1 and word not in _COMMON and not word.isdecimal()
    }


def project_path(cwd):
    path = Path(cwd).expanduser().resolve()
    for parent in (path, *path.parents):
        if (parent / ".git").exists():
            return str(parent)
    return str(path)


def select_memories(records, request, cwd):
    """Use explicit scope or distinctive topical matches; do not match on tool history."""
    query = terms(request)
    project = project_path(cwd)
    ranked = []
    for record in records:
        scope = record["scope"]
        key_terms = terms(record["key"].replace("_", " ").replace("-", " "))
        # Keep identifiers such as a hyphenated product name as well as key words.
        key_terms |= terms(record["key"])
        overlap = query & terms(record["value"])
        identity = query & key_terms
        coverage = len(overlap | identity) / max(1, len(query))
        if scope == "global":
            score = 100
        elif scope == "project":
            if record["project"] != project:
                continue
            score = 50
        elif identity and (query <= key_terms or (len(overlap | identity) >= 2 and coverage >= 0.35)):
            score = 10 + len(identity) + len(overlap) / max(1, len(query))
        elif len(overlap) >= 2 and len(overlap) / max(1, len(query)) >= 0.6:
            score = len(overlap) / len(query)
        else:
            continue
        ranked.append((score, record))
    selected = []
    for _, record in sorted(ranked, key=lambda item: (-item[0], item[1]["key"])):
        candidate = selected + [record]
        if len(candidate) <= MAX_ENTRIES and len(json.dumps(candidate, ensure_ascii=False)) <= MAX_CHARS:
            selected = candidate
    return selected
