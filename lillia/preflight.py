"""Estimate proposed read output from arguments and metadata, without reading file contents."""

from __future__ import annotations

import json
import stat

from jsonschema import Draft202012Validator

from .progress import READ_TOOLS
from .tools import MAX_FILE_BYTES


def proposed_reads(registry, calls, read_cap):
    """Only pure, valid read batches can be reassigned before execution.

    Estimates guide a planning decision, not permission, completion, or a token guarantee.
    Never inspect shell strings or delay a read that shares a batch with a mutation.
    """
    allowed = {schema["function"]["name"] for schema in registry.schemas()}
    reads = []
    for call in calls:
        try:
            name = call["function"]["name"]
            if name not in READ_TOOLS or name not in allowed:
                return []
            args = json.loads(call["function"]["arguments"])
            if not Draft202012Validator(registry.tools[name].parameters).is_valid(args):
                return []
            path = registry.resolve(args.get("path", "."))
            if name == "read_artifact" and not path.is_relative_to(
                (registry.store.root / "artifacts").resolve()
            ):
                return []
            info = path.stat()
            regular, directory = stat.S_ISREG(info.st_mode), stat.S_ISDIR(info.st_mode)
            limit = registry.output_limit
            if name == "read_file":
                if not regular or info.st_size > MAX_FILE_BYTES:
                    return []
                estimate = min(info.st_size, args.get("max_lines", 200) * 120, limit)
            elif name == "read_artifact":
                if not regular:
                    return []
                estimate = min(info.st_size, args.get("max_chars", 4000), limit)
            elif name == "search_files":
                if not directory:
                    return []
                estimate = min(args.get("max_results", 50) * 500, limit)
            else:
                if not directory:
                    return []
                # Directory metadata does not predict entry count. Ordinary discovery
                # still has the existing three-tool assessment.
                estimate = 0
            reads.append(
                {
                    "call_id": call["id"],
                    "tool": name,
                    "arguments": args,
                    "path": str(path),
                    "file_bytes": info.st_size if regular else None,
                    "mtime_ns": info.st_mtime_ns,
                    "estimated_chars": estimate,
                }
            )
        except (KeyError, TypeError, ValueError, OSError, RuntimeError):
            return []
    return reads[:read_cap]


def covered_by(registry, path, resources):
    try:
        target = registry.resolve(path)
    except (OSError, ValueError, RuntimeError):
        return False
    for resource in resources:
        try:
            owned = registry.resolve(resource)
        except (OSError, ValueError, RuntimeError):
            continue
        if target == owned or owned in target.parents:
            return True
    return False
