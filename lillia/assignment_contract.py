"""Shared tool schemas for the subagent's output contract."""

REPORT_DELIVERABLE = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"kind": {"const": "report"}},
    "required": ["kind"],
    "description": 'Runtime-saved findings: use {"kind":"report"} without a path.',
}

DELIVERABLE = {
    "oneOf": [
        REPORT_DELIVERABLE,
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "kind": {"const": "workspace_artifact"},
                "path": {"type": "string", "minLength": 1, "maxLength": 4096},
            },
            "required": ["kind", "path"],
        },
    ],
    "description": "Defaults to a runtime-saved report for compatibility. workspace_artifact requires "
    "a writable assignment, an owned output path, and a declared available writing tool.",
}
