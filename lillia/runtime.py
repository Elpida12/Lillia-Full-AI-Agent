"""Inspectable effective settings and startup source identity, without credentials."""

from __future__ import annotations

import hashlib
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from . import __version__


def source_digest():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


# Capture once when the runtime module loads, never relabel an old process with new disk code.
STARTUP_SOURCE_SHA256 = source_digest()
STARTED_AT = datetime.now(timezone.utc).isoformat()


def endpoint_origin(base_url):
    """Expose the server address without credentials, paths, queries, or fragments."""
    try:
        url = urlsplit(base_url)
        host = url.hostname or ""
        host = f"[{host}]" if ":" in host else host
        return f"{url.scheme}://{host}" + (f":{url.port}" if url.port else "")
    except (TypeError, ValueError):
        return "invalid endpoint"


class RuntimeIdentity:
    def __init__(self, config, registry):
        self.config, self.registry = config, registry
        self.startup = {
            "version": __version__,
            "pid": os.getpid(),
            "runtime_loaded_at": STARTED_AT,
            "startup_source_sha256": STARTUP_SOURCE_SHA256,
            "config_path": getattr(config, "_source_path", None),
            "loaded_config_sha256": getattr(config, "_source_sha256", None),
        }

    def effective(self):
        """Allowlisted settings: do not serialize arbitrary provider extra_body or environment."""
        c, m, d = self.config, self.config.model, self.config.delegation
        return {
            "profile": getattr(c, "_profile", None),
            "model_profile": getattr(c, "_model_profile", None),
            "mode": self.registry.mode,
            "model": m.name,
            "model_endpoint_origin": endpoint_origin(m.base_url),
            "max_tokens": m.max_tokens,
            "reasoning_budget_tokens": m.extra_body.get("reasoning_budget_tokens"),
            "reasoning_effort": m.extra_body.get("reasoning_effort"),
            "context_window_tokens": m.context_window_tokens,
            "max_context_chars": c.max_context_chars,
            "parent_context_chars": c.parent_context_chars,
            "compaction_trigger_ratio": c.compaction_trigger_ratio,
            "compaction_keep_ratio": c.compaction_keep_ratio,
            "degenerate_detection": m.degenerate_detection,
            "degenerate_retry_sampling": dict(m.degenerate_retry_sampling),
            "model_concurrency": d.model_concurrency,
            "subagent_model": (
                {
                    "name": c.subagent_model.name,
                    "endpoint_origin": endpoint_origin(c.subagent_model.base_url),
                    "context_window_tokens": c.subagent_model.context_window_tokens,
                    "max_tokens": c.subagent_model.max_tokens,
                    "reasoning_budget_tokens": c.subagent_model.extra_body.get("reasoning_budget_tokens"),
                    "reasoning_effort": c.subagent_model.extra_body.get("reasoning_effort"),
                    "degenerate_detection": c.subagent_model.degenerate_detection,
                    "model_concurrency": d.subagent_model_concurrency,
                }
                if c.subagent_model is not None
                else None
            ),
            "max_model_calls": d.max_model_calls,
            "max_steps": c.max_steps,
            "empty_response_retries": c.empty_response_retries,
            "length_retries": c.length_retries,
            "max_turn_seconds": c.max_turn_seconds,
            "command_timeout_seconds": self.registry.command_timeout,
            "delegation_enabled": d.enabled,
            "max_active_subagents": d.max_active,
            "delegation_strategy": d.strategy,
            "recovery_enabled": c.recovery.enabled,
            "recovery_max_tokens": c.recovery.max_tokens,
            "recovery_reasoning_tokens": c.recovery.reasoning_tokens,
            "read_loop_limit": c.recovery.read_loop_limit,
            "retrieval_enabled": c.retrieval.enabled,
            "automatic_recall": c.retrieval.enabled and c.retrieval.auto_recall,
            "continuous_indexing": c.retrieval.enabled and c.retrieval.continuous_indexing,
            "playbooks": asdict(c.playbooks),
            "outcome_review": asdict(c.outcome_review),
            "embedding_model": (
                {
                    "name": c.embedding_model.name,
                    "index_model": c.embedding_model.vector_model,
                    "endpoint_origin": endpoint_origin(c.embedding_model.base_url),
                    "dimensions": c.embedding_model.dimensions,
                    "batch_size": c.embedding_model.batch_size,
                }
                if c.embedding_model.enabled
                else None
            ),
        }

    def snapshot(self, *, compare_disk=False):
        result = {**self.startup, "effective": self.effective()}
        if compare_disk:
            try:
                result["source_changed_on_disk"] = source_digest() != STARTUP_SOURCE_SHA256
            except OSError:
                result["source_changed_on_disk"] = None
            path = self.startup["config_path"]
            if path:
                try:
                    current = hashlib.sha256(Path(path).read_bytes()).hexdigest()
                    result["config_changed_on_disk"] = current != self.startup["loaded_config_sha256"]
                except OSError:
                    result["config_changed_on_disk"] = None
            else:
                result["config_changed_on_disk"] = None
            result["reload_note"] = (
                "Effective values belong to this process. Disk edits are not hot-reloaded; "
                "restart and resume the saved session to load code or configuration changes."
            )
        return result
