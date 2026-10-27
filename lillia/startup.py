"""Request model readiness for a session without owning server process lifetimes."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Callable
from urllib.parse import urlsplit

import httpx

from .config import Config, ModelConfig
from .embeddings import EmbeddingError, EmbeddingProvider
from .provider import ChatProvider, ModelError
from .runtime import endpoint_origin

OLLAMA_PORT = 11434
LLAMA_SWAP_HINT = "for this local stack: journalctl -u llama-swap -n 100 --no-pager"
OLLAMA_HINT = (
    "for Ollama: journalctl -u ollama -n 100 --no-pager; 'ollama list' must show each :cloud model "
    "(add one with 'ollama pull NAME'), and cloud models need a signed-in daemon ('ollama signin')"
)


class StartupError(RuntimeError):
    pass


def is_ollama(base_url: str) -> bool:
    """Ollama's daemon (its default port) or its hosted API. Only chooses diagnostic wording."""
    try:
        url = urlsplit(base_url)
        port = url.port
    except ValueError:
        return False
    host = (url.hostname or "").lower()
    return port == OLLAMA_PORT or host == "ollama.com" or host.endswith(".ollama.com")


async def ollama_account(model: ModelConfig, transport=None) -> dict:
    """Which ollama.com account and plan an Ollama endpoint bills, from its /api/me.

    Through the local daemon this is the account the daemon is signed in to, and its cloud
    generations draw on that subscription. Only the display name and plan are kept: never the
    email address or account ID.
    """
    root = model.base_url.rstrip("/").removesuffix("/v1")
    headers = {"Authorization": f"Bearer {model.api_key}"} if model.api_key else {}
    async with httpx.AsyncClient(
        headers=headers, timeout=httpx.Timeout(20, connect=10), transport=transport
    ) as client:
        response = await client.post(f"{root}/api/me")
    if response.status_code in {401, 403}:
        return {"signed_in": False}
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict):
        raise ValueError("unexpected /api/me response")
    name, plan = body.get("name"), body.get("plan")
    return {
        "signed_in": True,
        "name": name if isinstance(name, str) else "",
        "plan": plan if isinstance(plan, str) else "",
    }


async def cloud_account_lines(config: Config, transport=None) -> list[tuple[str, bool]]:
    """Report the Ollama account behind each cloud chat endpoint as (text, is_warning) lines.

    Informational and never fatal: the model probes decide readiness. A signed-out daemon or a free
    plan is a warning because cloud requests would be refused or drawn from the free allowance.
    """
    models = [config.model]
    if config.subagent_model is not None and config.delegation.enabled:
        models.append(config.subagent_model)
    lines, seen = [], set()
    for model in models:
        if not is_ollama(model.base_url) or model.base_url in seen:
            continue
        seen.add(model.base_url)
        origin = endpoint_origin(model.base_url)
        try:
            account = await ollama_account(model, transport)
        except (httpx.HTTPError, ValueError) as exc:
            lines.append((f"Ollama account at {origin}: not confirmed ({type(exc).__name__}).", True))
            continue
        if not account["signed_in"]:
            lines.append(
                (
                    f"Ollama at {origin} is not signed in to ollama.com, so cloud models will be refused. "
                    "Sign the daemon in with 'ollama signin', or use https://ollama.com/v1 with an API key.",
                    True,
                )
            )
            continue
        plan = account["plan"] or "unknown"
        text = (
            f"Ollama account {account['name'] or '(unnamed)'} · plan {plan}: "
            f"cloud generations through {origin} draw on this subscription's usage."
        )
        lines.append((text, plan.lower() in {"free", "unknown"}))
    return lines


async def load_startup_models(config: Config, on_status: Callable[[str], None]) -> None:
    """Probe all enabled roles and report failures before accepting session work.

    Separate probe clients allow a cold load its startup deadline without relaxing
    normal chat or retrieval deadlines. No keepalive, preload service, or unload is
    installed: the configured server remains responsible for its GPU placement and
    idle lifetime. No unload request is sent; gateway routing may evict competing models.
    """
    if not config.startup_models:
        return
    backends = [("Parent", config.model, ChatProvider)]
    if config.subagent_model is not None and config.delegation.enabled:
        backends.append(("Subagent", config.subagent_model, ChatProvider))
    if config.retrieval.enabled and config.embedding_model.enabled:
        backends.append(("Embeddings", config.embedding_model, EmbeddingProvider))
    failures, failed_urls = [], []
    for role, model, factory in backends:
        label = f"{role}: {model.name} at {model.base_url}"
        on_status(f"Loading/checking {label} (cold loads can take several minutes)…")
        probe_config = replace(
            model,
            timeout_seconds=config.startup_timeout_seconds,
            request_timeout_seconds=config.startup_timeout_seconds,
        )
        provider = factory(probe_config)
        try:
            async with asyncio.timeout(config.startup_timeout_seconds):
                result = await provider.probe()
        except (ModelError, EmbeddingError, TimeoutError) as exc:
            detail = str(exc) or f"Timed out after {config.startup_timeout_seconds} seconds"
            failures.append(f"{label}: {detail}")
            failed_urls.append(model.base_url)
            on_status(f"Failed {label}: {detail}")
        else:
            dimensions = (
                f" ({result['native_dimensions']} native dimensions, {result['stored_dimensions']} stored"
                + (f", index model {model.vector_model}" if model.vector_model != model.name else "")
                + ")"
                if role == "Embeddings"
                else ""
            )
            on_status(f"Ready {label}{dimensions}")
        finally:
            await provider.close()
    if failures:
        hints = []
        if not all(map(is_ollama, failed_urls)):
            hints.append(LLAMA_SWAP_HINT)
        if any(map(is_ollama, failed_urls)):
            hints.append(OLLAMA_HINT)
        raise StartupError(
            "Model startup failed:\n"
            + "\n".join(failures)
            + f"\nCheck the configured endpoints and model-server logs ({'; '.join(hints)}), then retry. "
            "To explicitly allow on-demand loading and lexical fallback, set startup_models = false."
        )
