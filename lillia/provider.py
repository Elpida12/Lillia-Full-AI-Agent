from __future__ import annotations

import asyncio
import json
import re
import uuid
from dataclasses import dataclass
from typing import Callable

import httpx

from .config import SAMPLING_KEYS, ModelConfig
from .degeneracy import StreamMonitor

MAX_OUTPUT_BYTES = 4_000_000
# Token-by-token SSE repeats metadata on every event. At the configured 32K-token
# budget this can exceed 4 MB even when the assembled response is only kilobytes.
MAX_STREAM_BYTES = 64_000_000
# Prepended to the forced end-of-reasoning tag when a per-request reasoning budget runs out,
# so the model knows why its reasoning stopped and still produces the required output.
REASONING_BUDGET_MESSAGE = (
    "\n\nReasoning budget exhausted. Stop reasoning and produce the required output now.\n"
)
# Servers without a token budget for reasoning (Ollama's OpenAI-compatible API) take a coarse
# reasoning_effort level instead. Unlisted levels such as "max" rank above "high".
REASONING_EFFORT_RANK = {"none": 0, "minimal": 1, "low": 2, "medium": 3, "high": 4}
# A refusal's own explanation (usage cap reached, not signed in, unknown model) is read from a
# structured JSON error only, bounded, and never echoes a raw body or the configured credential.
ERROR_BODY_BYTES = 4096
ERROR_DETAIL_CHARS = 300
RETRY_STATUSES = frozenset({429, 502, 503, 504})
# A short Retry-After is a pacing hint worth honoring; a long one (a usage window) is reported.
MAX_RETRY_AFTER_SECONDS = 60
# Account usage limits arrive as HTTP 429 like congestion, but no retry can succeed before the window
# resets or usage is added. Ollama words them "you (...) have reached your weekly usage limit, add
# extra usage: https://ollama.com/settings"; the status alone cannot tell them apart.
QUOTA_MARKERS = ("usage limit", "extra usage", "higher limits", "usage credit")
CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]")
# A backend steered by reasoning_effort can spend a lone token on a hidden thinking marker: GLM-5.3
# on Ollama at "low" returns no text at all for max_tokens=1 and "OK" from 2. Still well under a second.
EFFORT_PROBE_TOKENS = 8


class ModelError(RuntimeError):
    pass


def effort_rank(level: str) -> int:
    return REASONING_EFFORT_RANK.get(level, len(REASONING_EFFORT_RANK))


def reasoning_effort_for(configured: str, reasoning_budget: int | None, zero: str = "none") -> str:
    """The reasoning_effort a request carries, given the configured level and its token budget.

    Ollama has no token budget for reasoning, only this level. A zero budget sends ``zero``
    ("none" unless the model config says otherwise: GLM-5.3 answers "none" by thinking in its
    answer text, so its profile uses "low"). Any other explicit budget (summaries, assessments,
    recovery reviews) asks for "low". A configured level already below either is never raised,
    and no budget keeps the configured level.
    """
    if reasoning_budget is None:
        return configured
    target = zero if reasoning_budget == 0 else "low"
    return target if effort_rank(target) < effort_rank(configured) else configured


def error_detail(data, secret: str = "") -> str:
    """Return a server's structured error message, sanitized and bounded, or an empty string."""
    try:
        parsed = json.loads(data) if isinstance(data, (bytes, str)) else data
    except (ValueError, UnicodeDecodeError):
        return ""
    if not isinstance(parsed, dict):
        return ""
    error = parsed.get("error", parsed)
    message = error.get("message") if isinstance(error, dict) else error
    if not isinstance(message, str):
        return ""
    if secret:
        message = message.replace(secret, "[redacted]")
    return " ".join(CONTROL.sub(" ", message).split())[:ERROR_DETAIL_CHARS]


class ModelHTTPError(ModelError):
    """The server refused the request before streaming; carries its own reason when it gave one."""

    def __init__(self, status_code: int, detail: str = "", retry_after: float | None = None):
        self.status_code = status_code
        self.detail = detail
        self.retry_after = retry_after
        self.quota = status_code in {402, 403, 429} and any(
            marker in detail.lower() for marker in QUOTA_MARKERS
        )
        advice = (
            "This is an account usage limit, not a server fault; no tools executed. Progress is saved: "
            "resume after the limit resets or more usage is added."
            if self.quota
            else "Check the endpoint, model, credentials, and server logs."
        )
        super().__init__(
            f"Model server returned HTTP {status_code}" + (f" ({detail})" if detail else "") + ". " + advice
        )


class ModelStreamInterruptedError(ModelError):
    """The stream broke off before finishing, and before any answer text or tool call.

    A cloud proxy closes a failed upstream stream without a finish_reason or [DONE]; a direct
    connection resets. Nothing was displayed as an answer or executed, so the whole request can be
    repeated; the model scheduler does so a bounded number of times and journals each attempt.
    """

    def __init__(self, cause: str, reasoning_chars: int = 0):
        self.cause = cause
        self.reasoning_chars = reasoning_chars
        super().__init__(
            f"Model stream interrupted before any answer or tool call ({cause}, "
            f"reasoning_chars={reasoning_chars}); no tools from this response executed."
        )


class ModelEmptyResponseError(ModelError):
    """A completed generation supplied neither an answer nor an executable call."""

    def __init__(self, usage, finish_reason="stop", reasoning_chars=0):
        self.usage = usage
        self.finish_reason = finish_reason
        self.reasoning_chars = reasoning_chars
        super().__init__(
            "Model completed generation without an answer or tool calls "
            f"(finish_reason={finish_reason!r}, reasoning_chars={reasoning_chars}). "
            "No tools from this response executed. Retry the next concrete action; "
            "if the problem repeats, inspect the model server's output and chat template."
        )


class ModelDegenerateResponseError(ModelError):
    """A streamed generation was aborted because its reasoning stopped carrying analysis.

    The trailing window of reasoning text became a repetition loop or raw hex/number enumeration
    (see lillia/degeneracy.py). Nothing from the stream is executed; the partial reasoning is kept
    only so the caller can save a bounded diagnostic excerpt.
    """

    def __init__(self, reason, metrics, partial_reasoning="", output_chunks=0):
        self.reason = reason
        self.metrics = dict(metrics)
        self.partial_reasoning = partial_reasoning
        self.partial_content = ""
        self.partial_tool_call_chars = 0
        self.output_chunks = output_chunks
        self.usage = {}
        details = ", ".join(
            f"{key} {value}"
            for key, value in self.metrics.items()
            if key in {"compression_ratio", "prose_density"}
        )
        super().__init__(
            f"Aborted a degenerate model generation after {len(partial_reasoning):,} reasoning characters: "
            f"{reason}" + (f" ({details})" if details else "") + ". "
            "No tools from this response executed; the generation was discarded."
        )


class ModelLengthError(ModelError):
    """A truncated response is diagnostic evidence, never an executable reply."""

    def __init__(
        self,
        usage,
        max_tokens,
        context_window_tokens=0,
        partial_content="",
        partial_reasoning="",
        partial_tool_call_chars=0,
    ):
        self.usage = usage
        self.max_tokens = max_tokens
        self.context_window_tokens = context_window_tokens
        # Diagnostics only: incomplete text is never executed or used as notes.
        self.partial_content = partial_content
        self.partial_reasoning = partial_reasoning
        self.partial_tool_call_chars = partial_tool_call_chars
        prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
        measured = all(type(value) is int and value >= 0 for value in (prompt, completion))
        details = (
            f" Server reported {prompt:,} input + {completion:,} generated tokens "
            f"({prompt + completion:,} total)."
            if measured
            else ""
        )
        if measured and context_window_tokens and prompt + completion >= context_window_tokens - 1:
            advice = (
                f" The configured {context_window_tokens:,}-token context window is full. "
                "Compact history or lower max_context_chars to leave response space; "
                "increasing the reasoning or output budget alone will not help."
            )
        elif type(completion) is int and completion >= max_tokens:
            advice = (
                f" The response reached its {max_tokens:,}-token output limit, which includes reasoning. "
                "Use smaller responses or increase the applicable output limit only if the server's "
                "context window has room."
            )
        else:
            advice = (
                " The server may have exhausted its context window or imposed a smaller output limit. "
                "Check the per-request context capacity and reserve space for reasoning and output."
            )
        if partial_reasoning and not partial_content and not partial_tool_call_chars:
            advice += (
                f" The discarded generation was reasoning only ({len(partial_reasoning):,} characters) "
                "with no answer text or tool call."
            )
        super().__init__(
            "Incomplete model response (finish_reason='length'). "
            "No tools from this response executed." + details + advice
        )


@dataclass
class Reply:
    message: dict
    usage: dict


class ChatProvider:
    """OpenAI-compatible streaming transport; execution belongs to the agent."""

    supports_request_limits = True
    supports_progress = True
    supports_reasoning = True
    supports_sampling_overrides = True

    def __init__(self, config: ModelConfig, transport=None):
        self.config = config
        headers = {"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}
        self.client = httpx.AsyncClient(
            headers=headers,
            timeout=httpx.Timeout(config.timeout_seconds or None, connect=10),
            # ModelScheduler owns generation concurrency. A separate pool ceiling
            # would silently queue requests even when cloud concurrency is unlimited.
            limits=httpx.Limits(max_connections=None),
            transport=transport,
        )

    async def close(self):
        await self.client.aclose()

    async def models(self) -> list[str]:
        response = await self.client.get(f"{self.config.base_url}/models")
        response.raise_for_status()
        return [item["id"] for item in response.json()["data"]]

    async def probe(self) -> dict:
        """Load the configured model and verify inference with a one-token request.

        A gateway may advertise an unloaded or broken model from /models. The probe
        uses the normal streaming transport but never executes or saves its reply.
        The tiny output limit deliberately permits a length finish after generation;
        an effort-controlled backend gets a few tokens (EFFORT_PROBE_TOKENS) instead.
        """
        limit = EFFORT_PROBE_TOKENS if isinstance(self.config.extra_body.get("reasoning_effort"), str) else 1
        try:
            await self.complete(
                [{"role": "user", "content": "Reply with OK."}], max_tokens=limit, reasoning_budget=0
            )
        except ModelLengthError as exc:
            if not (exc.partial_content or exc.partial_reasoning):
                raise ModelError("Model readiness probe ended without generating any text") from None
        return {"ok": True, "model": self.config.name}

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        on_text: Callable[[str], None] | None = None,
        on_progress: Callable[[dict], None] | None = None,
        *,
        on_reasoning: Callable[[str], None] | None = None,
        max_tokens: int | None = None,
        reasoning_budget: int | None = None,
        sampling: dict | None = None,
    ) -> Reply:
        payload = {
            **self.config.extra_body,
            "model": self.config.name,
            "messages": messages,
            "stream": True,
            self.config.token_parameter: self.config.max_tokens if max_tokens is None else max_tokens,
        }
        if max_tokens is not None and (type(max_tokens) is not int or max_tokens < 1):
            raise ValueError("max_tokens must be a positive integer")
        if reasoning_budget is not None:
            if type(reasoning_budget) is not int or reasoning_budget < 0:
                raise ValueError("reasoning_budget must be a nonnegative integer")
            # These extensions are backend-specific; only override already configured capabilities.
            budgeted = "reasoning_budget_tokens" in self.config.extra_body
            template_kwargs = self.config.extra_body.get("chat_template_kwargs")
            if reasoning_budget == 0:
                # Zero means no reasoning. llama.cpp (b10380) does not enforce a zero budget: the
                # model keeps thinking until max_tokens. Disable thinking through the chat template
                # and keep the budget sampler out of the request. Without template control, a
                # one-token budget is enforced and ends reasoning before the first thought.
                if isinstance(template_kwargs, dict):
                    payload["chat_template_kwargs"] = {**template_kwargs, "enable_thinking": False}
                    if budgeted:
                        payload["reasoning_budget_tokens"] = -1
                elif budgeted:
                    payload["reasoning_budget_tokens"] = 1
            elif budgeted:
                payload["reasoning_budget_tokens"] = reasoning_budget
                payload.setdefault("reasoning_budget_message", REASONING_BUDGET_MESSAGE)
            effort = self.config.extra_body.get("reasoning_effort")
            if isinstance(effort, str):
                payload["reasoning_effort"] = reasoning_effort_for(
                    effort, reasoning_budget, self.config.zero_budget_effort
                )
        if sampling:
            # Recovery retries may vary sampling for one request; nothing else is overridable here.
            payload.update({key: value for key, value in sampling.items() if key in SAMPLING_KEYS})
        if tools:
            payload.update(tools=tools, tool_choice="auto")
        if self.config.include_usage:
            payload["stream_options"] = {"include_usage": True}
        try:
            # A logical request includes connection retries and their backoff.
            # Starting a fresh timeout for every attempt could exceed the stated
            # request deadline by several multiples, especially on slow servers.
            async with asyncio.timeout(self.config.request_timeout_seconds or None):
                return await self._complete_with_retries(payload, on_text, on_progress, on_reasoning)
        except TimeoutError:
            raise ModelError(
                "Model request interrupted (TimeoutError); no tools from the partial response were executed."
            ) from None

    async def _complete_with_retries(self, payload, on_text, on_progress, on_reasoning=None):
        for attempt in range(3):
            try:
                return await self._stream(payload, on_text, on_progress, on_reasoning)
            except ModelHTTPError as exc:
                if exc.status_code not in RETRY_STATUSES or exc.quota or attempt == 2:
                    raise
                if exc.retry_after is not None and exc.retry_after > MAX_RETRY_AFTER_SECONDS:
                    # A long wait is a usage window, not congestion: report it rather than sleep.
                    raise
                await asyncio.sleep(2**attempt if exc.retry_after is None else exc.retry_after)
            except (httpx.ConnectError, httpx.ConnectTimeout):
                if attempt < 2:
                    await asyncio.sleep(2**attempt)
                    continue
                raise ModelError(
                    f"Cannot connect to {self.config.base_url}. Run 'lillia doctor' or set LILLIA_BASE_URL."
                ) from None
            except httpx.HTTPError as exc:
                # Failed before any answer text or tool call (after one, _stream reports it itself):
                # a stall before the first byte or a reset mid-reasoning. Safe for the caller to repeat.
                raise ModelStreamInterruptedError(type(exc).__name__) from None
            except TimeoutError as exc:
                raise ModelError(
                    f"Model request interrupted ({type(exc).__name__}); "
                    "no tools from the partial response were executed."
                ) from None
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                raise ModelError(
                    f"Invalid model response ({type(exc).__name__}); no tools executed."
                ) from None
        raise ModelError("Model request failed")

    async def _refusal(self, response) -> ModelHTTPError:
        """Build the error for a non-2xx response from a bounded read of its body and Retry-After."""
        body = b""
        try:
            async for chunk in response.aiter_bytes():
                body += chunk
                if len(body) >= ERROR_BODY_BYTES:
                    break
        except httpx.HTTPError:
            body = b""
        try:
            retry_after = float(response.headers.get("retry-after", ""))
        except ValueError:
            retry_after = None
        if retry_after is not None and not 0 <= retry_after < float("inf"):
            retry_after = None
        detail = error_detail(body, self.config.api_key) if len(body) < ERROR_BODY_BYTES else ""
        return ModelHTTPError(response.status_code, detail, retry_after)

    async def _stream(self, payload: dict, on_text, on_progress=None, on_reasoning=None) -> Reply:
        content, reasoning = [], []
        # Calls in order of first appearance; slots maps each stream index to its current call.
        calls: dict[int, dict] = {}
        slots: dict[int, int] = {}
        usage, finish = {}, None
        total = 0
        output_bytes = 0
        output_chars = 0
        output_chunks = 0
        # Watches the thinking phase only; once answer text or a tool call starts, the model is acting.
        monitor = StreamMonitor(self.config) if self.config.degenerate_detection else None

        def count_output(text: str):
            nonlocal output_bytes
            output_bytes += len(text.encode("utf-8"))
            if output_bytes > MAX_OUTPUT_BYTES:
                raise ModelError("Model output exceeded 4 MB; no tools executed")

        done = False
        async with self.client.stream(
            "POST", f"{self.config.base_url}/chat/completions", json=payload
        ) as response:
            if response.is_error:
                raise await self._refusal(response)
            lines = response.aiter_lines()
            while True:
                try:
                    line = await anext(lines)
                except StopAsyncIteration:
                    break
                except httpx.HTTPError as exc:
                    if content or calls:
                        raise ModelError(
                            f"Model request interrupted ({type(exc).__name__}) after the answer began; "
                            "no tools from the partial response were executed."
                        ) from None
                    raise ModelStreamInterruptedError(type(exc).__name__, sum(map(len, reasoning))) from None
                total += len(line.encode("utf-8")) + 1
                if total > MAX_STREAM_BYTES:
                    raise ModelError("Model stream exceeded 64 MB; no tools executed")
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done = True
                    break
                if not data:
                    continue
                event = json.loads(data)
                if "error" in event:
                    detail = error_detail(event, self.config.api_key)
                    raise ModelError(
                        "Model server sent a streaming error"
                        + (f" ({detail})" if detail else "")
                        + "; no tools from this response executed. Check its logs."
                    )
                if event.get("usage") is not None and not isinstance(event["usage"], dict):
                    raise ModelError("Model returned invalid usage data")
                if event.get("usage"):
                    usage = event["usage"]
                    if on_progress:
                        on_progress({"usage": usage, "output_chars": output_chars})
                choices = event.get("choices", [])
                if not choices:
                    continue
                choice = choices[0]
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
                delta = choice.get("delta", {})
                text = delta.get("content")
                if text is not None and not isinstance(text, str):
                    raise ModelError("Model returned non-text content")
                if text:
                    count_output(text)
                    content.append(text)
                    output_chars += len(text)
                    if on_text:
                        on_text(text)
                thought = delta.get("reasoning_content")
                if not thought and delta.get("reasoning") is not None:
                    # Ollama and newer OpenAI-compatible servers name the same field "reasoning".
                    # Servers that send both carry identical text, so it is never counted twice.
                    thought = delta["reasoning"]
                if thought is not None and not isinstance(thought, str):
                    raise ModelError("Model returned non-text reasoning content")
                if isinstance(thought, str):
                    count_output(thought)
                    reasoning.append(thought)
                    output_chars += len(thought)
                    if thought and on_reasoning:
                        on_reasoning(thought)
                    if monitor is not None and not content and not calls:
                        verdict = monitor.feed(thought)
                        if verdict is not None:
                            # Leaving the stream context closes the connection, which cancels
                            # the server-side generation instead of letting it run to the limit.
                            raise ModelDegenerateResponseError(
                                verdict.reason,
                                verdict.metrics,
                                partial_reasoning="".join(reasoning),
                                output_chunks=output_chunks + 1,
                            )
                fragments = delta.get("tool_calls")
                if fragments is not None and not isinstance(fragments, list):
                    raise ModelError("Model returned invalid tool call data")
                for fragment in fragments or []:
                    index = fragment.get("index", 0)
                    if type(index) is not int or not 0 <= index < 32:
                        raise ModelError("Invalid tool call index")
                    fragment_id = fragment.get("id")
                    if fragment_id and not isinstance(fragment_id, str):
                        raise ModelError("Invalid tool call ID")
                    slot = slots.get(index)
                    # Only a call's first fragment carries its id, so a different id at an index
                    # already in use starts another call: MiniMax-M3 through Ollama numbers every
                    # whole call 0, which would otherwise glue the calls into one broken call.
                    if slot is None or (fragment_id and calls[slot]["id"] and fragment_id != calls[slot]["id"]):
                        if len(calls) >= 32:
                            raise ModelError("Model returned too many tool calls")
                        slot = slots[index] = len(calls)
                        calls[slot] = {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                    call = calls[slot]
                    if fragment_id:
                        count_output(fragment_id)
                        call["id"] = fragment_id
                    function = fragment.get("function") or {}
                    for key in ("name", "arguments"):
                        if function.get(key):
                            if not isinstance(function[key], str):
                                raise ModelError("Model returned non-text tool call data")
                            count_output(function[key])
                            call["function"][key] += function[key]
                            output_chars += len(function[key])
                if thought or text or delta.get("tool_calls"):
                    output_chunks += 1
                if on_progress:
                    phase = (
                        "thinking"
                        if thought
                        else "responding"
                        if text
                        else "preparing tools"
                        if delta.get("tool_calls")
                        else None
                    )
                    on_progress({"phase": phase, "output_chars": output_chars, "output_chunks": output_chunks})
        if finish is None and not done and not content and not calls:
            raise ModelStreamInterruptedError("stream ended without finish_reason", sum(map(len, reasoning)))
        if finish == "length":
            raise ModelLengthError(
                usage,
                payload[self.config.token_parameter],
                self.config.context_window_tokens,
                partial_content="".join(content),
                partial_reasoning="".join(reasoning),
                partial_tool_call_chars=sum(
                    len(call["function"]["name"]) + len(call["function"]["arguments"]) for call in calls.values()
                ),
            )
        if finish not in {"stop", "tool_calls"}:
            raise ModelError(
                f"Incomplete model response (finish_reason={finish!r}). "
                "No tools from this response executed. Check the model server's termination details."
            )
        if not "".join(content).strip() and not calls:
            raise ModelEmptyResponseError(usage, finish, sum(map(len, reasoning)))
        message = {"role": "assistant", "content": "".join(content) or None}
        if calls:
            seen = set()
            for call in calls.values():
                call["id"] = call["id"] or f"call_{uuid.uuid4().hex}"
                if not isinstance(call["id"], str) or call["id"] in seen:
                    raise ModelError("Duplicate or invalid tool call ID")
                seen.add(call["id"])
                if not call["function"]["name"]:
                    raise ModelError("Tool call is missing its name")
            message["tool_calls"] = [calls[i] for i in sorted(calls)]
        if reasoning:
            message["reasoning_content"] = "".join(reasoning)
        return Reply(message, usage)
