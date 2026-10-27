"""Outbound HTTP policy: how Lillia identifies itself, retries, and searches.

Three facts from probing the live web drove this module, and each one is counterintuitive
enough to be worth recording:

* **Wikimedia rejects a spoofed browser User-Agent exactly as hard as a bare tool name.**
  ``Lillia-Full/0.1`` and a current Chrome string both returned 403; only a string carrying a
  parenthesised contact URL returned 200. Pretending to be a browser is not just impolite, it
  does not work. See :func:`headers` and ``WebConfig.contact``.

* **Some failures are transient.** ``search.yahoo.com`` answered ``500 INKApi Error`` on one
  request in eight, and ``archive.org`` answered 429 once and then 200 five times running.
  A single attempt turns a blip into a dead end, so :func:`with_retry` exists.

* **HTTP 200, and even a clean parse, does not mean success.** Mojeek serves a captcha under
  200 and DuckDuckGo's HTML endpoints serve an anomaly page under 202 -- but the sharpest case
  was Bing's RSS view, which returned 200 with ten well-formed <item> elements and a channel
  title echoing the query, while the items themselves matched only the query's *first token*:
  "RTX 3090 memory bandwidth" came back as articles about impressing new clients. It parsed
  perfectly and still lied. A backend is therefore believed only when it yields results that
  came from the query; otherwise the next backend is tried.
"""

from __future__ import annotations

import asyncio
import random
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html import unescape
from urllib.parse import urlsplit

import httpx

# Conditions that may differ on a second attempt: overload, throttling, and the Cloudflare
# 52x family. 403/404/410 are settled answers -- retrying them only burns wall-clock.
RETRY_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504, 509, 520, 521, 522, 523, 524, 525, 527, 530})
# Deliberate refusals of an automated client. A different source is the only way through.
BLOCKED_STATUS = frozenset({401, 403, 406, 451})

HTML_ACCEPT = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
XML_ACCEPT = "application/rss+xml,application/xml;q=0.9,*/*;q=0.8"
JSON_ACCEPT = "application/json,*/*;q=0.8"

# Favicons and thumbnails sit inside a result block and must not be mistaken for the result.
BRAVE_ASSET_HOSTS = re.compile(r"^https?://(imgs\.search|cdn\.search|search)\.brave\.com/")


class WebError(ValueError):
    """A network read that failed for good, after any retries the config allowed."""


class Retryable(Exception):
    """Internal signal: this attempt failed in a way another attempt might survive."""

    def __init__(self, detail: str, retry_after: float | None = None):
        super().__init__(detail)
        self.detail = detail
        self.retry_after = retry_after


def headers(config, accept: str = HTML_ACCEPT) -> dict:
    """Identify this client honestly, with a contact route operators can act on."""
    return {
        "User-Agent": config.header(),
        "Accept": accept,
        "Accept-Language": "en-US,en;q=0.9",
    }


def parse_retry_after(value: str | None) -> float | None:
    """Read a Retry-After header, which is either seconds or an HTTP date."""
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())


def retry_delay(config, attempt: int, retry_after: float | None, rng=None) -> float:
    """Exponential backoff with jitter, capped, and overridden by an explicit Retry-After."""
    ceiling = config.retry_max_sleep_seconds
    if retry_after is not None:
        return max(0.0, min(retry_after, ceiling))
    base = config.retry_backoff_seconds * (2 ** (attempt - 1))
    # Jitter keeps concurrent subagents from re-colliding on the same throttled host.
    jitter = (rng or random).uniform(0, base * 0.25)
    return min(base + jitter, ceiling)


def guidance(status: int) -> str:
    """Tell the model what to do next, so it stops re-fetching a host that will never answer."""
    if status in BLOCKED_STATUS:
        return (
            " This host refuses automated clients, so retrying or reformatting the URL will not "
            "help. Use web_search to find another source covering the same facts, or look for an "
            "official API or an archive.org snapshot."
        )
    if status in {404, 410}:
        return (
            " That URL does not exist. Do not guess further URLs on this host; use web_search to "
            "find the real address of the page you want."
        )
    if status == 429:
        return (
            " Rate limited even after retries. Leave this host alone for a while and get the "
            "information from a different source."
        )
    if status in RETRY_STATUS:
        return (
            " The server failed repeatedly, so this looks like an outage rather than a bad "
            "request. Try a different source, or an archive.org snapshot."
        )
    if status == 400:
        return " The server rejected the request itself; check the query parameters you built."
    return ""


def status_error(response: httpx.Response, url: str) -> WebError:
    reason = response.reason_phrase or ""
    return WebError(
        f"HTTP {response.status_code} {reason}".rstrip() + f" for {url}." + guidance(response.status_code)
    )


async def with_retry(config, operation, *, sleep=asyncio.sleep, rng=None):
    """Run ``operation`` until it succeeds, gives up, or runs out of attempts.

    ``operation`` raises :class:`Retryable` to ask for another attempt. Transport and timeout
    errors are retried on the same schedule; anything else propagates untouched.
    """
    attempts = max(1, config.retry_attempts)
    last = "request failed"
    for attempt in range(1, attempts + 1):
        try:
            return await operation()
        except Retryable as exc:
            last = exc.detail
            if attempt == attempts:
                break
            await sleep(retry_delay(config, attempt, exc.retry_after, rng))
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last = f"{type(exc).__name__}: {exc}".strip()
            if attempt == attempts:
                raise WebError(f"{last} (after {attempts} attempts)") from exc
            await sleep(retry_delay(config, attempt, None, rng))
    raise WebError(f"{last} (after {attempts} attempts)")


def check(response: httpx.Response, url: str) -> None:
    """Raise Retryable for a transient status, WebError for a settled refusal."""
    status = response.status_code
    if status in RETRY_STATUS:
        reason = response.reason_phrase or ""
        raise Retryable(
            f"HTTP {status} {reason}".rstrip() + f" for {url}." + guidance(status),
            parse_retry_after(response.headers.get("retry-after")),
        )
    if status >= 400:
        raise status_error(response, url)


# --------------------------------------------------------------------------------------
# Search backends. Each returns a list of {title, url, snippet}; an empty list means the
# backend did not answer usefully and the next one should be tried.
# --------------------------------------------------------------------------------------


def _clean(text: str | None) -> str:
    return " ".join((text or "").split())


def parse_brave(body: str, limit: int) -> list[dict]:
    """Pull organic results out of Brave's result page.

    Anchored on the ``data-type="web"`` / ``data-pos`` data attributes rather than class names:
    Brave's classes carry a per-deploy Svelte hash (``svelte-jmfu5f``) that would rot silently,
    while the data attributes describe what the block *is*. Blocks of any other data-type --
    notably the ``llm-snippet`` AI answer that renders first and shares the ``snippet`` class --
    are skipped, because they are not results.
    """
    results = []
    for block in body.split('data-type="web"')[1:]:
        # Stop at the next result so one block cannot bleed into the following one.
        nxt = block.find("data-type=")
        if nxt != -1:
            block = block[:nxt]
        # Drop the tail of the opening tag the split left behind (` data-keynav="true">`).
        tag_end = block.find(">")
        if tag_end != -1:
            block = block[tag_end + 1 :]
        url = next(
            (
                href
                for href in re.findall(r'href="(https?://[^"]+)"', block)
                if not BRAVE_ASSET_HOSTS.search(href)
            ),
            None,
        )
        if not url:
            continue
        title = ""
        match = re.search(r'class="title[^"]*"[^>]*\stitle="([^"]*)"', block)
        if match:
            title = unescape(match.group(1))
        else:
            match = re.search(r'class="title[^"]*"[^>]*>(.*?)</div>', block, re.S)
            if match:
                title = unescape(re.sub(r"<[^>]+>", " ", match.group(1)))
        # <cite> holds a breadcrumb restating the URL; it crowds out the real description.
        text = re.sub(r"<cite\b[^>]*>.*?</cite>", " ", block, flags=re.S)
        text = unescape(re.sub(r"<[^>]+>", " ", text))
        snippet = _clean(text.replace(title, " ", 1) if title else text)
        results.append({"title": _clean(title), "url": url, "snippet": snippet[:400]})
        if len(results) >= limit:
            break
    return results


async def search_brave(config, client, query: str, limit: int) -> list[dict]:
    """Brave's own index. It answers the compliant User-Agent, so no spoofing is needed."""

    async def attempt():
        response = await client.get(
            "https://search.brave.com/search",
            params={"q": query},
            headers=headers(config, HTML_ACCEPT),
        )
        check(response, "brave")
        return response.text

    return parse_brave(await with_retry(config, attempt), limit)


async def search_wikipedia(config, client, query: str, limit: int) -> list[dict]:
    """The MediaWiki search API -- the supported way in, and it needs the compliant UA."""

    async def attempt():
        response = await client.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": min(limit, 20),
                "format": "json",
            },
            headers=headers(config, JSON_ACCEPT),
        )
        check(response, "wikipedia")
        return response.json()

    try:
        payload = await with_retry(config, attempt)
    except ValueError as exc:
        if isinstance(exc, WebError):
            raise
        return []
    hits = payload.get("query", {}).get("search", []) if isinstance(payload, dict) else []
    results = []
    for hit in hits[:limit]:
        title = _clean(hit.get("title"))
        if not title:
            continue
        results.append(
            {
                "title": title,
                "url": "https://en.wikipedia.org/wiki/" + title.replace(" ", "_"),
                # snippet carries HTML search-term highlighting; strip the tags crudely.
                "snippet": _clean(
                    hit.get("snippet", "").replace('<span class="searchmatch">', "").replace("</span>", "")
                )[:400],
            }
        )
    return results


async def search_ddg(config, client, query: str, limit: int) -> list[dict]:
    """DuckDuckGo's Instant Answer API. Often empty, so it is a supplement, never the primary."""

    async def attempt():
        response = await client.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1, "no_redirect": 1},
            headers=headers(config, JSON_ACCEPT),
        )
        check(response, "duckduckgo")
        return response.json()

    try:
        payload = await with_retry(config, attempt)
    except ValueError as exc:
        if isinstance(exc, WebError):
            raise
        return []
    if not isinstance(payload, dict):
        return []
    results = []
    abstract = _clean(payload.get("AbstractText"))
    if abstract and payload.get("AbstractURL"):
        results.append(
            {
                "title": _clean(payload.get("Heading")) or query,
                "url": payload["AbstractURL"],
                "snippet": abstract[:400],
            }
        )
    for topic in payload.get("RelatedTopics", []):
        if len(results) >= limit:
            break
        if not isinstance(topic, dict) or "FirstURL" not in topic:
            continue
        results.append(
            {
                "title": _clean(topic.get("Text", ""))[:120],
                "url": topic["FirstURL"],
                "snippet": _clean(topic.get("Text", ""))[:400],
            }
        )
    return results


BACKENDS = {"brave": search_brave, "wikipedia": search_wikipedia, "ddg": search_ddg}


async def search(config, client, query: str, limit: int) -> dict:
    """Try each configured backend in order until one returns real results."""
    attempted, errors = [], {}
    for name in config.search_backends:
        backend = BACKENDS.get(name)
        if backend is None:
            continue
        attempted.append(name)
        try:
            results = await backend(config, client, query, limit)
        except WebError as exc:
            errors[name] = str(exc)
            continue
        if results:
            return {
                "query": query,
                "backend": name,
                "results": results[:limit],
                "backends_attempted": attempted,
                "backend_errors": errors,
            }
        errors.setdefault(name, "returned no usable results")
    return {
        "query": query,
        "backend": None,
        "results": [],
        "backends_attempted": attempted,
        "backend_errors": errors,
        "detail": (
            "No configured search backend returned results. Rephrase the query with fewer or more "
            "common terms, or fetch a known URL directly with fetch_url."
        ),
    }


def validate_url(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Use an HTTP(S) URL without embedded credentials")
    return url
