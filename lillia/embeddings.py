"""OpenAI-compatible embedding transport for the local evidence index.

Vectors are reference data for ranking candidates, never evidence. Nothing here decides
whether a passage answers a question; it only measures topical proximity. Callers read the
actual bytes through read_artifact before relying on a hit.
"""

from __future__ import annotations

import asyncio
import math
from array import array

import httpx

from .config import DEFAULT_TOOL_TIMEOUT_SECONDS, EmbeddingConfig

# One request carries at most this many characters, whatever batch_size asks for. The
# server splits a batch across its own ubatch anyway; an oversized body only risks a
# refusal far from the caller.
MAX_REQUEST_CHARS = 120_000
# Truncate an individual chunk rather than let one pathological blob fail a whole batch.
MAX_CHUNK_CHARS = 24_000
# Unlike a chat request, an embedding request is NOT deliberately unlimited. A search holds
# the index write lock across it, and a server that accepts the connection and then never
# answers (a llama-server still loading, for instance) would otherwise hang the tool call and
# the turn forever -- httpx bounds only `connect` by default. Zero in the configuration means
# "use this bound" for embeddings, not "no deadline"; set a positive value to override it.
# Match the generous tool timeout so slow indexing can finish.
DEFAULT_TIMEOUT_SECONDS = DEFAULT_TOOL_TIMEOUT_SECONDS


class EmbeddingError(RuntimeError):
    """The index degrades to lexical-only rather than failing a tool call."""


def truncate(vector: list[float], dimensions: int) -> array:
    """Matryoshka truncation, then renormalize so cosine stays a dot product.

    Qwen3-Embedding is trained so a leading slice remains usable. Measured on this
    corpus: 1024 of 4096 dims kept 4/5 top-1 agreement with the full width at a
    quarter of the storage and a quarter of the scan cost.
    """
    if dimensions and 0 < dimensions < len(vector):
        vector = vector[:dimensions]
    norm = math.sqrt(sum(value * value for value in vector))
    if not norm or not math.isfinite(norm):
        raise EmbeddingError("Embedding server returned a zero or non-finite vector")
    return array("f", [value / norm for value in vector])


class EmbeddingProvider:
    """Batched /v1/embeddings client. Failures raise EmbeddingError; callers degrade."""

    def __init__(self, config: EmbeddingConfig, transport=None):
        self.config = config
        headers = {"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}
        self.client = httpx.AsyncClient(
            headers=headers,
            timeout=httpx.Timeout(config.timeout_seconds or DEFAULT_TIMEOUT_SECONDS),
            transport=transport,
        )
        self._dimensions = 0
        self._native = 0

    @property
    def dimensions(self) -> int:
        """Width actually stored, known only after the first successful response."""
        return self._dimensions

    async def close(self):
        await self.client.aclose()

    async def probe(self) -> dict:
        """Confirm the endpoint answers and report the native and stored widths."""
        vectors = await self.embed(["probe"])
        return {
            "ok": True,
            "model": self.config.name,
            "native_dimensions": self._native,
            "stored_dimensions": len(vectors[0]),
        }

    async def embed(self, texts: list[str], *, instruction: str | None = None) -> list[array]:
        """Embed texts in order. An instruction marks a QUERY; documents take none."""
        if not texts:
            return []
        prepared = []
        for text in texts:
            text = text[:MAX_CHUNK_CHARS]
            if instruction:
                text = f"Instruct: {instruction}\nQuery: {text}"
            # An all-whitespace input makes last-token pooling meaningless and some
            # servers answer with a zero vector; give it a stable placeholder instead.
            prepared.append(text if text.strip() else "(empty)")
        out: list[array] = []
        for batch in self._batches(prepared):
            out.extend(await self._request(batch))
        return out

    def _batches(self, texts: list[str]):
        batch: list[str] = []
        size = 0
        for text in texts:
            if batch and (len(batch) >= self.config.batch_size or size + len(text) > MAX_REQUEST_CHARS):
                yield batch
                batch, size = [], 0
            batch.append(text)
            size += len(text)
        if batch:
            yield batch

    async def _request(self, batch: list[str]) -> list[array]:
        payload = {"model": self.config.name, "input": batch}
        try:
            async with asyncio.timeout(self.config.request_timeout_seconds or DEFAULT_TIMEOUT_SECONDS):
                return await self._with_retries(payload, len(batch))
        except TimeoutError:
            raise EmbeddingError("Embedding request timed out") from None

    async def _with_retries(self, payload: dict, expected: int) -> list[array]:
        for attempt in range(3):
            try:
                response = await self.client.post(f"{self.config.base_url}/embeddings", json=payload)
                response.raise_for_status()
                return self._parse(response.json(), expected)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in {429, 502, 503, 504} and attempt < 2:
                    await asyncio.sleep(2**attempt)
                    continue
                raise EmbeddingError(
                    f"Embedding server returned HTTP {exc.response.status_code} "
                    f"for model {self.config.name!r}. Check the endpoint and its logs."
                ) from None
            except (httpx.ConnectError, httpx.ConnectTimeout):
                if attempt < 2:
                    await asyncio.sleep(2**attempt)
                    continue
                raise EmbeddingError(
                    f"Cannot connect to {self.config.base_url}. Check the configured embedding endpoint "
                    "and its model server logs."
                ) from None
            except httpx.HTTPError as exc:
                raise EmbeddingError(f"Embedding request failed ({type(exc).__name__})") from None
            except (ValueError, KeyError, TypeError, OverflowError) as exc:
                raise EmbeddingError(f"Invalid embedding response ({type(exc).__name__})") from None
        raise EmbeddingError("Embedding request failed")

    def _parse(self, body: dict, expected: int) -> list[array]:
        if not isinstance(body, dict):
            raise EmbeddingError("Embedding server returned a response that is not an object")
        data = body.get("data")
        if not isinstance(data, list) or len(data) != expected:
            raise EmbeddingError(
                f"Embedding server returned {len(data) if isinstance(data, list) else 'no'} "
                f"vectors for {expected} inputs"
            )
        # Validate the entire mapping before pairing vectors with stored chunks. Sorting
        # duplicate or missing indices would silently attach vectors to the wrong evidence.
        indices = set()
        for item in data:
            if not isinstance(item, dict):
                raise EmbeddingError("Embedding server returned a vector entry that is not an object")
            index = item.get("index")
            if type(index) is not int or not 0 <= index < expected or index in indices:
                raise EmbeddingError(
                    "Embedding server returned missing, duplicate, or invalid vector indices"
                )
            indices.add(index)
        ordered = sorted(data, key=lambda item: item["index"])
        vectors = []
        native = self._native
        dimensions = self._dimensions
        for item in ordered:
            raw = item.get("embedding")
            if not isinstance(raw, list) or not raw:
                raise EmbeddingError("Embedding server returned an empty vector")
            for component, value in enumerate(raw):
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise EmbeddingError(
                        "Embedding server returned a nonnumeric or non-finite vector "
                        f"(input index {item['index']}, component {component})"
                    )
            # Two models can both truncate to 1024 dimensions while having incompatible
            # native spaces. Check the native width as well as the stored slice.
            if native and len(raw) != native:
                raise EmbeddingError(
                    f"Embedding native width changed from {native} to {len(raw)}; "
                    "check that the endpoint is serving the configured model."
                )
            native = len(raw)
            vector = truncate(raw, self.config.dimensions)
            if dimensions and len(vector) != dimensions:
                raise EmbeddingError(
                    f"Embedding width changed from {dimensions} to {len(vector)}; "
                    "the index would be inconsistent. Rebuild it with 'lillia index --rebuild'."
                )
            dimensions = len(vector)
            vectors.append(vector)
        # A rejected batch must not pin the client to its first, untrusted entry's width.
        self._native = native
        self._dimensions = dimensions
        return vectors
