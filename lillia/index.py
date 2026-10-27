"""Lexical-plus-vector recall over saved context artifacts.

WHY THIS EXISTS
    save_artifact names every artifact by the SHA-256 of its contents, and read_artifact
    needs that exact path. After compaction the surviving handoff notes carry, mechanically,
    one archive path (context.py) plus whatever paths the model chose to copy forward.
    Measured on the live store: across 42 sessions with a summary the median number of
    distinct artifact paths still named is 1; the worst session had 201 artifacts on disk
    and 10 reachable. The evidence was never lost, only unaddressable.

WHAT THIS IS NOT
    A ranking is not a reading. Cosine similarity measures topical proximity, not whether
    a passage answers a question -- a query whose answer was never recorded still returns
    a confident-looking score against text that merely shares its vocabulary. Every result
    is therefore a POINTER (path + char_offset) into read_artifact, and the tool reports
    its own coverage so that "not in these results" is never mistaken for "not recorded".

DESIGN
    Two retrieval arms over one chunk table. The lexical arm is sqlite FTS5: no model, no
    network, and it indexes the whole live corpus in about a second, so it stays available
    when the embedding server is down. The vector arm is a brute-force cosine scan in pure
    Python -- no numpy, keeping the project's four runtime dependencies -- over Matryoshka-
    truncated vectors. Measured on the live index: 58.2 us per vector at 1024 dims, so ~1.4s
    for the current 24,189-chunk corpus. Ranks are fused with reciprocal rank fusion rather
    than by comparing a BM25 score to a cosine, which are not on the same scale.

    Artifacts are content-addressed and therefore immutable: a path's contents never change,
    so indexing is append-only and never needs invalidation. Only new paths are considered.
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import heapq
import json
import math
import re
import sqlite3
import struct
import threading
import time
from array import array
from operator import mul
from pathlib import Path

from .config import EmbeddingConfig, RetrievalConfig
from .embeddings import EmbeddingError, EmbeddingProvider

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sources(
    ref TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    bytes INTEGER NOT NULL,
    chunks INTEGER NOT NULL,
    indexed TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE TABLE IF NOT EXISTS chunks(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ref TEXT NOT NULL REFERENCES sources(ref) ON DELETE CASCADE,
    session_id TEXT NOT NULL,
    char_offset INTEGER NOT NULL,
    text_sha TEXT NOT NULL,
    text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_ref ON chunks(ref);
CREATE INDEX IF NOT EXISTS chunks_sha ON chunks(text_sha);
CREATE INDEX IF NOT EXISTS chunks_recall_order ON chunks(
    session_id, (substr(text, 1, 19) = 'Historical attempt:') DESC, id DESC
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, content='chunks', content_rowid='id');
CREATE TABLE IF NOT EXISTS vectors(
    chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    data BLOB NOT NULL
);
"""

# Reciprocal rank fusion constant. The standard 60 damps the top of each list enough that
# one arm ranking something first does not by itself decide the fused order.
RRF_K = 60
# FTS5 treats bare punctuation as syntax: "llama-swap" parses as a column reference and
# raises OperationalError. Terms are SPLIT on punctuation, never stripped of it -- the
# default unicode61 tokenizer indexes "llama-swap" as the two tokens "llama" and "swap",
# so fusing them into "llamaswap" produces a term that can never match anything (verified:
# MATCH 'llamaswap' returns 0 rows against that text, MATCH 'llama OR swap' returns it).
TERM = re.compile(r"[^0-9a-z_]+")
# Two characters is the floor, not three: "kv", "io", "ai" and "q8" are exactly the kind of
# term worth searching for in this corpus. Single characters are noise.
MIN_TERM_CHARS = 2
# Stripped from a natural-language question before the OR-join. Deliberately short: every
# additional word here is a term the lexical arm can no longer discriminate on.
STOPWORDS = frozenset(
    """a an and are as at be been being by can could did do does for from had has have how
    if in is it its me my no not of on or our should so that the these this those to up us
    was we were what when where which while who why will with would you your""".split()
)


def chunk_text(text: str, size: int, overlap: int):
    """Yield (char_offset, chunk). Offsets index the artifact exactly as read_artifact does."""
    step = max(1, size - overlap)
    position = 0
    length = len(text)
    while position < length:
        piece = text[position : position + size]
        if piece.strip():
            yield position, piece
        if position + size >= length:
            break
        position += step


def match_expression(query: str) -> str:
    """Translate a question into an FTS5 OR-join.

    A bare natural-language string is an implicit AND in FTS5 and reliably matches nothing;
    an OR-join over sanitized content words is what makes the lexical arm answer at all.
    """
    terms = []
    for word in query.lower().split():
        for part in TERM.split(word):
            if len(part) >= MIN_TERM_CHARS and part not in STOPWORDS and part not in terms:
                terms.append(part)
    return " OR ".join(terms)


def pack(vector: array) -> bytes:
    return vector.tobytes()


def unpack(blob: bytes) -> array:
    values = array("f")
    values.frombytes(blob)
    return values


class EvidenceIndex:
    """Owns index.sqlite3. Every database call runs off the event loop."""

    def __init__(self, store, sid, retrieval: RetrievalConfig, embedding: EmbeddingConfig, provider=None):
        self.store = store
        self.sid = sid
        self.retrieval = retrieval
        self.embedding = embedding
        self.path = Path(store.root) / "index.sqlite3"
        self.lock_path = Path(store.root) / "index.lock"
        self._provider = provider
        self._owns_provider = provider is None
        self._degraded: str | None = None
        from .live_index import LiveIndex

        self.live = LiveIndex(self)

    # ---------------------------------------------------------------- provider

    def provider(self) -> EmbeddingProvider | None:
        if not self.embedding.enabled:
            return None
        if self._provider is None:
            self._provider = EmbeddingProvider(self.embedding)
        return self._provider

    async def close(self):
        await self.live.stop()
        if self._provider is not None and self._owns_provider:
            await self._provider.close()
            self._provider = None

    # ---------------------------------------------------------------- sqlite

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.executescript(SCHEMA)
        return connection

    def signature(self) -> str:
        """Vectors are only comparable within one model and width.

        The model is ``index_model`` when set: another gateway route to the same weights (the
        --cloud profile's GPU2 route) keeps writing to and reading from the same vector space.
        """
        return f"{self.embedding.vector_model}@{self.embedding.dimensions or 'native'}"

    def _check_signature(self, connection) -> bool:
        row = connection.execute("SELECT value FROM meta WHERE key='vector_signature'").fetchone()
        if row is None:
            # OR IGNORE, not a bare INSERT: a parent and two subagents can reach this at the
            # same moment on a fresh index, and the loser of that race would otherwise raise
            # sqlite3.IntegrityError out of the tool call and out of the turn.
            connection.execute(
                "INSERT OR IGNORE INTO meta(key,value) VALUES('vector_signature',?)",
                (self.signature(),),
            )
            connection.commit()
            row = connection.execute("SELECT value FROM meta WHERE key='vector_signature'").fetchone()
            if row is None:
                return True
        return row["value"] == self.signature()

    # ---------------------------------------------------------------- indexing

    def artifact_root(self) -> Path:
        return (Path(self.store.root) / "artifacts").resolve()

    def _known_refs(self, connection) -> set[str]:
        return {row["ref"] for row in connection.execute("SELECT ref FROM sources")}

    def _discover(self, connection, limit: int) -> list[tuple[Path, str]]:
        """Artifact paths not yet indexed. Content-addressed, so a known path never changes."""
        known = self._known_refs(connection)
        root = self.artifact_root()
        found = []
        if not root.is_dir():
            return found
        for session_dir in sorted(root.iterdir()):
            if not session_dir.is_dir():
                continue
            for artifact in sorted(session_dir.iterdir()):
                if artifact.name.startswith(".") or not artifact.is_file() or str(artifact) in known:
                    continue
                found.append((artifact, session_dir.name))
                if len(found) >= limit:
                    return found
        return found

    @staticmethod
    def _is_archive(text: str) -> bool:
        """A compaction source archive re-serializes journal rows already retired."""
        head = text[:400].lstrip()
        return head.startswith("{") and '"previous_summary"' in text[:2000]

    def _ingest(self, connection, artifact: Path, sid: str) -> int:
        try:
            text = artifact.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return 0
        if not self.retrieval.index_archives and self._is_archive(text):
            # Record it as seen so the scan does not reconsider it on every call.
            connection.execute(
                "INSERT OR REPLACE INTO sources(ref,session_id,bytes,chunks) VALUES(?,?,?,0)",
                (str(artifact), sid, len(text)),
            )
            return 0
        rows = list(chunk_text(text, self.retrieval.chunk_chars, self.retrieval.chunk_overlap))
        connection.execute(
            "INSERT OR REPLACE INTO sources(ref,session_id,bytes,chunks) VALUES(?,?,?,?)",
            (str(artifact), sid, len(text), len(rows)),
        )
        for offset, piece in rows:
            sha = hashlib.sha256(piece.encode("utf-8")).hexdigest()
            cursor = connection.execute(
                "INSERT INTO chunks(ref,session_id,char_offset,text_sha,text) VALUES(?,?,?,?,?)",
                (str(artifact), sid, offset, sha, piece),
            )
            connection.execute("INSERT INTO chunks_fts(rowid,text) VALUES(?,?)", (cursor.lastrowid, piece))
        return len(rows)

    def _sync_lexical(self, limit: int) -> dict:
        """Bring FTS5 current. Cheap by measurement; safe to run on every search."""
        connection = self.connect()
        try:
            with self.exclusive() as held:
                if not held:
                    return {"indexed_artifacts": 0, "locked": True}
                pending = self._discover(connection, limit)
                added = 0
                for artifact, sid in pending:
                    added += self._ingest(connection, artifact, sid)
                connection.commit()
                return {"indexed_artifacts": len(pending), "indexed_chunks": added, "locked": False}
        finally:
            connection.close()

    def _missing_vectors(self, connection, limit: int) -> list[sqlite3.Row]:
        return connection.execute(
            "SELECT c.id, c.text FROM chunks c LEFT JOIN vectors v ON v.chunk_id=c.id "
            "WHERE v.chunk_id IS NULL ORDER BY c.id LIMIT ?",
            (limit,),
        ).fetchall()

    def _write_vectors(self, rows: list[tuple[int, array]]):
        connection = self.connect()
        try:
            connection.executemany(
                "INSERT OR REPLACE INTO vectors(chunk_id,data) VALUES(?,?)",
                [(chunk_id, pack(vector)) for chunk_id, vector in rows],
            )
            connection.commit()
        finally:
            connection.close()

    def exclusive(self):
        """Non-blocking exclusive lock; a busy writer means skip, never queue."""
        index = self

        class Guard:
            def __enter__(self):
                self.handle = index.lock_path.open("a")
                try:
                    fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    self.handle.close()
                    self.handle = None
                    return False
                return True

            def __exit__(self, *exc):
                if self.handle is not None:
                    fcntl.flock(self.handle, fcntl.LOCK_UN)
                    self.handle.close()
                return False

        return Guard()

    async def refresh(self, *, embed: bool = True) -> dict:
        """Catch the lexical index up, then embed a bounded slice of what still lacks vectors.

        ``embed=False`` keeps the lexical sync and makes no network request at all, so
        mode="lexical" really is the model-free path its documentation promises.
        """
        report = await asyncio.to_thread(self._sync_lexical, 2000)
        budget = self.retrieval.backfill_chunks_per_call
        provider = self.provider() if embed else None
        report["embedded"] = 0
        if provider is None or budget <= 0:
            return report

        def pending():
            connection = self.connect()
            try:
                if not self._check_signature(connection):
                    return "stale", []
                return "ok", [(row["id"], row["text"]) for row in self._missing_vectors(connection, budget)]
            finally:
                connection.close()

        with self.exclusive() as held:
            if not held:
                report["locked"] = True
                return report
            # A rebuild can replace chunk IDs and the model signature. Read both only
            # after acquiring the same lock that protects the eventual vector write.
            state, rows = await asyncio.to_thread(pending)
            if state == "stale":
                self._degraded = (
                    f"Stored vectors were built for a different model or width than {self.signature()}. "
                    "Run 'lillia index --rebuild'."
                )
                return report
            if not rows:
                return report
            try:
                vectors = await provider.embed([text for _, text in rows])
            except EmbeddingError as exc:
                self._degraded = str(exc)
                return report
            await asyncio.to_thread(
                self._write_vectors, [(rows[i][0], vector) for i, vector in enumerate(vectors)]
            )
            report["embedded"] = len(vectors)
        return report

    # ---------------------------------------------------------------- search

    def _lexical(self, expression: str, k: int, sid: str | None) -> list[sqlite3.Row]:
        if not expression:
            return []
        connection = self.connect()
        try:
            clause = "AND c.session_id = ?" if sid else ""
            params: list = [expression]
            if sid:
                params.append(sid)
            params.append(k)
            return connection.execute(
                "SELECT c.id, c.ref, c.session_id, c.char_offset, c.text_sha, c.text, "
                "snippet(chunks_fts, 0, '', '', ' … ', 24) AS excerpt "
                f"FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid "
                f"WHERE chunks_fts MATCH ? {clause} ORDER BY rank LIMIT ?",
                params,
            ).fetchall()
        except sqlite3.OperationalError:
            # A query that survives sanitizing but still confuses the parser is a miss,
            # not a tool failure; the vector arm and the caller's own tools remain.
            return []
        finally:
            connection.close()

    def _semantic(self, query_vector: array, k: int, sid: str | None) -> tuple[list[dict], dict]:
        connection = self.connect()
        try:
            if not self._check_signature(connection):
                return [], {"scanned": 0, "stale": True}
            clause = "WHERE c.session_id = ?" if sid else ""
            params = [sid] if sid else []
            # ORDER BY id DESC is not cosmetic. When the corpus outgrows max_scan_chunks the
            # scan stops early, and without an explicit order sqlite returns rowid order --
            # so the cap would silently drop the NEWEST evidence and keep the oldest, which
            # is the wrong half. Newest-first means the cap drops the oldest instead, and
            # scan_oldest_dropped says so rather than leaving the omission invisible.
            cursor = connection.execute(
                "SELECT c.id, c.ref, c.session_id, c.char_offset, c.text_sha, c.text, v.data "
                f"FROM vectors v JOIN chunks c ON c.id = v.chunk_id {clause} ORDER BY c.id DESC",
                params,
            )
            width = len(query_vector)
            # A bounded heap, not a full sort: holding every scanned row would retain the
            # text of up to max_scan_chunks chunks (tens of MB) for a k-sized answer.
            best: list[tuple[float, int, dict]] = []
            scanned = 0
            mismatched = 0
            capped = False
            limit = self.retrieval.max_scan_chunks
            query = query_vector.tolist()
            unpacker = f"<{width}f"
            for row in cursor:
                if scanned >= limit:
                    capped = True
                    break
                blob = row["data"]
                if len(blob) != width * 4:
                    # A vector stored at another width cannot be compared; count it so a
                    # partially-rebuilt index is visible rather than quietly under-scanned.
                    mismatched += 1
                    continue
                scanned += 1
                # struct.unpack on the raw blob avoids building an array per row.
                score = sum(map(mul, struct.unpack(unpacker, blob), query))
                if len(best) < k:
                    heapq.heappush(best, (score, row["id"], dict(row)))
                elif score > best[0][0]:
                    heapq.heapreplace(best, (score, row["id"], dict(row)))
            results = [
                {
                    "id": row["id"],
                    "ref": row["ref"],
                    "session_id": row["session_id"],
                    "char_offset": row["char_offset"],
                    "text_sha": row["text_sha"],
                    "text": row["text"],
                    "score": score,
                }
                for score, _, row in sorted(best, key=lambda item: -item[0])
            ]
            return results, {
                "scanned": scanned,
                "stale": False,
                "capped": capped,
                "width_mismatched": mismatched,
                "scan_oldest_dropped": capped,
            }
        finally:
            connection.close()

    def _stats(self, sid: str | None = None) -> dict:
        """Counts for the population that was actually searched, not the whole index.

        A this_session_only search that reported global coverage would be describing
        chunks it never looked at.
        """
        connection = self.connect()
        try:
            where = "WHERE c.session_id = ?" if sid else ""
            params = [sid] if sid else []
            chunks = connection.execute(f"SELECT COUNT(*) AS n FROM chunks c {where}", params).fetchone()["n"]
            vectors = connection.execute(
                "SELECT COUNT(*) AS n FROM vectors v JOIN chunks c ON c.id = v.chunk_id " + where,
                params,
            ).fetchone()["n"]
            # Only sources that produced searchable text count as indexed; archives skipped
            # under index_archives = false are recorded to stop rescanning, not searched.
            source_where = "WHERE chunks > 0" + (" AND session_id = ?" if sid else "")
            sources = connection.execute(
                f"SELECT COUNT(*) AS n FROM sources {source_where}", params
            ).fetchone()["n"]
            skipped_where = "WHERE chunks = 0" + (" AND session_id = ?" if sid else "")
            skipped = connection.execute(
                f"SELECT COUNT(*) AS n FROM sources {skipped_where}", params
            ).fetchone()["n"]
            return {
                "artifacts_indexed": sources,
                "artifacts_skipped_unindexed": skipped,
                "chunks": chunks,
                "chunks_with_vectors": vectors,
            }
        finally:
            connection.close()

    async def stats(self, sid: str | None = None) -> dict:
        return await asyncio.to_thread(self._stats, sid)

    def _existing_connection(self, deadline: float, stopped: threading.Event) -> sqlite3.Connection:
        """Read an existing database without creating a schema, signature, or write lock."""
        connection = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True, timeout=0.05)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        connection.set_progress_handler(lambda: int(stopped.is_set() or time.monotonic() >= deadline), 1000)
        return connection

    def _existing_lexical(self, expression, pool, sid, deadline, stopped):
        connection = self._existing_connection(deadline, stopped)
        try:
            # Counts, candidate text, and vector signature describe the same read snapshot.
            connection.execute("BEGIN")
            lexical = []
            if expression:
                lexical = connection.execute(
                    "SELECT c.id, c.ref, c.session_id, c.char_offset, c.text_sha, c.text, "
                    "snippet(chunks_fts, 0, '', '', ' … ', 24) AS excerpt "
                    "FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid "
                    "WHERE chunks_fts MATCH ? AND c.session_id = ? ORDER BY rank LIMIT ?",
                    (expression, sid, pool),
                ).fetchall()
            row = connection.execute(
                "SELECT COUNT(*) AS chunks, COUNT(v.chunk_id) AS chunks_with_vectors "
                "FROM chunks c LEFT JOIN vectors v ON v.chunk_id=c.id WHERE c.session_id=?",
                (sid,),
            ).fetchone()
            signature = connection.execute("SELECT value FROM meta WHERE key='vector_signature'").fetchone()
            return lexical, dict(row), signature["value"] if signature else None
        finally:
            connection.close()

    def _existing_semantic(self, vector, pool, sid, deadline, stopped):
        connection = self._existing_connection(deadline, stopped)
        try:
            connection.execute("BEGIN")
            signature = connection.execute("SELECT value FROM meta WHERE key='vector_signature'").fetchone()
            # A rebuild can begin while the query embedding is in flight. Recheck in the
            # snapshot being scanned, without _check_signature's INSERT/commit behavior.
            if signature is None or signature["value"] != self.signature():
                return [], {"stale": True, "scanned": 0}
            cursor = connection.execute(
                "SELECT c.id, c.ref, c.session_id, c.char_offset, c.text_sha, c.text, v.data "
                "FROM chunks c JOIN vectors v ON v.chunk_id=c.id "
                # Preserve compact attempts when large, newer archives consume the
                # scan budget. Within each group, recent evidence still comes first.
                "WHERE c.session_id=? "
                "ORDER BY (substr(c.text, 1, 19) = 'Historical attempt:') DESC, c.id DESC",
                (sid,),
            )
            query = vector.tolist()
            width = len(query)
            unpacker = f"<{width}f"
            best = []
            scanned = visited = mismatched = 0
            capped = False
            limit = max(0, min(self.retrieval.max_scan_chunks, 8000))
            for row in cursor:
                if stopped.is_set() or time.monotonic() >= deadline or visited >= limit:
                    capped = True
                    break
                visited += 1
                blob = row["data"]
                if not isinstance(blob, bytes) or len(blob) != width * 4:
                    mismatched += 1
                    continue
                score = sum(map(mul, struct.unpack(unpacker, blob), query))
                if not math.isfinite(score):
                    mismatched += 1
                    continue
                scanned += 1
                if len(best) < pool:
                    heapq.heappush(best, (score, row["id"], dict(row)))
                elif score > best[0][0]:
                    heapq.heapreplace(best, (score, row["id"], dict(row)))
            results = []
            for score, _, row in sorted(best, key=lambda item: -item[0]):
                row.pop("data")
                results.append({**row, "score": score})
            return results, {
                "scanned": scanned,
                "capped": capped,
                "width_mismatched": mismatched,
                "stale": False,
            }
        finally:
            connection.close()

    async def search_existing(
        self, query: str, *, session_id: str, max_results: int = 3,
        timeout_seconds: float = 5.0, max_chars: int = 1800,
    ) -> dict:
        """Bounded automatic recall over already-indexed text from this session only.

        This deliberately bypasses refresh/search: automatic reminders must not enqueue
        corpus embeddings or compete with an active bulk indexer. Only the short query is
        embedded. Missing vectors, timeouts, and unavailable storage degrade to whatever
        lexical candidates were obtained. Cancellation still propagates to the caller.
        ``max_chars`` bounds the total exact source text in returned excerpts.
        """
        max_results = max(1, min(max_results, 5))
        max_chars = max(1, min(max_chars, 8000))
        max_results = min(max_results, max_chars)
        timeout_seconds = min(10.0, max(0.01, timeout_seconds)) if math.isfinite(timeout_seconds) else 5.0
        query = query[:1600].strip()
        expression = " OR ".join(match_expression(query).split(" OR ")[:32])
        stats = {"chunks": 0, "chunks_with_vectors": 0}
        lexical, semantic = [], []
        scan = {}
        degraded = []
        reachable = False
        stopped = threading.Event()
        deadline = time.monotonic() + timeout_seconds
        pool = max(max_results * 4, 12)
        try:
            if not self.retrieval.enabled or not query or not session_id:
                degraded.append("Automatic recall requires enabled retrieval, a query, and a session ID.")
            else:
                async with asyncio.timeout(timeout_seconds):
                    lexical, stats, signature = await asyncio.to_thread(
                        self._existing_lexical, expression, pool, session_id, deadline, stopped
                    )
                    if not self.embedding.enabled:
                        degraded.append("Embeddings are disabled; existing lexical evidence only.")
                    elif not stats["chunks_with_vectors"]:
                        degraded.append("No existing vectors in this session; lexical evidence only.")
                    elif signature != self.signature():
                        degraded.append("Stored vectors have a missing or incompatible model signature.")
                    else:
                        provider = self.provider()
                        remaining = max(0.0, deadline - time.monotonic())
                        try:
                            # Reserve part of the budget for the CPU scan. The embedding
                            # coroutine runs directly here, so timeout cancels its HTTP
                            # request rather than leaving a background task running.
                            async with asyncio.timeout(remaining * 0.75):
                                vectors = await provider.embed(
                                    [query], instruction=self.embedding.query_instruction
                                )
                            reachable = True
                            semantic, scan = await asyncio.to_thread(
                                self._existing_semantic, vectors[0], pool, session_id, deadline, stopped
                            )
                            if scan.get("stale"):
                                degraded.append("Stored vectors changed model signature during lookup.")
                            if scan.get("width_mismatched"):
                                degraded.append("Some existing vectors have invalid dimensions or values.")
                        except TimeoutError:
                            degraded.append("Automatic semantic recall timed out; lexical evidence only.")
                        except EmbeddingError as exc:
                            degraded.append(str(exc))
        except TimeoutError:
            degraded.append("Automatic recall reached its time limit; coverage may be incomplete.")
        except (sqlite3.Error, OSError) as exc:
            degraded.append(f"Existing evidence index unavailable ({type(exc).__name__}).")
        finally:
            # sqlite work runs off the event loop; progress handlers and the scan loop
            # stop promptly even if the awaiting task was cancelled.
            stopped.set()

        fused = self._fuse(lexical, semantic, max_results)
        rows = {row["id"]: row for row in [*semantic, *lexical]}
        similarities = {row["text_sha"]: row["score"] for row in reversed(semantic)}
        per_excerpt = max(1, max_chars // max(1, len(fused)))
        for item in fused:
            # FTS snippets contain ellipses and may join separated passages. Inject exact
            # contiguous bytes so the supplied source offset always addresses the text.
            item["excerpt"] = rows[item["chunk_id"]]["text"][:per_excerpt]
            item["lexical_match"] = item["lexical_rank"] is not None
            item["similarity"] = similarities.get(item["text_sha"])
        return {
            "results": fused,
            "query_terms": expression,
            "scope": "this session",
            "semantic_available": bool(scan.get("scanned") and not scan.get("stale")),
            "embedding_server_reachable": reachable,
            "semantic_chunks_scanned": scan.get("scanned", 0),
            "scan_capped": bool(scan.get("capped")),
            "vector_coverage": round(stats["chunks_with_vectors"] / stats["chunks"], 4)
            if stats["chunks"] else 0.0,
            "embedded_this_call": 0,
            "index_current": False,
            **stats,
            **({"degraded": " ".join(degraded)} if degraded else {}),
            "coverage_notes": [
                "Only already-indexed artifacts in this session were searched. Recent artifacts and "
                "unarchived messages may be absent. Similarity does not establish completed work."
            ],
        }

    async def search(self, query: str, max_results: int, mode: str, session_id: str | None) -> dict:
        """Return fused candidate locations plus an honest account of coverage."""
        self._degraded = None
        notes: list[str] = []
        # mode="lexical" is documented as the model-free path, so it must not trigger the
        # embedding backfill on its way through.
        refresh = await self.refresh(embed=mode != "lexical")
        if refresh.get("locked"):
            notes.append(
                "Another process holds the index write lock, so this search ran against the "
                "index as it already stood; artifacts saved very recently may not be in it yet."
            )
        pool = max(max_results * 4, 20)
        expression = match_expression(query)

        lexical: list[sqlite3.Row] = []
        if mode in {"auto", "lexical"}:
            lexical = await asyncio.to_thread(self._lexical, expression, pool, session_id)

        semantic: list[dict] = []
        scan: dict = {}
        provider = self.provider() if mode in {"auto", "semantic"} else None
        server_reachable = False
        if provider is not None:
            try:
                vectors = await provider.embed([query], instruction=self.embedding.query_instruction)
                server_reachable = True
                semantic, scan = await asyncio.to_thread(self._semantic, vectors[0], pool, session_id)
                if scan.get("stale"):
                    self._degraded = (
                        "Stored vectors do not match the configured embedding model or width. "
                        "Run 'lillia index --rebuild'."
                    )
                if scan.get("width_mismatched"):
                    notes.append(
                        f"{scan['width_mismatched']} stored vectors are at another width and "
                        "could not be compared; run 'lillia index --rebuild'."
                    )
            except EmbeddingError as exc:
                self._degraded = str(exc)
        elif mode in {"auto", "semantic"} and not self.embedding.enabled:
            self._degraded = "embedding_model.enabled is false; lexical results only."

        fused = self._fuse(lexical, semantic, max_results)
        stats = await asyncio.to_thread(self._stats, session_id)
        coverage = round(stats["chunks_with_vectors"] / stats["chunks"], 4) if stats["chunks"] else 0.0
        scanned = scan.get("scanned", 0)
        # "Available" must mean the arm actually looked at something. A reachable server that
        # scanned zero vectors is not semantic coverage, and reporting it as such would make a
        # 0%-covered search read exactly like a thorough one.
        semantic_available = bool(server_reachable and not scan.get("stale") and scanned)
        if server_reachable and not scanned and not scan.get("stale"):
            notes.append(
                "The embedding server answered, but no stored vectors were in scope, so the "
                "semantic arm contributed nothing to these results."
            )
        if scan.get("capped"):
            notes.append(
                f"The vector scan stopped at retrieval.max_scan_chunks "
                f"({self.retrieval.max_scan_chunks:,}); the oldest chunks were not scanned."
            )
        return {
            "results": fused,
            "query_terms": expression,
            "mode_requested": mode,
            "scope": "this session" if session_id else "all sessions",
            "lexical_hits": len(lexical),
            "semantic_hits": len(semantic),
            "semantic_available": semantic_available,
            "embedding_server_reachable": server_reachable,
            "semantic_chunks_scanned": scanned,
            "vector_coverage": coverage,
            "pending_vectors": max(0, stats["chunks"] - stats["chunks_with_vectors"]),
            "embedded_this_call": refresh.get("embedded", 0),
            "index_current": not refresh.get("locked", False),
            "scan_capped": bool(scan.get("capped")),
            **stats,
            **({"degraded": self._degraded} if self._degraded else {}),
            **({"coverage_notes": notes} if notes else {}),
            "interpretation": (
                "Ranked candidate locations, not answers and not verified evidence. Similarity "
                "reflects shared topic, not whether a passage answers the question. Read each "
                "path with read_artifact at the given char_offset before relying on it. "
                "Absence from these results is not evidence that something was never recorded: "
                "weigh index_current, vector_coverage and semantic_chunks_scanned, and note that "
                "the lexical arm matches only the terms in query_terms."
            ),
        }

    def _fuse(self, lexical, semantic, max_results: int) -> list[dict]:
        """Reciprocal rank fusion keyed on chunk TEXT, not on chunk id.

        Byte-identical text routinely lives at two chunk ids: a compaction archive
        re-serializes messages that often also exist as their own spilled artifact. Keying
        on id and collapsing afterwards loses information -- the lexical arm can rank copy A
        while the vector arm ranks copy B, and the survivor would then report one of the two
        ranks as None and carry only half the fused score. Keying on the text hash makes both
        arms contribute to the same entry, and the surviving location is the best-ranked one.
        """
        fused: dict[str, dict] = {}
        # Distinct chunk ids seen per text, so that the SAME chunk returned by both arms is
        # not miscounted as a second location holding the same text.
        locations: dict[str, set] = {}

        def entry_for(row, excerpt):
            key = row["text_sha"]
            locations.setdefault(key, set()).add(row["id"])
            entry = fused.get(key)
            if entry is None:
                entry = fused[key] = {
                    "chunk_id": row["id"],
                    "path": row["ref"],
                    "session_id": row["session_id"],
                    "char_offset": row["char_offset"],
                    "chunk_chars": len(row["text"]),
                    "text_sha": key,
                    "excerpt": excerpt,
                    "score": 0.0,
                    "lexical_rank": None,
                    "semantic_rank": None,
                    "similarity": None,
                }
            return entry

        # Each arm contributes AT MOST ONE rank term per distinct text, taken at its best
        # rank. Adding a term per copy would score text merely for being duplicated -- and
        # the most duplicated text in this corpus is boilerplate, not the best answer.
        for rank, row in enumerate(lexical, 1):
            entry = entry_for(row, (row["excerpt"] or row["text"][:400]).strip())
            if entry["lexical_rank"] is None:
                entry["lexical_rank"] = rank
                entry["score"] += 1.0 / (RRF_K + rank)
        for rank, row in enumerate(semantic, 1):
            entry = entry_for(row, row["text"][:400].strip())
            if entry["semantic_rank"] is None:
                entry["semantic_rank"] = rank
                entry["similarity"] = round(row["score"], 4)
                entry["score"] += 1.0 / (RRF_K + rank)
        ordered = sorted(fused.values(), key=lambda e: -e["score"])
        for entry in ordered:
            entry["score"] = round(entry["score"], 6)
            entry["duplicate_locations"] = len(locations[entry["text_sha"]]) - 1
        return ordered[:max_results]

    # ---------------------------------------------------------------- bulk build

    def _drop_all(self):
        connection = self.connect()
        try:
            connection.executescript(
                "DELETE FROM vectors;"
                "INSERT INTO chunks_fts(chunks_fts) VALUES('delete-all');"
                "DELETE FROM chunks; DELETE FROM sources; DELETE FROM meta;"
            )
            connection.commit()
        finally:
            connection.close()

    async def build(self, *, rebuild: bool = False, on_progress=None) -> dict:
        """Full backfill for 'lillia index'. Bounded only by the corpus."""
        # The lock is taken BEFORE anything is dropped. Deleting first and then discovering
        # the lock is held would destroy the index and return an error, losing it for nothing.
        with self.exclusive() as held:
            if not held:
                return {"ok": False, "error": "Another process is indexing (index.lock held)"}
            if rebuild:
                await asyncio.to_thread(self._drop_all)
            provider = self.provider()
            if provider is not None:
                # A signature mismatch means the stored vectors came from another model or
                # width. Continuing would write a second, incomparable vector space into the
                # same table and then report full coverage over it, so refuse and say how to
                # recover rather than silently producing an index that can never rank.
                current = await asyncio.to_thread(self._signature_matches)
                if not current:
                    return {
                        "ok": False,
                        "error": (
                            f"Stored vectors were not built for {self.signature()}. "
                            "Re-run with --rebuild to discard them and start again."
                        ),
                    }
            lexical = await asyncio.to_thread(self._sync_lexical_unlocked, 1_000_000)
            embedded = 0
            if provider is not None:
                while True:
                    # Commit each successful batch. Grouping eight requests before a
                    # commit loses their valid results if a later request fails.
                    rows = await asyncio.to_thread(self._pending_vectors, self.embedding.batch_size)
                    if not rows:
                        break
                    try:
                        vectors = await provider.embed([text for _, text in rows])
                    except EmbeddingError as exc:
                        raise EmbeddingError(
                            f"Chunks {rows[0][0]}–{rows[-1][0]}: {exc}. "
                            "Earlier batches remain saved; resume with 'lillia index' without --rebuild."
                        ) from exc
                    await asyncio.to_thread(
                        self._write_vectors, [(rows[i][0], v) for i, v in enumerate(vectors)]
                    )
                    embedded += len(vectors)
                    if on_progress:
                        on_progress(embedded)
            stats = await asyncio.to_thread(self._stats)
            return {"ok": True, "embedded": embedded, **lexical, **stats}

    def _signature_matches(self) -> bool:
        connection = self.connect()
        try:
            return self._check_signature(connection)
        finally:
            connection.close()

    def _pending_vectors(self, limit: int) -> list[tuple[int, str]]:
        connection = self.connect()
        try:
            return [(row["id"], row["text"]) for row in self._missing_vectors(connection, limit)]
        finally:
            connection.close()

    def _sync_lexical_unlocked(self, limit: int) -> dict:
        """Same as _sync_lexical but assumes the caller already holds the lock."""
        connection = self.connect()
        try:
            pending = self._discover(connection, limit)
            added = 0
            for artifact, sid in pending:
                added += self._ingest(connection, artifact, sid)
            connection.commit()
            return {"indexed_artifacts": len(pending), "indexed_chunks": added}
        finally:
            connection.close()

    # ---------------------------------------------------------------- tool

    def install(self, registry):
        from .tools import integer, string

        registry.add(
            "search_evidence",
            "Find saved context artifacts by meaning and by literal terms, returning ranked "
            "candidate locations for read_artifact. Use it to recover evidence whose exact "
            "path is no longer in context -- compaction preserves the files but usually not "
            "their paths. Each result carries path, char_offset and chunk_chars: READ IT with "
            "read_artifact before relying on it. Ranking measures topical similarity, not "
            "whether a passage answers the question. Absence from these results is not "
            "evidence that something was never recorded: before concluding that, check "
            "index_current, vector_coverage, semantic_chunks_scanned and query_terms, which "
            "say what was actually searched. This verifies nothing and establishes no "
            "completion.",
            {
                "query": string("What to recall, in plain words", 1000),
                "max_results": integer("Maximum candidates", 1, 50),
                "mode": {
                    "enum": ["auto", "lexical", "semantic"],
                    "description": (
                        "auto fuses both arms and falls back to lexical when the embedding "
                        "server is unavailable; lexical is literal-term FTS5 only"
                    ),
                },
                "this_session_only": {
                    "type": "boolean",
                    "description": "Restrict to this session's artifacts; default false",
                },
            },
            ["query"],
            "read",
            self.tool,
        )

    async def tool(self, query, max_results=None, mode="auto", this_session_only=False):
        if not query or not query.strip():
            raise ValueError("query must not be empty")
        if not self.retrieval.enabled:
            raise ValueError("retrieval.enabled is false in this configuration")
        limit = min(max_results or self.retrieval.max_results, self.retrieval.max_results)
        return await self.search(query, limit, mode, self.sid if this_session_only else None)


def describe(text: str) -> str:
    """Short label for CLI output; artifacts are frequently one long JSON line."""
    try:
        body = json.loads(text)
    except (ValueError, TypeError):
        return text[:80].replace("\n", " ")
    if isinstance(body, dict):
        return ", ".join(sorted(body)[:6])[:80]
    return text[:80].replace("\n", " ")
