"""Small, resumable indexing batches alongside an active agent turn."""

from __future__ import annotations

import asyncio


async def finish_thread(function, *args):
    """Do not release a writer lock while cancelled SQLite work still uses it."""
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                pass
        task.result()
        raise


class LiveIndex:
    def __init__(self, index):
        self.index = index
        self.task = None
        self.failure = None

    def start(self):
        if self.task is None and self.index.sid and self.index.retrieval.enabled:
            self.task = asyncio.create_task(self.run(), name=f"lillia-index-{self.index.sid}")

    async def stop(self):
        if self.task is not None:
            task, self.task = self.task, None
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    def pending(self):
        index = self.index
        connection = index.connect()
        try:
            known = {
                row[0]
                for row in connection.execute("SELECT ref FROM sources WHERE session_id=?", (index.sid,))
            }
            root = index.artifact_root() / index.sid
            artifacts = (
                []
                if not root.is_dir()
                else [
                    path
                    for path in root.iterdir()
                    if not path.name.startswith(".") and path.is_file() and str(path) not in known
                ]
            )
            # New session evidence has priority over a backlog of old archives.
            artifacts.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
            # Within this slice, give the newest artifact the largest chunk IDs
            # so both pending-vector selection and recall's capped scan favor it.
            for artifact in reversed(artifacts[:64]):
                index._ingest(connection, artifact, index.sid)
            connection.commit()
            if not index.embedding.enabled:
                return [], None, min(len(artifacts), 64)
            if not index._check_signature(connection):
                return [], "Embedding signature changed; run lillia index --rebuild.", min(len(artifacts), 64)
            rows = connection.execute(
                "SELECT c.id, c.text FROM chunks c LEFT JOIN vectors v ON v.chunk_id=c.id "
                "WHERE c.session_id=? AND v.chunk_id IS NULL "
                "ORDER BY (substr(c.text, 1, 19) = 'Historical attempt:') DESC, c.id DESC LIMIT ?",
                (index.sid, index.retrieval.index_batch_chunks),
            ).fetchall()
            return [(row["id"], row["text"]) for row in rows], None, min(len(artifacts), 64)
        finally:
            connection.close()

    async def cycle(self):
        index = self.index
        # The same lock protects manual builds, search backfills, and other workers.
        # Never queue behind another writer or launch competing document batches.
        with index.exclusive() as held:
            if not held:
                return {"locked": True, "embedded": 0, "indexed_artifacts": 0}
            rows, error, added = await finish_thread(self.pending)
            if error:
                raise RuntimeError(error)
            if rows:
                vectors = await index.provider().embed([text for _, text in rows])
                await finish_thread(
                    index._write_vectors,
                    [(row[0], vector) for row, vector in zip(rows, vectors, strict=True)],
                )
            return {"embedded": len(rows), "indexed_artifacts": added}

    async def run(self):
        index = self.index
        while True:
            delay = index.retrieval.index_interval_seconds
            try:
                report = await self.cycle()
                if self.failure and not report.get("locked"):
                    index.store.event(index.sid, "continuous_index_recovered", {})
                    self.failure = None
                if report["embedded"] or report["indexed_artifacts"]:
                    index.store.event(index.sid, "continuous_index_progress", report)
            except Exception as exc:
                failure = f"{type(exc).__name__}: {exc}"[:500]
                if failure != self.failure:
                    self.failure = failure
                    index.store.event(index.sid, "continuous_index_unavailable", {"detail": failure})
                delay = max(delay, 30.0)
            await asyncio.sleep(delay)
