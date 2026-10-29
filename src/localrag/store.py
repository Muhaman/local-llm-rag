import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

import sqlite_vec


@dataclass(frozen=True)
class Hit:
    text: str
    path: str
    distance: float


class VectorStore:
    """Chunks and their embeddings in one SQLite file, searched through a sqlite-vec table."""

    def __init__(self, db_path: Path, dim: int) -> None:
        self._dim = dim
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)
        self._conn.executescript(
            f"""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                path TEXT NOT NULL UNIQUE,
                sha256 TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                document_id INTEGER NOT NULL,
                text TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS chunks_document_id ON chunks(document_id);
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(embedding float[{int(dim)}]);
            """
        )

    def get_hash(self, path: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT sha256 FROM documents WHERE path = ?", (path,))
            found = row.fetchone()
        return str(found[0]) if found else None

    def list_paths(self) -> set[str]:
        with self._lock:
            rows = self._conn.execute("SELECT path FROM documents").fetchall()
        return {str(row[0]) for row in rows}

    def upsert_document(
        self, path: str, sha256: str, chunks: list[str], embeddings: list[list[float]]
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if any(len(embedding) != self._dim for embedding in embeddings):
            raise ValueError(f"embeddings must have {self._dim} dimensions")
        with self._lock, self._conn:
            self._delete(path)
            doc_id = self._insert_id(
                "INSERT INTO documents(path, sha256) VALUES (?, ?)", (path, sha256)
            )
            for text, embedding in zip(chunks, embeddings, strict=True):
                chunk_id = self._insert_id(
                    "INSERT INTO chunks(document_id, text) VALUES (?, ?)", (doc_id, text)
                )
                self._conn.execute(
                    "INSERT INTO vec_chunks(rowid, embedding) VALUES (?, ?)",
                    (chunk_id, sqlite_vec.serialize_float32(embedding)),
                )

    def delete_document(self, path: str) -> None:
        with self._lock, self._conn:
            self._delete(path)

    def search(self, embedding: list[float], k: int) -> list[Hit]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT c.text, d.path, v.distance
                FROM (
                    SELECT rowid, distance FROM vec_chunks WHERE embedding MATCH ? AND k = ?
                ) AS v
                JOIN chunks AS c ON c.id = v.rowid
                JOIN documents AS d ON d.id = c.document_id
                ORDER BY v.distance
                """,
                (sqlite_vec.serialize_float32(embedding), k),
            ).fetchall()
        return [Hit(text=row[0], path=row[1], distance=row[2]) for row in rows]

    def _insert_id(self, sql: str, params: tuple[object, ...]) -> int:
        row_id = self._conn.execute(sql, params).lastrowid
        if row_id is None:
            raise RuntimeError("insert did not produce a row id")
        return row_id

    def _delete(self, path: str) -> None:
        found = self._conn.execute("SELECT id FROM documents WHERE path = ?", (path,)).fetchone()
        if found is None:
            return
        doc_id = found[0]
        chunk_ids = self._conn.execute("SELECT id FROM chunks WHERE document_id = ?", (doc_id,))
        self._conn.executemany(
            "DELETE FROM vec_chunks WHERE rowid = ?", [(row[0],) for row in chunk_ids.fetchall()]
        )
        self._conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
        self._conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
