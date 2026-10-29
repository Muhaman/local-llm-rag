import asyncio
import hashlib
from dataclasses import dataclass
from pathlib import Path

import structlog

from localrag.chunk import chunk_text
from localrag.loaders import is_supported, load_file
from localrag.ollama import Embedder
from localrag.store import VectorStore

log = structlog.get_logger()


@dataclass(frozen=True)
class IngestReport:
    added: int
    updated: int
    skipped: int
    removed: int


async def ingest_folder(folder: Path, store: VectorStore, embedder: Embedder) -> IngestReport:
    files = await asyncio.to_thread(_scan, folder)
    added = updated = skipped = 0
    for rel_path, path in files.items():
        digest = await asyncio.to_thread(_sha256, path)
        known = store.get_hash(rel_path)
        if known == digest:
            skipped += 1
            continue
        text = await asyncio.to_thread(load_file, path)
        chunks = chunk_text(text)
        embeddings = await embedder.embed(chunks) if chunks else []
        store.upsert_document(rel_path, digest, chunks, embeddings)
        if known is None:
            added += 1
        else:
            updated += 1
        log.info("document_indexed", path=rel_path, chunks=len(chunks))
    vanished = store.list_paths() - files.keys()
    for rel_path in vanished:
        store.delete_document(rel_path)
        log.info("document_removed", path=rel_path)
    return IngestReport(added=added, updated=updated, skipped=skipped, removed=len(vanished))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scan(folder: Path) -> dict[str, Path]:
    if not folder.is_dir():
        raise NotADirectoryError(f"Docs folder not found: {folder}")
    return {
        path.relative_to(folder).as_posix(): path
        for path in sorted(folder.rglob("*"))
        if path.is_file() and is_supported(path)
    }
