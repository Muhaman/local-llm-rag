from pathlib import Path

import pytest
from fakes import FakeEmbedder

from localrag.ingest import IngestReport, ingest_folder
from localrag.store import VectorStore


@pytest.fixture
def docs(tmp_path: Path) -> Path:
    folder = tmp_path / "docs"
    (folder / "sub").mkdir(parents=True)
    (folder / "a.md").write_text("# Alpha\nFirst document.")
    (folder / "sub" / "b.txt").write_text("Second document.")
    (folder / "ignored.png").write_bytes(b"\x89PNG")
    return folder


@pytest.fixture
def store(tmp_path: Path) -> VectorStore:
    return VectorStore(tmp_path / "index.db", dim=3)


async def test_ingest_lifecycle(docs: Path, store: VectorStore) -> None:
    embedder = FakeEmbedder()

    first = await ingest_folder(docs, store, embedder)
    assert first == IngestReport(added=2, updated=0, skipped=0, removed=0)
    assert store.list_paths() == {"a.md", "sub/b.txt"}

    second = await ingest_folder(docs, store, embedder)
    assert second == IngestReport(added=0, updated=0, skipped=2, removed=0)

    (docs / "a.md").write_text("# Alpha\nEdited.")
    third = await ingest_folder(docs, store, embedder)
    assert third == IngestReport(added=0, updated=1, skipped=1, removed=0)

    (docs / "sub" / "b.txt").unlink()
    fourth = await ingest_folder(docs, store, embedder)
    assert fourth == IngestReport(added=0, updated=0, skipped=1, removed=1)
    assert store.list_paths() == {"a.md"}


async def test_unchanged_files_are_not_re_embedded(docs: Path, store: VectorStore) -> None:
    embedder = FakeEmbedder()
    await ingest_folder(docs, store, embedder)
    calls = len(embedder.calls)

    await ingest_folder(docs, store, embedder)

    assert len(embedder.calls) == calls


async def test_missing_folder_does_not_wipe_the_index(
    docs: Path, store: VectorStore, tmp_path: Path
) -> None:
    await ingest_folder(docs, store, FakeEmbedder())

    with pytest.raises(NotADirectoryError):
        await ingest_folder(tmp_path / "nope", store, FakeEmbedder())

    assert store.list_paths() == {"a.md", "sub/b.txt"}
