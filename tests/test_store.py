from pathlib import Path

import pytest

from localrag.store import VectorStore


@pytest.fixture
def store(tmp_path: Path) -> VectorStore:
    store = VectorStore(tmp_path / "index.db", dim=3)
    store.upsert_document("a.md", "h-a", ["alpha"], [[1.0, 0.0, 0.0]])
    store.upsert_document("b.md", "h-b", ["beta"], [[0.0, 1.0, 0.0]])
    store.upsert_document("c.md", "h-c", ["gamma"], [[0.0, 0.0, 1.0]])
    return store


def test_search_returns_nearest_first(store: VectorStore) -> None:
    hits = store.search([1.0, 0.0, 0.0], k=3)

    assert len(hits) == 3
    assert hits[0].path == "a.md"
    assert hits[0].text == "alpha"
    assert hits[0].distance == pytest.approx(0.0)
    assert hits[0].distance < hits[1].distance


def test_search_respects_k(store: VectorStore) -> None:
    assert len(store.search([1.0, 0.0, 0.0], k=2)) == 2


def test_hash_and_paths(store: VectorStore) -> None:
    assert store.get_hash("a.md") == "h-a"
    assert store.get_hash("missing.md") is None
    assert store.list_paths() == {"a.md", "b.md", "c.md"}


def test_delete_removes_rows(store: VectorStore) -> None:
    store.delete_document("b.md")

    assert store.list_paths() == {"a.md", "c.md"}
    assert store.get_hash("b.md") is None
    assert "b.md" not in {hit.path for hit in store.search([0.0, 1.0, 0.0], k=5)}


def test_upsert_replaces_existing_chunks(store: VectorStore) -> None:
    store.upsert_document("a.md", "h-a2", ["alpha v2"], [[0.0, 0.0, 1.0]])

    texts = {hit.text for hit in store.search([1.0, 0.0, 0.0], k=10)}
    assert "alpha" not in texts
    assert "alpha v2" in texts
    assert store.get_hash("a.md") == "h-a2"


def test_search_on_empty_store_returns_nothing(tmp_path: Path) -> None:
    assert VectorStore(tmp_path / "empty.db", dim=3).search([1.0, 0.0, 0.0], k=5) == []


def test_mismatched_chunk_and_embedding_counts_are_rejected(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "x.db", dim=3)

    with pytest.raises(ValueError):
        store.upsert_document("a.md", "h", ["one", "two"], [[1.0, 0.0, 0.0]])


def test_has_chunks(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "h.db", dim=3)
    assert not store.has_chunks()

    store.upsert_document("blank.md", "h", [], [])
    assert not store.has_chunks()

    store.upsert_document("a.md", "h", ["alpha"], [[1.0, 0.0, 0.0]])
    assert store.has_chunks()
