from pathlib import Path

import pytest

from fakes import FakeChat, FakeEmbedder
from localrag.rag import NO_DOCUMENTS_REPLY, ask
from localrag.store import VectorStore


async def collect(question: str, store: VectorStore, chat: FakeChat) -> str:
    return "".join([token async for token in ask(question, store, FakeEmbedder(), chat)])


@pytest.fixture
def store(tmp_path: Path) -> VectorStore:
    return VectorStore(tmp_path / "index.db", dim=3)


async def test_empty_index_replies_without_calling_the_model(store: VectorStore) -> None:
    chat = FakeChat()
    embedder = FakeEmbedder()

    tokens = [token async for token in ask("anything?", store, embedder, chat)]

    assert tokens == ["No documents indexed yet."]
    assert NO_DOCUMENTS_REPLY == "No documents indexed yet."
    assert chat.calls == []
    assert embedder.calls == []


async def test_prompt_cites_sources_by_path(store: VectorStore) -> None:
    store.upsert_document("a.md", "h", ["Alpha facts"], [[1.0, 1.0, 0.0]])
    chat = FakeChat(tokens=["Answer ", "[a.md]"])

    answer = await collect("What is alpha?", store, chat)

    assert answer == "Answer [a.md]"
    (messages,) = chat.calls
    prompt = "\n".join(message["content"] for message in messages)
    assert "[a.md]" in prompt
    assert "Alpha facts" in prompt
    assert "What is alpha?" in prompt
