import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from fakes import FakeChat, FakeEmbedder, FakeHealth
from fastapi import FastAPI

from localrag.api import create_app, get_chat, get_embedder, get_health, get_settings, get_store
from localrag.config import Settings
from localrag.ollama import OllamaError
from localrag.store import VectorStore


def sse_events(body: str) -> list[tuple[str, dict[str, object]]]:
    events = []
    for block in body.strip().split("\n\n"):
        name = "message"
        data = ""
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line.removeprefix("event: ")
            elif line.startswith("data: "):
                data = line.removeprefix("data: ")
        events.append((name, json.loads(data)))
    return events


@pytest.fixture
def store(tmp_path: Path) -> VectorStore:
    store = VectorStore(tmp_path / "index.db", dim=3)
    store.upsert_document("a.md", "h", ["Alpha facts"], [[1.0, 1.0, 0.0]])
    return store


@pytest.fixture
def chat() -> FakeChat:
    return FakeChat(tokens=["Hel", "lo"])


@pytest.fixture
def app(tmp_path: Path, store: VectorStore, chat: FakeChat) -> FastAPI:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "n.md").write_text("note")
    app = create_app()
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_embedder] = lambda: FakeEmbedder()
    app.dependency_overrides[get_chat] = lambda: chat
    app.dependency_overrides[get_health] = lambda: FakeHealth(up=True)
    app.dependency_overrides[get_settings] = lambda: Settings(docs_dir=docs)
    return app


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as http:
        yield http


async def test_ask_streams_tokens(client: httpx.AsyncClient) -> None:
    response = await client.post("/v1/ask", json={"question": "What is alpha?"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert sse_events(response.text) == [
        ("message", {"token": "Hel"}),
        ("message", {"token": "lo"}),
        ("done", {}),
    ]


async def test_ask_reports_model_failure_as_error_event(
    client: httpx.AsyncClient, chat: FakeChat
) -> None:
    chat.error = OllamaError("model not found")

    response = await client.post("/v1/ask", json={"question": "hi"})

    events = sse_events(response.text)
    assert events[-1] == (
        "error",
        {"error": {"code": "ollama_error", "message": "model not found"}},
    )


@pytest.mark.parametrize("payload", [{"question": ""}, {"question": "x" * 2001}, {}])
async def test_ask_rejects_invalid_questions(
    client: httpx.AsyncClient, payload: dict[str, object]
) -> None:
    response = await client.post("/v1/ask", json=payload)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


async def test_healthz_ok(client: httpx.AsyncClient) -> None:
    response = await client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_healthz_is_503_when_ollama_is_down(
    app: FastAPI, client: httpx.AsyncClient
) -> None:
    app.dependency_overrides[get_health] = lambda: FakeHealth(up=False)

    response = await client.get("/healthz")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "ollama_unreachable"


async def test_ingest_returns_report(client: httpx.AsyncClient) -> None:
    response = await client.post("/v1/ingest")

    assert response.status_code == 200
    assert response.json() == {"added": 1, "updated": 0, "skipped": 0, "removed": 1}


async def test_index_page_is_served(client: httpx.AsyncClient) -> None:
    response = await client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


async def test_unknown_route_uses_error_shape(client: httpx.AsyncClient) -> None:
    response = await client.get("/nope")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
