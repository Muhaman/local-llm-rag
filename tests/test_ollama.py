import json

import httpx
import pytest

from localrag.ollama import OllamaClient, OllamaError


def make_client(transport: httpx.MockTransport, batch_size: int = 32) -> OllamaClient:
    http = httpx.AsyncClient(base_url="http://ollama", transport=transport)
    return OllamaClient(http, chat_model="chat-m", embed_model="embed-m", batch_size=batch_size)


async def drain(client: OllamaClient, messages: list[dict[str, str]] | None = None) -> list[str]:
    return [token async for token in client.stream(messages or [])]


def ndjson(*objects: dict[str, object]) -> bytes:
    return "".join(json.dumps(obj) + "\n" for obj in objects).encode()


async def test_embed_sends_model_and_input_and_batches() -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/embed"
        body = json.loads(request.content)
        bodies.append(body)
        return httpx.Response(200, json={"embeddings": [[float(len(t))] for t in body["input"]]})

    client = make_client(httpx.MockTransport(handler), batch_size=2)

    result = await client.embed(["a", "bb", "ccc"])

    assert result == [[1.0], [2.0], [3.0]]
    assert bodies == [
        {"model": "embed-m", "input": ["a", "bb"]},
        {"model": "embed-m", "input": ["ccc"]},
    ]


async def test_embed_model_not_pulled_raises_with_ollama_message() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model 'embed-m' not found"})

    client = make_client(httpx.MockTransport(handler))

    with pytest.raises(OllamaError, match="model 'embed-m' not found"):
        await client.embed(["a"])


async def test_chat_streams_content_until_done() -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            content=ndjson(
                {"message": {"role": "assistant", "content": "Hel"}, "done": False},
                {"message": {"role": "assistant", "content": "lo"}, "done": False},
                {"message": {"role": "assistant", "content": ""}, "done": True},
            ),
        )

    client = make_client(httpx.MockTransport(handler))
    messages = [{"role": "user", "content": "hi"}]

    tokens = await drain(client, messages)

    assert tokens == ["Hel", "lo"]
    assert seen == [{"model": "chat-m", "messages": messages, "stream": True}]


async def test_chat_in_stream_error_raises() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=ndjson(
                {"message": {"content": "partial"}, "done": False},
                {"error": "model runner crashed"},
            ),
        )

    client = make_client(httpx.MockTransport(handler))
    received: list[str] = []

    with pytest.raises(OllamaError, match="model runner crashed"):
        async for token in client.stream([]):
            received.append(token)

    assert received == ["partial"]


async def test_chat_model_not_pulled_raises_with_ollama_message() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model 'chat-m' not found"})

    client = make_client(httpx.MockTransport(handler))

    with pytest.raises(OllamaError, match="model 'chat-m' not found"):
        await drain(client)


async def test_chat_stream_truncated_before_done_raises() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=ndjson({"message": {"content": "Hel"}, "done": False}))

    client = make_client(httpx.MockTransport(handler))

    with pytest.raises(OllamaError, match="ended before completion"):
        await drain(client)


async def test_connection_errors_become_ollama_errors() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    client = make_client(httpx.MockTransport(handler))

    with pytest.raises(OllamaError):
        await client.embed(["a"])
    with pytest.raises(OllamaError):
        await drain(client)


async def test_ping_reflects_reachability() -> None:
    def ok(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": []})

    def down(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    def failing(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    assert await make_client(httpx.MockTransport(ok)).ping() is True
    assert await make_client(httpx.MockTransport(down)).ping() is False
    assert await make_client(httpx.MockTransport(failing)).ping() is False
