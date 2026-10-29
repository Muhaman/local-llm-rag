import json
from collections.abc import AsyncGenerator
from typing import Protocol

import httpx


class OllamaError(Exception):
    """Ollama is unreachable or returned an error."""


class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class ChatModel(Protocol):
    def stream(self, messages: list[dict[str, str]]) -> AsyncGenerator[str, None]: ...


class HealthCheck(Protocol):
    async def ping(self) -> bool: ...


class OllamaClient:
    """Embedder, ChatModel and HealthCheck backed by Ollama's REST API."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        chat_model: str,
        embed_model: str,
        batch_size: int = 32,
    ) -> None:
        self._client = client
        self._chat_model = chat_model
        self._embed_model = embed_model
        self._batch_size = batch_size

    async def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            try:
                response = await self._client.post(
                    "/api/embed", json={"model": self._embed_model, "input": batch}
                )
            except httpx.HTTPError as exc:
                raise OllamaError(f"Ollama request failed: {exc!r}") from exc
            _raise_for_status(response)
            embeddings.extend(response.json()["embeddings"])
        return embeddings

    async def stream(self, messages: list[dict[str, str]]) -> AsyncGenerator[str, None]:
        payload = {"model": self._chat_model, "messages": messages, "stream": True}
        try:
            async with self._client.stream("POST", "/api/chat", json=payload) as response:
                if response.is_error:
                    await response.aread()
                _raise_for_status(response)
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if "error" in chunk:
                        raise OllamaError(str(chunk["error"]))
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        yield content
                    if chunk.get("done"):
                        return
                raise OllamaError("Ollama stream ended before completion")
        except httpx.HTTPError as exc:
            raise OllamaError(f"Ollama request failed: {exc!r}") from exc

    async def ping(self) -> bool:
        try:
            response = await self._client.get("/api/tags", timeout=3.0)
        except httpx.HTTPError:
            return False
        return response.is_success


def _raise_for_status(response: httpx.Response) -> None:
    if response.is_error:
        raise OllamaError(f"Ollama returned {response.status_code}: {response.text}")
