from collections.abc import AsyncGenerator


class FakeEmbedder:
    """Deterministic 3-d embeddings so tests never need Ollama."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[float(len(text) % 7 + 1), 1.0, 0.0] for text in texts]


class FakeChat:
    def __init__(self, tokens: list[str] | None = None, error: Exception | None = None) -> None:
        self.tokens = tokens if tokens is not None else ["Hello", " world"]
        self.error = error
        self.calls: list[list[dict[str, str]]] = []

    async def stream(self, messages: list[dict[str, str]]) -> AsyncGenerator[str, None]:
        self.calls.append(messages)
        for token in self.tokens:
            yield token
        if self.error is not None:
            raise self.error


class FakeHealth:
    def __init__(self, up: bool) -> None:
        self.up = up

    async def ping(self) -> bool:
        return self.up
