from collections.abc import AsyncGenerator
from contextlib import aclosing

from localrag.ollama import ChatModel, Embedder
from localrag.store import Hit, VectorStore

NO_DOCUMENTS_REPLY = "No documents indexed yet."

SYSTEM_PROMPT = (
    "You answer questions using only the provided context excerpts. "
    "Cite the source of every claim as [path], using the path shown before each excerpt. "
    "If the context does not contain the answer, say that you do not know."
)


def build_messages(question: str, hits: list[Hit]) -> list[dict[str, str]]:
    context = "\n\n".join(f"[{hit.path}]\n{hit.text}" for hit in hits)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]


async def ask(
    question: str,
    store: VectorStore,
    embedder: Embedder,
    chat: ChatModel,
    top_k: int = 5,
) -> AsyncGenerator[str, None]:
    if not store.has_chunks():
        yield NO_DOCUMENTS_REPLY
        return
    (embedding,) = await embedder.embed([question])
    hits = store.search(embedding, top_k)
    if not hits:
        yield NO_DOCUMENTS_REPLY
        return
    async with aclosing(chat.stream(build_messages(question, hits))) as tokens:
        async for token in tokens:
            yield token
