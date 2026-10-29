import json
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing, asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import httpx
import structlog
from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, StringConstraints
from starlette.exceptions import HTTPException as StarletteHTTPException

from localrag.config import Settings, configure_logging, get_settings
from localrag.ingest import IngestReport, ingest_folder
from localrag.ollama import ChatModel, Embedder, HealthCheck, OllamaClient, OllamaError
from localrag.rag import ask
from localrag.store import VectorStore

log = structlog.get_logger()

INDEX_HTML = Path(__file__).parent / "static" / "index.html"


class ApiError(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class AskRequest(BaseModel):
    question: Question


def error_body(code: str, message: str) -> dict[str, dict[str, str]]:
    return {"error": {"code": code, "message": message}}


def get_store(request: Request) -> VectorStore:
    store: VectorStore = request.app.state.store
    return store


def get_ollama(request: Request) -> OllamaClient:
    ollama: OllamaClient = request.app.state.ollama
    return ollama


def get_embedder(ollama: Annotated[OllamaClient, Depends(get_ollama)]) -> Embedder:
    return ollama


def get_chat(ollama: Annotated[OllamaClient, Depends(get_ollama)]) -> ChatModel:
    return ollama


def get_health(ollama: Annotated[OllamaClient, Depends(get_ollama)]) -> HealthCheck:
    return ollama


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(
        base_url=settings.ollama_url, timeout=httpx.Timeout(300.0, connect=5.0)
    ) as http:
        app.state.ollama = OllamaClient(http, settings.chat_model, settings.embed_model)
        app.state.store = VectorStore(settings.db_path, settings.embed_dim)
        yield


def sse(data: object, event: str | None = None) -> str:
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {json.dumps(data)}\n\n"


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(title="local-llm-rag", lifespan=lifespan)

    @app.exception_handler(ApiError)
    async def api_error_handler(_request: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(error_body(exc.code, exc.message), status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
        message = "; ".join(
            f"{'.'.join(str(part) for part in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        return JSONResponse(error_body("validation_error", message), status_code=422)

    @app.exception_handler(StarletteHTTPException)
    async def http_handler(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = "not_found" if exc.status_code == 404 else "http_error"
        return JSONResponse(error_body(code, str(exc.detail)), status_code=exc.status_code)

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(INDEX_HTML, media_type="text/html")

    @app.get("/healthz")
    async def healthz(health: Annotated[HealthCheck, Depends(get_health)]) -> dict[str, str]:
        if not await health.ping():
            raise ApiError(503, "ollama_unreachable", "Ollama is not reachable.")
        return {"status": "ok"}

    @app.post("/v1/ingest")
    async def ingest(
        settings: Annotated[Settings, Depends(get_settings)],
        store: Annotated[VectorStore, Depends(get_store)],
        embedder: Annotated[Embedder, Depends(get_embedder)],
    ) -> dict[str, int]:
        try:
            report: IngestReport = await ingest_folder(settings.docs_dir, store, embedder)
        except NotADirectoryError as exc:
            raise ApiError(400, "docs_dir_missing", str(exc)) from exc
        except OllamaError as exc:
            raise ApiError(502, "ollama_error", str(exc)) from exc
        return asdict(report)

    @app.post("/v1/ask")
    async def ask_route(
        body: AskRequest,
        settings: Annotated[Settings, Depends(get_settings)],
        store: Annotated[VectorStore, Depends(get_store)],
        embedder: Annotated[Embedder, Depends(get_embedder)],
        chat: Annotated[ChatModel, Depends(get_chat)],
    ) -> StreamingResponse:
        async def events() -> AsyncGenerator[str, None]:
            # aclosing() propagates a client disconnect (task cancellation) down to the
            # Ollama stream, which closes its httpx response and stops generation.
            answer = ask(body.question, store, embedder, chat, settings.top_k)
            try:
                async with aclosing(answer) as tokens:
                    async for token in tokens:
                        yield sse({"token": token})
            except OllamaError as exc:
                log.warning("ask_failed", error=str(exc))
                yield sse(error_body("ollama_error", str(exc)), event="error")
                return
            yield sse({}, event="done")

        return StreamingResponse(
            events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
        )

    return app


app = create_app()
