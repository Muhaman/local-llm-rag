FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv venv /opt/venv && uv pip install --python /opt/venv/bin/python .

FROM python:3.12-slim
RUN useradd --uid 10001 --create-home app && mkdir -p /data && chown app /data
COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1
USER app
EXPOSE 8000
CMD ["uvicorn", "localrag.api:app", "--host", "0.0.0.0", "--port", "8000"]
