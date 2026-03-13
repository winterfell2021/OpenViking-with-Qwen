FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml .
RUN uv sync

COPY main.py .

# Download model from ModelScope during build
ENV MODELSCOPE_CACHE=/app/models
RUN uv run python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3-Embedding-0.6B', cache_dir='/app/models')"

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
