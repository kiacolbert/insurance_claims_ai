
FROM python:3.11-slim AS base

# Install system dependencies
# curl: needed for HEALTHCHECK
# build-essential: needed for chromadb native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Dependencies layer (cached unless requirements.txt changes) ──
FROM base AS deps

COPY requirements.txt .

# Install all deps EXCEPT streamlit (backend container doesn't need it)
RUN pip install --no-cache-dir -r requirements.txt

# ── Final image ───────────────────────────────────────────────
FROM base AS production

WORKDIR /app

# Copy installed packages from deps stage
COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy application source
COPY src/ ./src/
COPY pyproject.toml .

# Install the package in editable mode
RUN pip install --no-cache-dir -e .

# Copy data directory
RUN mkdir -p data/chroma_db

# Security: non-root user
RUN useradd --create-home --shell /bin/bash --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Health check — ECS/App Runner uses this
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

COPY entrypoint.sh .

ENTRYPOINT ["./entrypoint.sh"]