# Multi-stage Dockerfile for NirnayaX API service
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy wheel configuration and dependencies
COPY pyproject.toml README.md /app/
COPY src/ /app/src/

# Install application with optional extras
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir ".[ml,retrieval,api,ui]"

# Final runtime image
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed site-packages and binaries from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy project source and data files
COPY pyproject.toml README.md /app/
COPY src/ /app/src/
COPY data/ /app/data/
RUN mkdir -p /app/models

ENV PYTHONPATH="/app/src"
ENV PYTHONUNBUFFERED="1"

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')" || exit 1

CMD ["uvicorn", "nirnayax.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
