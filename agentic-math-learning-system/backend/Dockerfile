# backend/Dockerfile
# ─────────────────────────────────────────────────────────────────────
# Multi-stage-friendly single-stage build for the FastAPI backend.
#
# Build strategy:
#   1. Start from python:3.11-slim to keep the image small.
#   2. Copy requirements.txt first so Docker can cache the pip install
#      layer — it only re-runs when requirements.txt changes, not on
#      every source code edit.
#   3. Copy the application source after dependencies are installed.
#   4. Run Uvicorn as a non-root user for basic container security.
# ─────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────
# gcc and libpq-dev are needed to compile psycopg2 from source if the
# binary wheel is unavailable for the target platform.
# curl is included for Docker health-check probes.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────
# Copy requirements first to leverage Docker layer caching.
# The pip install layer is only invalidated when requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Application source ────────────────────────────────────────────────
# Copy the rest of the backend source after dependencies are installed.
COPY . .

# ── Non-root user ─────────────────────────────────────────────────────
# Running as a non-root user reduces the blast radius of a container
# escape vulnerability.
RUN adduser --disabled-password --gecos "" appuser
USER appuser

# ── Port ──────────────────────────────────────────────────────────────
# Document the port the application listens on.
# docker-compose.yml maps this to the host as 8080:8080.
EXPOSE 8080

# ── Health check ──────────────────────────────────────────────────────
# Docker will mark the container unhealthy if /health stops responding.
# This lets docker-compose wait for the backend before the frontend starts.
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────
# Uvicorn serves the FastAPI app defined in main.py.
# --host 0.0.0.0 binds to all interfaces so the container is reachable
# from the Docker Compose network and from the host machine.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
