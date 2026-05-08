# main.py
# ─────────────────────────────────────────────────────────────────────
# FastAPI application factory for the Agentic Math Learning System.
#
# This file is the entry point for the backend service.  It wires
# together three concerns students should understand:
#
#   1. CORS middleware  — allows the React frontend (localhost:3000) to
#                         call the API from a browser without CORS errors.
#   2. Router mounting  — each agentic path lives in its own router
#                         module; mounting them here keeps main.py thin.
#   3. Startup event    — Base.metadata.create_all() runs once when the
#                         server starts, ensuring all PostgreSQL tables
#                         exist before the first request arrives.
# ─────────────────────────────────────────────────────────────────────
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database import create_tables

# ── Agentic path routers ──────────────────────────────────────────────
# Each router is defined in its own module so students can study one
# agentic path in isolation without reading unrelated code.
from adaptive_tutor.router import router as tutor_router
from practice_agent.router import router as practice_router
from doc_to_concept.router import router as docs_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """
    Construct and configure the FastAPI application.

    Separating app creation into a factory function makes the app
    easier to test (instantiate a fresh app per test) and keeps the
    module-level namespace clean.

    Returns:
        A fully configured FastAPI instance ready to be served by Uvicorn.
    """
    app = FastAPI(
        title="Agentic Math Learning System",
        description=(
            "A demo backend exposing three agentic paths: "
            "Adaptive Tutor (Path A), Practice Agent (Path B), "
            "and Doc-to-Concept Agent (Path C)."
        ),
        version="1.0.0",
    )

    # ── CORS middleware ───────────────────────────────────────────────
    # Allow the React dev server (http://localhost:3000) to make
    # cross-origin requests.  In production you would restrict
    # allow_origins to your actual frontend domain.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],   # GET, POST, OPTIONS, etc.
        allow_headers=["*"],   # Content-Type, Authorization, etc.
    )

    # ── Router mounting ───────────────────────────────────────────────
    # Each router carries its own prefix (/api/tutor, /api/practice,
    # /api/docs) so the URL structure is self-documenting.
    app.include_router(tutor_router)     # Path A — /api/tutor/*
    app.include_router(practice_router)  # Path B — /api/practice/*
    app.include_router(docs_router)      # Path C — /api/docs/*

    # ── Startup event ─────────────────────────────────────────────────
    @app.on_event("startup")
    async def on_startup() -> None:
        """
        Run once when Uvicorn starts the application.

        Creates all PostgreSQL tables defined by the ORM models so the
        database schema is always in sync with the code without requiring
        a separate migration step for this demo.
        """
        logger.info("Starting Agentic Math Learning System backend…")
        create_tables()
        logger.info("Backend startup complete — all tables ready")

    @app.get("/health", tags=["health"])
    async def health_check() -> dict:
        """Simple liveness probe used by Docker Compose health checks."""
        return {"status": "ok"}

    return app


# ── Application instance ──────────────────────────────────────────────
# Uvicorn imports `app` from this module:
#   uvicorn main:app --host 0.0.0.0 --port 8080
app = create_app()
