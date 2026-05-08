# Agentic Math Learning System

A monorepo demo application that teaches students how to build **Agentic RAG** systems and **stateful agentic workflows**. It exposes three distinct agentic paths, each demonstrating a different combination of LangGraph orchestration, ChromaDB vector search, LlamaParse PDF extraction, and a local Ollama LLM.

| Path | Name | What it demonstrates |
|------|------|----------------------|
| A | Adaptive Tutor | RAG + CoT prompting + conditional routing |
| B | Practice Agent | Stateful MCQ generation + Deep-Dive tool |
| C | Doc-to-Concept | PDF ingestion + formula extraction + synthetic examples |

---

## Architecture at a Glance

```
Frontend (React + Tailwind, :3000)
        │
        ▼
Backend (FastAPI, :8080)
  ├── Path A — adaptive_tutor/
  ├── Path B — practice_agent/
  ├── Path C — doc_to_concept/
  └── Shared — embeddings/ | reranker/ | prompts.py | llm_wrapper.py
        │                    │
        ▼                    ▼
   ChromaDB (:8000)    PostgreSQL (:5432)

Ollama (host machine, :11434) — NOT a compose service; needs GPU access
```

---

## Quick Start

### Prerequisites

1. **Docker & Docker Compose** — [Install Docker Desktop](https://docs.docker.com/get-docker/)
2. **Ollama** — running on your host machine with `llama3:latest` pulled (Ollama is **not** a Docker Compose service — it runs directly on the host so it can access the GPU):
   ```bash
   # Install Ollama: https://ollama.com
   ollama pull llama3:latest
   ollama serve          # starts on :11434 by default
   ```
   In your `.env` file set:
   ```
   OLLAMA_BASE_URL=http://host.docker.internal:11434
   ```
   This address lets containers reach the host's Ollama process. On Linux, use the host's LAN IP or `172.17.0.1` if `host.docker.internal` is not available.
3. **LlamaParse API key** — sign up at [cloud.llamaindex.ai](https://cloud.llamaindex.ai) (free tier available)

### Start the Stack

```bash
# 1. Clone the repo and enter the project directory
git clone <repo-url>
cd agentic-math-learning-system

# 2. Copy the example env file and fill in your values
cp .env.example .env
#    → Set LLAMA_PARSE_API_KEY to your key
#    → Adjust OLLAMA_BASE_URL if Ollama is not on localhost

# 3. Start all services
docker-compose up --build

# Services will be available at:
#   Frontend  → http://localhost:3000
#   Backend   → http://localhost:8080
#   ChromaDB  → http://localhost:8000
#   PostgreSQL→ localhost:5432
```

### Stop the Stack

```bash
docker-compose down          # stop containers, keep volumes
docker-compose down -v       # stop containers AND delete all data volumes
```

---

## Project Structure

```
agentic-math-learning-system/
├── docker-compose.yml          # All services defined here
├── .env.example                # Template — copy to .env
├── README.md                   # This file
│
├── backend/                    # FastAPI Python application
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py                 # App factory; mounts all routers
│   ├── config.py               # LLMConfig dataclass + named constants
│   ├── llm_wrapper.py          # Single Ollama call wrapper
│   ├── prompts.py              # All prompt templates as named constants
│   ├── database.py             # SQLAlchemy engine + session factory
│   ├── embeddings/             # parse → chunk → embed → store pipeline
│   ├── reranker/               # Cross-encoder reranker module
│   ├── adaptive_tutor/         # Path A — LangGraph RAG chat agent
│   ├── practice_agent/         # Path B — stateful MCQ agent
│   ├── doc_to_concept/         # Path C — PDF ingestion agent
│   └── models/                 # Pydantic API models + SQLAlchemy ORM models
│
└── frontend/                   # React + Tailwind SPA
    ├── Dockerfile
    ├── package.json
    ├── vite.config.ts
    └── src/
        ├── App.tsx             # Tab navigation shell
        ├── api/client.ts       # Typed API client
        ├── components/         # Shared UI components
        └── pages/              # One page per agentic path
```

---

## Environment Variables Reference

| Variable | Description |
|----------|-------------|
| `LLAMA_PARSE_API_KEY` | LlamaParse cloud API key for PDF extraction |
| `OLLAMA_BASE_URL` | Base URL of the Ollama server (host machine) |
| `POSTGRES_DSN` | Full SQLAlchemy connection string for PostgreSQL |
| `POSTGRES_USER` | PostgreSQL username (used by the postgres service) |
| `POSTGRES_PASSWORD` | PostgreSQL password |
| `POSTGRES_DB` | PostgreSQL database name |
| `CHROMA_HOST` | ChromaDB hostname (use service name `chromadb` in compose) |
| `CHROMA_PORT` | ChromaDB port (default `8000`) |

---

## Learning Objectives

After exploring this codebase you will understand:

- How to structure a **LangGraph state machine** with conditional edges
- How to implement a **two-stage RAG pipeline** (vector search → cross-encoder reranking)
- How to write **XML-delimited prompts** with Chain-of-Thought instructions
- How to register **Python functions as LangGraph tools** with informative docstrings
- How to use **LlamaParse** to extract math-preserving Markdown from PDFs
- How to wire a **FastAPI + React** full-stack app with Docker Compose
