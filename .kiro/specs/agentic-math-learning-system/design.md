# Design Document — Agentic Math Learning System

## Overview

The Agentic Math Learning System is a monorepo demo application that teaches students how to build Agentic RAG systems and stateful agentic workflows. It exposes three distinct agentic paths — an Adaptive Tutor (Path A), a Practice Agent (Path B), and a Doc-to-Concept Agent (Path C) — each demonstrating a different combination of LangGraph orchestration, ChromaDB vector search, LlamaParse PDF extraction, and a local Ollama LLM.

The system is intentionally "simple by design": every architectural decision (prompt structure, tool definition, embedding pipeline, reranking) is visible and educational. The backend is FastAPI (Python 3.11+), the frontend is React + Tailwind CSS, and the entire stack is deployed via Docker Compose.

### Design Goals

- **Educational clarity**: Every module boundary, class, and non-trivial function carries an inline comment explaining its purpose.
- **Single-responsibility modules**: Each concern (embeddings, reranking, prompts, tools, LLM config) lives in its own file so students can read one file to understand one concept.
- **Reproducibility**: docker-compose up starts the full stack; no manual setup beyond copying .env.example.
- **Traceability**: Structured logging at INFO and DEBUG levels lets students observe every pipeline step at runtime.

---

## Architecture

### High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Docker Compose Network                        │
│                                                                     │
│  ┌──────────────┐    ┌──────────────────────────────────────────┐  │
│  │   Frontend   │    │              Backend (FastAPI)            │  │
│  │ React+Tailwind│───▶│                                          │  │
│  │  :3000       │    │  ┌──────────┐  ┌──────────┐  ┌────────┐ │  │
│  └──────────────┘    │  │ Path A   │  │ Path B   │  │ Path C │ │  │
│                      │  │ Adaptive │  │ Practice │  │ Doc-to │ │  │
│                      │  │ Tutor    │  │ Agent    │  │Concept │ │  │
│                      │  └────┬─────┘  └────┬─────┘  └───┬────┘ │  │
│                      │       │              │             │      │  │
│                      │  ┌────▼──────────────▼─────────────▼───┐ │  │
│                      │  │         Shared Utilities             │ │  │
│                      │  │  embeddings | reranker | database    │ │  │
│                      │  │  llm_wrapper | prompts | tools       │ │  │
│                      │  └────┬──────────────┬──────────────────┘ │  │
│                      └───────┼──────────────┼────────────────────┘  │
│                              │              │                        │
│  ┌───────────────┐    ┌──────▼──────┐  ┌───▼──────────┐            │
│  │   Ollama LLM  │    │  ChromaDB   │  │  PostgreSQL  │            │
│  │ llama3:latest │    │  :8000      │  │  :5432       │            │
│  │  :11434       │    └─────────────┘  └──────────────┘            │
│  └───────────────┘                                                  │
└─────────────────────────────────────────────────────────────────────┘
```

### Request Flow — Path A (Adaptive Tutor)

```
POST /api/tutor/chat
        │
        ▼
  FastAPI Route Handler
        │
        ▼
  LangGraph Graph (adaptive_tutor/graph.py)
        │
        ▼
  Router Node ──── classify query ────▶ "direct_answer" ──▶ LLM directly
        │
        │ "knowledge_required"
        ▼
  Retrieve Node (ChromaDB top-K)
        │
        ▼
  Rerank Node (BAAI/bge-reranker-v2-m3, top-N)
        │
        ▼
  Generate Node (Ollama LLM + CoT prompt)
        │
        ▼
  Persist Node (PostgreSQL tutor_sessions)
        │
        ▼
  JSON Response
```

### Request Flow — Path B (Practice Agent)

```
POST /api/practice/start          POST /api/practice/answer
        │                                  │
        ▼                                  ▼
  LangGraph State Machine          LangGraph State Machine
  State: "generate_question"       State: "evaluate_answer"
        │                                  │
        ▼                          ┌───────┴────────┐
  MCQ Generation Node              │                │
  (Ollama, temp≤0.2)          correct?=True    correct?=False
        │                          │                │
        ▼                          ▼                ▼
  Persist to DB            "next_question"   Deep_Dive_Tool
                                             (detailed explanation)
                                                     │
                                                     ▼
                                               State: "retry"
```

### Request Flow — Path C (Doc-to-Concept Agent)

```
POST /api/docs/upload
        │
        ▼
  LlamaParse Pipeline
  parse → chunk → classify → embed → store
        │
        ▼
  Formula Extraction (Ollama)
        │
        ▼
  Synthetic Example Generation (Ollama, temp≥0.7)
        │
        ▼
  Store in PostgreSQL (document_id → formulas + examples)
        │
        ▼
GET /api/docs/{document_id}/concepts
```

---

## Components and Interfaces

### 1. Monorepo Directory Structure

```
agentic-math-learning-system/
├── docker-compose.yml              # Defines fastapi, postgres, chromadb services
├── .env.example                    # Documents every required env variable
├── README.md
│
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py                     # FastAPI app factory; mounts all routers
│   │
│   ├── config.py                   # LLMConfig dataclass + named constants
│   ├── llm_wrapper.py              # Single Ollama wrapper function/class
│   ├── prompts.py                  # All prompt templates as named constants
│   ├── database.py                 # SQLAlchemy engine + session factory
│   │
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── pipeline.py             # parse → chunk → embed → store steps
│   │   └── constants.py            # EMBEDDING_MODEL, TOP_K, CHUNK_SIZE
│   │
│   ├── reranker/
│   │   ├── __init__.py
│   │   └── reranker.py             # rerank(query, chunks) → top-N chunks
│   │
│   ├── adaptive_tutor/
│   │   ├── __init__.py
│   │   ├── router.py               # FastAPI router for /api/tutor/*
│   │   ├── graph.py                # LangGraph graph definition (nodes + edges)
│   │   ├── nodes.py                # Individual node functions
│   │   └── tools.py                # LangGraph tools for Path A
│   │
│   ├── practice_agent/
│   │   ├── __init__.py
│   │   ├── router.py               # FastAPI router for /api/practice/*
│   │   ├── graph.py                # LangGraph state machine definition
│   │   ├── nodes.py                # Node functions (generate, evaluate, deep_dive)
│   │   ├── state.py                # TypedDict state schema
│   │   └── tools.py                # Deep_Dive_Tool and other tools
│   │
│   ├── doc_to_concept/
│   │   ├── __init__.py
│   │   ├── router.py               # FastAPI router for /api/docs/*
│   │   ├── graph.py                # LangGraph pipeline graph
│   │   ├── nodes.py                # Node functions (parse, chunk, classify, embed)
│   │   ├── llamaparse_client.py    # LlamaParse API wrapper
│   │   └── tools.py                # Formula extraction + example generation tools
│   │
│   └── models/
│       ├── __init__.py
│       ├── api_models.py           # Pydantic request/response models
│       └── db_models.py            # SQLAlchemy ORM models
│
└── frontend/
    ├── Dockerfile
    ├── package.json
    ├── tailwind.config.js
    ├── vite.config.ts
    ├── index.html
    └── src/
        ├── main.tsx
        ├── App.tsx                 # Tab navigation shell
        ├── api/
        │   └── client.ts           # Typed API client (fetch wrappers)
        ├── components/
        │   ├── TabNav.tsx          # Three-tab navigation bar
        │   ├── LoadingSpinner.tsx  # Shared loading indicator
        │   └── ErrorBanner.tsx     # Shared error display
        └── pages/
            ├── AdaptiveTutor.tsx   # Path A UI
            ├── PracticeAgent.tsx   # Path B UI
            └── DocToConcept.tsx    # Path C UI
```

---

### 2. Docker Compose Services

```yaml
# docker-compose.yml — annotated structure
services:
  backend:
    build: ./backend
    ports:
      - "8080:8080"
    environment:
      - OLLAMA_BASE_URL=${OLLAMA_BASE_URL}
      - POSTGRES_DSN=${POSTGRES_DSN}
      - CHROMA_HOST=${CHROMA_HOST}
      - LLAMA_PARSE_API_KEY=${LLAMA_PARSE_API_KEY}
    depends_on:
      - postgres
      - chromadb

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=${POSTGRES_USER}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - POSTGRES_DB=${POSTGRES_DB}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  chromadb:
    image: chromadb/chroma:latest
    ports:
      - "8000:8000"
    volumes:
      - chroma_data:/chroma/chroma

volumes:
  postgres_data:
  chroma_data:
```

The `.env.example` documents every variable:

```bash
# .env.example
LLAMA_PARSE_API_KEY=your_llamaparse_api_key_here
OLLAMA_BASE_URL=http://host.docker.internal:11434   # Ollama runs on host machine
POSTGRES_DSN=postgresql://user:password@postgres:5432/mathdb
POSTGRES_USER=user
POSTGRES_PASSWORD=password
POSTGRES_DB=mathdb
CHROMA_HOST=chromadb                                # Service name in compose network
CHROMA_PORT=8000
```

---

### 3. LLM Configuration (`config.py`)

```python
# config.py
# ─────────────────────────────────────────────────────────────────────
# LLMConfig centralises all LLM parameters so students can see at a
# glance how temperature and max_tokens differ per task type.
# ─────────────────────────────────────────────────────────────────────
from dataclasses import dataclass

@dataclass(frozen=True)
class LLMConfig:
    model: str
    temperature: float
    max_tokens: int

# Low temperature (≤ 0.2) for deterministic, accurate math reasoning.
# High temperature would introduce randomness that corrupts formula steps.
TUTOR_CONFIG = LLMConfig(model="llama3:latest", temperature=0.1, max_tokens=1024)

# Low temperature for MCQ generation ensures well-formed, unambiguous questions.
PRACTICE_CONFIG = LLMConfig(model="llama3:latest", temperature=0.2, max_tokens=512)

# Low temperature for formula extraction — we want exact LaTeX, not creative paraphrases.
FORMULA_EXTRACT_CONFIG = LLMConfig(model="llama3:latest", temperature=0.1, max_tokens=512)

# High temperature (≥ 0.7) for synthetic example generation encourages variety
# so students see multiple worked examples rather than near-identical repetitions.
EXAMPLE_GEN_CONFIG = LLMConfig(model="llama3:latest", temperature=0.8, max_tokens=1024)

# Chunk classification needs moderate determinism — low temperature avoids
# misclassifying "formula" chunks as "explanation" due to sampling noise.
CLASSIFY_CONFIG = LLMConfig(model="llama3:latest", temperature=0.1, max_tokens=64)
```

---

### 4. LLM Wrapper (`llm_wrapper.py`)

All call sites use a single `call_llm()` function so the invocation pattern is consistent and students can find all LLM calls in one place.

```python
# llm_wrapper.py
# ─────────────────────────────────────────────────────────────────────
# Single entry point for all Ollama LLM calls.
# Centralising here means students only need to read one file to
# understand how the system communicates with the language model.
# ─────────────────────────────────────────────────────────────────────
import logging
import os
from langchain_ollama import OllamaLLM
from config import LLMConfig

logger = logging.getLogger(__name__)

def call_llm(prompt: str, config: LLMConfig, task_name: str) -> str:
    """
    Invoke the Ollama LLM with the given prompt and configuration.

    Args:
        prompt:    The fully-rendered prompt string (XML-delimited).
        config:    LLMConfig dataclass specifying model, temperature, max_tokens.
        task_name: Human-readable label logged at DEBUG level for observability.

    Returns:
        The raw text response from the LLM.
    """
    # Log parameters at DEBUG so students can observe them with: LOG_LEVEL=DEBUG
    logger.debug(
        "LLM call | task=%s model=%s temperature=%s max_tokens=%s",
        task_name, config.model, config.temperature, config.max_tokens,
    )

    llm = OllamaLLM(
        base_url=os.environ["OLLAMA_BASE_URL"],
        model=config.model,
        temperature=config.temperature,
        num_predict=config.max_tokens,
    )
    response = llm.invoke(prompt)
    return response
```

---

### 5. Embedding Pipeline (`embeddings/pipeline.py`)

The pipeline is implemented as four clearly named, sequential functions that can be called independently or chained together.

```python
# embeddings/pipeline.py
# ─────────────────────────────────────────────────────────────────────
# The four pipeline steps are intentionally separate functions so
# students can read each step in isolation and understand its role.
# Step names match the conceptual model: parse → chunk → embed → store
# ─────────────────────────────────────────────────────────────────────
import logging
from typing import List
import chromadb
from sentence_transformers import SentenceTransformer
from embeddings.constants import EMBEDDING_MODEL, CHUNK_SIZE, TOP_K

logger = logging.getLogger(__name__)

# ── Step 1: Parse ────────────────────────────────────────────────────
def parse(raw_text: str) -> str:
    """Normalise raw text (strip noise, fix encoding). Returns clean text."""
    logger.info("pipeline_step=parse")
    return raw_text.strip()

# ── Step 2: Chunk ────────────────────────────────────────────────────
def chunk(text: str, chunk_size: int = CHUNK_SIZE) -> List[dict]:
    """
    Semantically segment text into chunks of ~chunk_size tokens.
    Each chunk dict carries: {text, chunk_index, chunk_type (placeholder)}.
    """
    # Simple sentence-boundary chunking; Doc-to-Concept path overrides
    # this with LLM-assisted semantic chunking for richer metadata.
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk_text = " ".join(words[i : i + chunk_size])
        chunks.append({"text": chunk_text, "chunk_index": len(chunks), "chunk_type": "explanation"})
    logger.info("pipeline_step=chunk chunks_produced=%d", len(chunks))
    return chunks

# ── Step 3: Embed ────────────────────────────────────────────────────
def embed(chunks: List[dict]) -> List[dict]:
    """
    Vectorise each chunk using the shared embedding model.
    Adds an 'embedding' key to each chunk dict.
    """
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [c["text"] for c in chunks]
    vectors = model.encode(texts, show_progress_bar=False)
    for c, v in zip(chunks, vectors):
        c["embedding"] = v.tolist()
    logger.info("pipeline_step=embed chunks_embedded=%d", len(chunks))
    return chunks

# ── Step 4: Store ────────────────────────────────────────────────────
def store(
    chunks: List[dict],
    collection_name: str,
    source_document: str,
    topic: str,
    chroma_client: chromadb.Client,
) -> None:
    """
    Persist embedded chunks to ChromaDB with full metadata.
    Metadata fields: source_document, chunk_index, topic, chunk_type.
    """
    collection = chroma_client.get_or_create_collection(collection_name)
    ids = [f"{source_document}_{c['chunk_index']}" for c in chunks]
    embeddings = [c["embedding"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {
            "source_document": source_document,
            "chunk_index": c["chunk_index"],
            "topic": topic,
            "chunk_type": c.get("chunk_type", "explanation"),
        }
        for c in chunks
    ]
    collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    logger.info("pipeline_step=store chunks_stored=%d collection=%s", len(chunks), collection_name)
```

**Named constants** (`embeddings/constants.py`):

```python
# embeddings/constants.py
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, fast, CPU-friendly
CHUNK_SIZE = 512        # Target token count per chunk (default per Req 8.3)
TOP_K = 10              # Candidates retrieved from ChromaDB before reranking (Req 4.5)
TOP_N = 3               # Final chunks passed to LLM after reranking (Req 5.3)
```

---

### 6. Reranker Module (`reranker/reranker.py`)

```python
# reranker/reranker.py
# ─────────────────────────────────────────────────────────────────────
# Two-stage retrieval: ChromaDB returns TOP_K candidates by cosine
# similarity (fast but imprecise); the cross-encoder reranker scores
# each candidate against the query for semantic relevance (slower but
# precise). Only TOP_N chunks reach the LLM, reducing hallucination
# risk by keeping the context window focused.
# ─────────────────────────────────────────────────────────────────────
import logging
from typing import List
from sentence_transformers import CrossEncoder
from embeddings.constants import TOP_N

logger = logging.getLogger(__name__)

# Model name as a named constant so students can swap it in one place.
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Module-level singleton — loaded once to avoid repeated disk I/O.
_cross_encoder: CrossEncoder | None = None

def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(RERANKER_MODEL)
    return _cross_encoder

def rerank(query: str, candidates: List[dict]) -> List[dict]:
    """
    Rerank candidate chunks against the query using a cross-encoder.

    Args:
        query:      The student's question string.
        candidates: List of chunk dicts from ChromaDB (each has 'text' key).

    Returns:
        Top-N chunks sorted by cross-encoder relevance score (descending).
    """
    model = _get_cross_encoder()
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)

    # Log original order vs reranked order so students can observe the improvement.
    original_order = [c.get("chunk_index", i) for i, c in enumerate(candidates)]
    scored = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    reranked_order = [c.get("chunk_index", i) for i, (_, c) in enumerate(scored)]
    logger.debug("reranker original_order=%s reranked_order=%s", original_order, reranked_order)

    # Return only TOP_N chunks — fewer chunks reduce hallucination risk because
    # the LLM cannot be distracted by loosely-related retrieved passages.
    top_chunks = [c for _, c in scored[:TOP_N]]
    return top_chunks
```

---

### 7. Path A — Adaptive Tutor

#### 7.1 LangGraph Graph (`adaptive_tutor/graph.py`)

```python
# adaptive_tutor/graph.py
# ─────────────────────────────────────────────────────────────────────
# The graph topology is defined in a single function so students can
# read the full execution flow in one place.
# Nodes: router → [retrieve → rerank → generate] | [direct_generate] → persist
# ─────────────────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, END
from adaptive_tutor.nodes import router_node, retrieve_node, rerank_node, generate_node, persist_node
from adaptive_tutor.state import TutorState

def build_tutor_graph() -> StateGraph:
    """
    Construct and compile the Adaptive Tutor LangGraph.

    Graph topology:
        router ──▶ retrieve ──▶ rerank ──▶ generate ──▶ persist ──▶ END
               └──▶ direct_generate ──────────────────▶ persist ──▶ END

    The conditional edge after 'router' implements the RAG bypass:
    simple arithmetic questions skip retrieval entirely.
    """
    graph = StateGraph(TutorState)

    # Register nodes — each node is a pure function in nodes.py
    graph.add_node("router", router_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("generate", generate_node)
    graph.add_node("direct_generate", generate_node)   # same fn, different path
    graph.add_node("persist", persist_node)

    graph.set_entry_point("router")

    # Conditional edge: LangGraph evaluates the lambda to choose the next node.
    # This is the key pattern students should study — routing based on state.
    graph.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {
            "knowledge_required": "retrieve",
            "direct_answer": "direct_generate",
        },
    )
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", "persist")
    graph.add_edge("direct_generate", "persist")
    graph.add_edge("persist", END)

    return graph.compile()
```

#### 7.2 State Schema (`adaptive_tutor/state.py`)

```python
# adaptive_tutor/state.py
from typing import TypedDict, List, Optional

class TutorState(TypedDict):
    session_id: str          # PostgreSQL session identifier
    query: str               # Student's question
    topic: Optional[str]     # Optional topic filter for retrieval
    route: str               # "knowledge_required" | "direct_answer" (set by router)
    candidates: List[dict]   # TOP_K chunks from ChromaDB
    context_chunks: List[dict]  # TOP_N chunks after reranking
    response: str            # Final LLM response
```

#### 7.3 Node Functions (`adaptive_tutor/nodes.py`)

```python
# adaptive_tutor/nodes.py
import logging
from adaptive_tutor.state import TutorState
from llm_wrapper import call_llm
from config import TUTOR_CONFIG
from prompts import TUTOR_ROUTER_PROMPT, TUTOR_RAG_PROMPT, TUTOR_DIRECT_PROMPT
from embeddings.pipeline import embed, chunk
from reranker.reranker import rerank
from embeddings.constants import TOP_K
import chromadb, os

logger = logging.getLogger(__name__)

def router_node(state: TutorState) -> TutorState:
    """Classify the query; set state['route'] to 'knowledge_required' or 'direct_answer'."""
    prompt = TUTOR_ROUTER_PROMPT.format(query=state["query"])
    classification = call_llm(prompt, TUTOR_CONFIG, task_name="router").strip().lower()
    # Normalise to one of the two valid route values
    route = "knowledge_required" if "knowledge" in classification else "direct_answer"
    logger.info("router query=%r route=%s", state["query"][:60], route)
    return {**state, "route": route}

def retrieve_node(state: TutorState) -> TutorState:
    """Query ChromaDB for TOP_K candidate chunks relevant to the student's question."""
    client = chromadb.HttpClient(host=os.environ["CHROMA_HOST"], port=int(os.environ.get("CHROMA_PORT", 8000)))
    collection = client.get_or_create_collection("math_knowledge")
    # Embed the query using the same model used during ingestion
    query_embedding = embed([{"text": state["query"]}])[0]["embedding"]
    results = collection.query(query_embeddings=[query_embedding], n_results=TOP_K,
                               where={"topic": state["topic"]} if state.get("topic") else None)
    candidates = [{"text": doc, "chunk_index": i, **meta}
                  for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0]))]
    return {**state, "candidates": candidates}

def rerank_node(state: TutorState) -> TutorState:
    """Rerank candidates and keep TOP_N for the generation context."""
    top_chunks = rerank(state["query"], state["candidates"])
    return {**state, "context_chunks": top_chunks}

def generate_node(state: TutorState) -> TutorState:
    """Build the CoT prompt and call the LLM to produce the final answer."""
    context = "\n\n".join(c["text"] for c in state.get("context_chunks", []))
    if context:
        prompt = TUTOR_RAG_PROMPT.format(context=context, question=state["query"])
    else:
        prompt = TUTOR_DIRECT_PROMPT.format(question=state["query"])
    response = call_llm(prompt, TUTOR_CONFIG, task_name="tutor_generate")
    return {**state, "response": response}

def persist_node(state: TutorState) -> TutorState:
    """Write the completed turn to the PostgreSQL tutor_sessions table."""
    from database import get_session
    from models.db_models import TutorSession
    with get_session() as db:
        record = db.query(TutorSession).filter_by(session_id=state["session_id"]).first()
        if not record:
            record = TutorSession(session_id=state["session_id"], turns=[])
            db.add(record)
        record.turns.append({
            "query": state["query"],
            "context_chunks": [c["text"] for c in state.get("context_chunks", [])],
            "response": state["response"],
        })
        db.commit()
    return state
```

#### 7.4 FastAPI Route (`adaptive_tutor/router.py`)

```python
# adaptive_tutor/router.py
from fastapi import APIRouter
from models.api_models import TutorChatRequest, TutorChatResponse
from adaptive_tutor.graph import build_tutor_graph

router = APIRouter(prefix="/api/tutor", tags=["adaptive_tutor"])
_graph = build_tutor_graph()   # compiled once at startup

@router.post("/chat", response_model=TutorChatResponse)
async def chat(request: TutorChatRequest) -> TutorChatResponse:
    """
    Accept a student question, run the Adaptive Tutor LangGraph, return the answer.
    The graph handles routing, retrieval, reranking, generation, and persistence.
    """
    initial_state = {
        "session_id": request.session_id,
        "query": request.query,
        "topic": request.topic,
        "route": "",
        "candidates": [],
        "context_chunks": [],
        "response": "",
    }
    final_state = _graph.invoke(initial_state)
    return TutorChatResponse(
        session_id=final_state["session_id"],
        response=final_state["response"],
        route_taken=final_state["route"],
        context_chunks=[c["text"] for c in final_state.get("context_chunks", [])],
    )
```

---

### 8. Path B — Practice Agent

#### 8.1 State Schema (`practice_agent/state.py`)

```python
# practice_agent/state.py
# ─────────────────────────────────────────────────────────────────────
# TypedDict state schema for the Practice Agent state machine.
# Each field is commented to explain its role in the workflow.
# ─────────────────────────────────────────────────────────────────────
from typing import TypedDict, List, Optional

class PracticeState(TypedDict):
    session_id: str           # Links to practice_sessions PostgreSQL row
    topic: str                # Math topic (e.g., "quadratic equations")
    difficulty_level: str     # "easy" | "medium" | "hard"
    question_id: str          # UUID for the current MCQ
    question_text: str        # The MCQ question stem
    options: List[str]        # Four answer options [A, B, C, D]
    correct_option: str       # The correct option letter
    selected_option: str      # The student's submitted answer
    outcome: str              # "correct" | "incorrect"
    explanation: Optional[str]  # Deep_Dive_Tool output (only on incorrect)
    graph_status: str         # "generate_question" | "evaluate_answer" | "next_question" | "retry"
    history: List[dict]       # Full turn history for persistence
```

#### 8.2 LangGraph State Machine (`practice_agent/graph.py`)

```python
# practice_agent/graph.py
# ─────────────────────────────────────────────────────────────────────
# The Practice Agent is a state machine with conditional branching:
# after evaluation, the graph routes to 'next_question' (correct) or
# 'deep_dive' (incorrect). This pattern demonstrates how LangGraph
# manages multi-turn, stateful interactions.
# ─────────────────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, END
from practice_agent.nodes import generate_question_node, evaluate_answer_node, deep_dive_node, persist_node
from practice_agent.state import PracticeState

def build_practice_graph() -> StateGraph:
    graph = StateGraph(PracticeState)

    graph.add_node("generate_question", generate_question_node)
    graph.add_node("evaluate_answer", evaluate_answer_node)
    graph.add_node("deep_dive", deep_dive_node)
    graph.add_node("persist", persist_node)

    graph.set_entry_point("generate_question")
    graph.add_edge("generate_question", END)   # /start returns after generation

    # /answer resumes from evaluate_answer
    graph.add_conditional_edges(
        "evaluate_answer",
        lambda state: state["outcome"],
        {
            "correct": "persist",
            "incorrect": "deep_dive",
        },
    )
    graph.add_edge("deep_dive", "persist")
    graph.add_edge("persist", END)

    return graph.compile()
```

#### 8.3 Tools (`practice_agent/tools.py`)

```python
# practice_agent/tools.py
# ─────────────────────────────────────────────────────────────────────
# LangGraph tools are plain Python functions. The docstring is critical:
# LangGraph passes it to the LLM so the model knows when and how to
# invoke the tool. Write docstrings as if explaining to the LLM.
# ─────────────────────────────────────────────────────────────────────
from langchain_core.tools import tool
from llm_wrapper import call_llm
from config import TUTOR_CONFIG
from prompts import DEEP_DIVE_PROMPT

@tool
def deep_dive_tool(topic: str, question: str, incorrect_answer: str) -> str:
    """
    Provide a detailed, step-by-step explanation of the concept tested by the question.

    Use this tool ONLY when a student has answered a multiple-choice question incorrectly.
    The explanation should:
    1. Identify the specific misconception implied by the incorrect answer.
    2. Re-explain the underlying concept from first principles.
    3. Walk through the correct solution step-by-step.
    4. Provide a similar worked example to reinforce understanding.

    Args:
        topic:            The math topic of the question (e.g., 'quadratic equations').
        question:         The full text of the MCQ question stem.
        incorrect_answer: The answer option the student selected.

    Returns:
        A detailed concept explanation string formatted for display to the student.
    """
    # LangGraph exposes this function to the LLM via its docstring above.
    # The LLM decides to call it when it detects an incorrect answer in state.
    prompt = DEEP_DIVE_PROMPT.format(
        topic=topic, question=question, incorrect_answer=incorrect_answer
    )
    return call_llm(prompt, TUTOR_CONFIG, task_name="deep_dive_tool")
```

---

### 9. Path C — Doc-to-Concept Agent

#### 9.1 LlamaParse Client (`doc_to_concept/llamaparse_client.py`)

```python
# doc_to_concept/llamaparse_client.py
# ─────────────────────────────────────────────────────────────────────
# Wraps the LlamaIndex LlamaParse API. Math-optimised settings preserve
# LaTeX and Unicode equation representations through the parse step.
# ─────────────────────────────────────────────────────────────────────
import os
from llama_parse import LlamaParse

def parse_pdf_to_markdown(pdf_bytes: bytes, filename: str) -> str:
    """
    Send a PDF to LlamaParse and return the extracted Markdown string.

    Math-optimised settings:
    - result_type='markdown': preserves equation structure better than plain text.
    - parsing_instruction: instructs LlamaParse to retain LaTeX delimiters.

    Raises:
        ValueError: if LlamaParse returns an error or the file is not a valid PDF.
    """
    parser = LlamaParse(
        api_key=os.environ["LLAMA_PARSE_API_KEY"],
        result_type="markdown",
        parsing_instruction=(
            "Preserve all mathematical formulas in LaTeX notation. "
            "Do not convert equations to plain text. "
            "Use $...$ for inline math and $$...$$ for display math."
        ),
        verbose=False,
    )
    # LlamaParse accepts file bytes via a temporary file path
    import tempfile, pathlib
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        documents = parser.load_data(tmp_path)
        if not documents:
            raise ValueError(f"LlamaParse returned no content for {filename}")
        return "\n\n".join(doc.text for doc in documents)
    finally:
        pathlib.Path(tmp_path).unlink(missing_ok=True)
```

#### 9.2 LangGraph Pipeline (`doc_to_concept/graph.py`)

```python
# doc_to_concept/graph.py
# ─────────────────────────────────────────────────────────────────────
# Linear pipeline graph: each node transforms the document state.
# Students can trace the full parse → chunk → classify → embed → extract
# → generate flow by reading this function top-to-bottom.
# ─────────────────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, END
from doc_to_concept.nodes import (
    parse_node, chunk_node, classify_node,
    embed_store_node, extract_formulas_node, generate_examples_node, persist_node
)
from doc_to_concept.state import DocState

def build_doc_graph() -> StateGraph:
    graph = StateGraph(DocState)
    for name, fn in [
        ("parse", parse_node),
        ("chunk", chunk_node),
        ("classify", classify_node),
        ("embed_store", embed_store_node),
        ("extract_formulas", extract_formulas_node),
        ("generate_examples", generate_examples_node),
        ("persist", persist_node),
    ]:
        graph.add_node(name, fn)

    graph.set_entry_point("parse")
    graph.add_edge("parse", "chunk")
    graph.add_edge("chunk", "classify")
    graph.add_edge("classify", "embed_store")
    graph.add_edge("embed_store", "extract_formulas")
    graph.add_edge("extract_formulas", "generate_examples")
    graph.add_edge("generate_examples", "persist")
    graph.add_edge("persist", END)
    return graph.compile()
```

#### 9.3 Tools (`doc_to_concept/tools.py`)

```python
# doc_to_concept/tools.py
from langchain_core.tools import tool
from llm_wrapper import call_llm
from config import FORMULA_EXTRACT_CONFIG, EXAMPLE_GEN_CONFIG
from prompts import FORMULA_EXTRACT_PROMPT, EXAMPLE_GEN_PROMPT
from typing import List

@tool
def extract_formulas_tool(formula_chunks: List[str]) -> List[str]:
    """
    Extract all distinct mathematical formulas from a list of formula-classified text chunks.

    Use this tool after chunk classification to identify every unique formula
    present in the document. Returns a list of LaTeX-formatted formula strings.

    Args:
        formula_chunks: List of text strings classified as 'formula' type.

    Returns:
        List of distinct LaTeX formula strings extracted from the chunks.
    """
    prompt = FORMULA_EXTRACT_PROMPT.format(chunks="\n---\n".join(formula_chunks))
    raw = call_llm(prompt, FORMULA_EXTRACT_CONFIG, task_name="extract_formulas")
    # Parse the LLM's newline-separated formula list
    return [line.strip() for line in raw.strip().splitlines() if line.strip()]

@tool
def generate_examples_tool(formula: str, topic: str) -> List[str]:
    """
    Generate at least two synthetic worked examples for a given mathematical formula.

    Use this tool for each formula extracted from the document to create
    educational worked examples that demonstrate the formula in context.
    Uses high temperature to encourage variety across examples.

    Args:
        formula: A LaTeX-formatted mathematical formula string.
        topic:   The math topic label for contextual relevance.

    Returns:
        List of at least two worked example strings, each showing the formula applied.
    """
    prompt = EXAMPLE_GEN_PROMPT.format(formula=formula, topic=topic)
    raw = call_llm(prompt, EXAMPLE_GEN_CONFIG, task_name="generate_examples")
    # Split on double newline to separate individual examples
    examples = [ex.strip() for ex in raw.strip().split("\n\n") if ex.strip()]
    return examples if len(examples) >= 2 else examples + ["(Additional example generation pending)"]
```

---

### 10. Prompt Engineering (`prompts.py`)

All prompts live in a single module as named constants. XML delimiters make the structure visible in source code. Each constant has a docstring explaining every delimiter section.

```python
# prompts.py
# ─────────────────────────────────────────────────────────────────────
# All prompts are defined here as named string constants or template
# functions. Keeping prompts out of business logic means students can
# study prompt engineering in isolation without reading graph code.
#
# XML delimiter convention:
#   <context>      — retrieved knowledge chunks passed to the LLM
#   <question>     — the student's question
#   <instructions> — task-specific directives for the LLM
#   <examples>     — in-context learning (few-shot) examples
#   <output_format>— specifies the expected response structure
# ─────────────────────────────────────────────────────────────────────

# ── Router Prompt ────────────────────────────────────────────────────
TUTOR_ROUTER_PROMPT = """
<instructions>
You are a query classifier for a math tutoring system.
Classify the student's question as exactly one of:
  - "knowledge_required"  (needs retrieved math knowledge to answer correctly)
  - "direct_answer"       (can be answered from general reasoning alone)
Respond with only the classification label, nothing else.
</instructions>

<question>
{query}
</question>
"""
# Purpose: The router node uses this to decide whether to trigger RAG.
# Low-temperature LLM call ensures consistent classification.

# ── Adaptive Tutor RAG Prompt (with CoT + few-shot example) ──────────
TUTOR_RAG_PROMPT = """
<context>
{context}
</context>

<instructions>
You are an expert math tutor. Use ONLY the information in <context> to answer
the student's question. If the context does not contain enough information,
say so clearly rather than guessing.

Chain-of-Thought instruction: Before stating your final answer, list your
reasoning steps explicitly under a "## Reasoning" heading. This helps the
student follow your logic.
</instructions>

<examples>
<!-- Few-shot example demonstrating CoT format — this is in-context learning -->
Student question: What is the quadratic formula?
## Reasoning
1. A quadratic equation has the form ax² + bx + c = 0.
2. Completing the square on the general form yields x = (-b ± √(b²-4ac)) / 2a.
3. The discriminant b²-4ac determines the number of real roots.
## Answer
The quadratic formula is: x = (-b ± √(b²-4ac)) / 2a
</examples>

<question>
{question}
</question>

<output_format>
Respond with:
## Reasoning
[numbered reasoning steps]
## Answer
[concise final answer]
</output_format>
"""
# Purpose: RAG path prompt. The <context> block is filled with reranked chunks.
# The <examples> block demonstrates the expected CoT output format (few-shot).
# The <output_format> block constrains the response structure for frontend parsing.

# ── Adaptive Tutor Direct Prompt (no retrieval) ───────────────────────
TUTOR_DIRECT_PROMPT = """
<instructions>
You are an expert math tutor. Answer the student's question directly.
Show your reasoning steps before stating the final answer.
</instructions>

<question>
{question}
</question>

<output_format>
## Reasoning
[numbered reasoning steps]
## Answer
[concise final answer]
</output_format>
"""

# ── MCQ Generation Prompt ─────────────────────────────────────────────
MCQ_GENERATION_PROMPT = """
<instructions>
Generate a single multiple-choice math question on the topic below.
Difficulty level: {difficulty_level}
Requirements:
- One clear question stem
- Exactly four answer options labelled A, B, C, D
- Exactly one correct answer
- Three plausible distractors that reflect common misconceptions
</instructions>

<topic>
{topic}
</topic>

<output_format>
Question: [question stem]
A) [option]
B) [option]
C) [option]
D) [option]
Correct: [letter]
</output_format>
"""

# ── Deep Dive Prompt ──────────────────────────────────────────────────
DEEP_DIVE_PROMPT = """
<instructions>
A student answered a math question incorrectly. Provide a detailed explanation
that helps them understand their mistake and learn the correct concept.
Structure your response as:
1. What the incorrect answer implies about the student's understanding
2. The correct concept explained from first principles
3. Step-by-step solution to the original question
4. A similar worked example to reinforce the concept
</instructions>

<context>
Topic: {topic}
Question: {question}
Student's incorrect answer: {incorrect_answer}
</context>

<output_format>
## What Went Wrong
[misconception analysis]
## The Concept
[first-principles explanation]
## Step-by-Step Solution
[numbered steps]
## Worked Example
[similar problem with full solution]
</output_format>
"""

# ── Formula Extraction Prompt ─────────────────────────────────────────
FORMULA_EXTRACT_PROMPT = """
<instructions>
Extract all distinct mathematical formulas from the text chunks below.
Return each formula on its own line in LaTeX notation.
Do not include explanatory text — only the formulas themselves.
</instructions>

<context>
{chunks}
</context>

<output_format>
[one LaTeX formula per line]
</output_format>
"""

# ── Synthetic Example Generation Prompt ──────────────────────────────
EXAMPLE_GEN_PROMPT = """
<instructions>
Generate two distinct worked examples that demonstrate the mathematical formula below.
Each example should:
- State a concrete problem that uses the formula
- Show the full step-by-step solution
- Be different from the other example (different numbers, context, or application)
</instructions>

<context>
Topic: {topic}
Formula: {formula}
</context>

<output_format>
[Example 1 with problem statement and full solution]

[Example 2 with problem statement and full solution]
</output_format>
"""

# ── Chunk Classification Prompt ───────────────────────────────────────
CHUNK_CLASSIFY_PROMPT = """
<instructions>
Classify the following text chunk from a math document into exactly one category:
  - "formula"      — contains a mathematical equation or formula as the primary content
  - "explanation"  — explains a concept, theorem, or definition in prose
  - "example"      — shows a worked example or problem solution
Respond with only the category label.
</instructions>

<context>
{chunk_text}
</context>
"""
```

---

## Data Models

### PostgreSQL ORM Models (`models/db_models.py`)

```python
# models/db_models.py
from sqlalchemy import Column, String, JSON, DateTime, func
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass

class TutorSession(Base):
    """Persists each Adaptive Tutor conversation turn."""
    __tablename__ = "tutor_sessions"
    session_id = Column(String, primary_key=True)
    turns = Column(JSON, default=list)          # List of {query, context_chunks, response}
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())

class PracticeSession(Base):
    """Persists the full state machine history for a Practice Agent session."""
    __tablename__ = "practice_sessions"
    session_id = Column(String, primary_key=True)
    topic = Column(String)
    difficulty_level = Column(String)
    history = Column(JSON, default=list)        # List of {question, answer, outcome, explanation}
    created_at = Column(DateTime, server_default=func.now())

class Document(Base):
    """Persists Doc-to-Concept Agent results for a processed PDF."""
    __tablename__ = "documents"
    document_id = Column(String, primary_key=True)
    filename = Column(String)
    topic = Column(String)
    formulas = Column(JSON, default=list)       # List of LaTeX formula strings
    examples = Column(JSON, default=dict)       # {formula: [example1, example2, ...]}
    chunk_count = Column(String)
    created_at = Column(DateTime, server_default=func.now())
```

### Pydantic API Models (`models/api_models.py`)

```python
# models/api_models.py
# ─────────────────────────────────────────────────────────────────────
# Pydantic models are the single source of truth for API serialization.
# All request bodies are validated here; all response bodies are typed
# here. This ensures round-trip integrity: serialize → deserialize
# produces an equivalent object (Req 12.1, 12.2).
# ─────────────────────────────────────────────────────────────────────
from pydantic import BaseModel, Field
from typing import List, Optional, Dict

# ── Path A ────────────────────────────────────────────────────────────
class TutorChatRequest(BaseModel):
    session_id: str = Field(..., description="UUID identifying the conversation session")
    query: str = Field(..., min_length=1, description="Student's math question")
    topic: Optional[str] = Field(None, description="Optional topic filter for retrieval")

class TutorChatResponse(BaseModel):
    session_id: str
    response: str
    route_taken: str                    # "knowledge_required" | "direct_answer"
    context_chunks: List[str] = []      # Reranked chunks used (empty for direct_answer)

# ── Path B ────────────────────────────────────────────────────────────
class PracticeStartRequest(BaseModel):
    topic: str = Field(..., description="Math topic for MCQ generation")
    difficulty_level: str = Field(..., pattern="^(easy|medium|hard)$")

class PracticeStartResponse(BaseModel):
    session_id: str
    question_id: str
    question_text: str
    options: List[str]                  # ["A) ...", "B) ...", "C) ...", "D) ..."]

class PracticeAnswerRequest(BaseModel):
    session_id: str
    question_id: str
    selected_option: str = Field(..., pattern="^[ABCD]$")

class PracticeAnswerResponse(BaseModel):
    outcome: str                        # "correct" | "incorrect"
    explanation: Optional[str] = None   # Populated by Deep_Dive_Tool on incorrect
    graph_status: str                   # "next_question" | "retry"

# ── Path C ────────────────────────────────────────────────────────────
class DocConceptsResponse(BaseModel):
    document_id: str
    filename: str
    topic: Optional[str]
    formulas: List[str]
    examples: Dict[str, List[str]]      # {formula: [worked_example_1, worked_example_2]}
    chunk_count: int
```

### ChromaDB Chunk Metadata Schema

Each chunk stored in ChromaDB carries the following metadata fields:

| Field             | Type   | Description                                      |
|-------------------|--------|--------------------------------------------------|
| `source_document` | string | Filename or document ID of the source PDF        |
| `chunk_index`     | int    | Sequential position of the chunk in the document |
| `topic`           | string | Math topic label (e.g., "calculus")              |
| `chunk_type`      | string | `"formula"` \| `"explanation"` \| `"example"`    |

---

## FastAPI Route Structure

### Main Application (`backend/main.py`)

```python
# main.py
# ─────────────────────────────────────────────────────────────────────
# FastAPI app factory. Each agentic path has its own APIRouter mounted
# here so students can find all routes in one place.
# ─────────────────────────────────────────────────────────────────────
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from adaptive_tutor.router import router as tutor_router
from practice_agent.router import router as practice_router
from doc_to_concept.router import router as doc_router

app = FastAPI(title="Agentic Math Learning System", version="1.0.0")

# Allow the React frontend (running on :3000) to call the API
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000"],
                   allow_methods=["*"], allow_headers=["*"])

app.include_router(tutor_router)      # /api/tutor/*
app.include_router(practice_router)   # /api/practice/*
app.include_router(doc_router)        # /api/docs/*
```

### Complete Route Table

| Method | Path                              | Handler                  | Description                              |
|--------|-----------------------------------|--------------------------|------------------------------------------|
| POST   | `/api/tutor/chat`                 | `tutor_router.chat`      | Adaptive Tutor chat turn                 |
| POST   | `/api/practice/start`             | `practice_router.start`  | Generate a new MCQ                       |
| POST   | `/api/practice/answer`            | `practice_router.answer` | Submit an answer; triggers Deep_Dive if wrong |
| POST   | `/api/docs/upload`                | `doc_router.upload`      | Upload PDF; run full pipeline            |
| GET    | `/api/docs/{document_id}/concepts`| `doc_router.get_concepts`| Retrieve extracted formulas + examples   |

### Path C FastAPI Route (`doc_to_concept/router.py`)

```python
# doc_to_concept/router.py
import uuid
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from models.api_models import DocConceptsResponse
from doc_to_concept.graph import build_doc_graph
from models.db_models import Document
from database import get_session

router = APIRouter(prefix="/api/docs", tags=["doc_to_concept"])
_graph = build_doc_graph()

@router.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    topic: str = Form(default=""),
) -> dict:
    """
    Accept a PDF upload, run the Doc-to-Concept pipeline, return document_id.
    Returns HTTP 422 if the file is not a valid PDF or LlamaParse fails.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=422, detail="Uploaded file must be a PDF.")
    pdf_bytes = await file.read()
    document_id = str(uuid.uuid4())
    initial_state = {
        "document_id": document_id,
        "filename": file.filename,
        "topic": topic,
        "pdf_bytes": pdf_bytes,
        "markdown_text": "",
        "chunks": [],
        "formula_chunks": [],
        "formulas": [],
        "examples": {},
    }
    try:
        _graph.invoke(initial_state)
    except ValueError as exc:
        # LlamaParse errors or invalid PDF are surfaced as 422
        raise HTTPException(status_code=422, detail=str(exc))
    return {"document_id": document_id}

@router.get("/{document_id}/concepts", response_model=DocConceptsResponse)
async def get_concepts(document_id: str) -> DocConceptsResponse:
    """Return the extracted formulas and synthetic examples for a processed document."""
    with get_session() as db:
        doc = db.query(Document).filter_by(document_id=document_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found.")
    return DocConceptsResponse(
        document_id=doc.document_id,
        filename=doc.filename,
        topic=doc.topic,
        formulas=doc.formulas,
        examples=doc.examples,
        chunk_count=int(doc.chunk_count or 0),
    )
```

---

## Frontend Design

### Component Architecture

```
App.tsx
├── TabNav.tsx                  — Three-tab navigation (Adaptive Tutor | Practice Agent | Doc-to-Concept)
├── pages/AdaptiveTutor.tsx     — Path A chat interface
├── pages/PracticeAgent.tsx     — Path B MCQ interface
└── pages/DocToConcept.tsx      — Path C PDF upload + results
```

### Shared Components

**`LoadingSpinner.tsx`** — Displayed during any in-flight API call. Uses Tailwind `animate-spin` on a border-radius circle. Shown via a `isLoading` boolean state in each page component.

**`ErrorBanner.tsx`** — Displays a descriptive error message when any API call returns a non-2xx status. Styled with a red background and dismissible via an ×  button.

### Path A — Adaptive Tutor (`AdaptiveTutor.tsx`)

```
┌─────────────────────────────────────────────────────┐
│  Adaptive Tutor                                     │
│  ┌───────────────────────────────────────────────┐  │
│  │  [Chat history — scrollable]                  │  │
│  │  User: What is integration by parts?          │  │
│  │  ─────────────────────────────────────────    │  │
│  │  Tutor:                                       │  │
│  │  ## Reasoning                                 │  │
│  │  1. Integration by parts derives from...      │  │
│  │  ## Answer                                    │  │
│  │  ∫u dv = uv − ∫v du                          │  │
│  └───────────────────────────────────────────────┘  │
│  [Topic filter input]  [Question input]  [Send]     │
└─────────────────────────────────────────────────────┘
```

- Renders `## Reasoning` and `## Answer` sections with distinct typography (reasoning in muted text, answer in bold).
- Displays `route_taken` badge ("RAG" or "Direct") next to each tutor response.
- Shows `LoadingSpinner` while awaiting `/api/tutor/chat`.

### Path B — Practice Agent (`PracticeAgent.tsx`)

```
┌─────────────────────────────────────────────────────┐
│  Practice Agent                                     │
│  Topic: [input]  Difficulty: [easy|medium|hard ▼]  │
│  [Start Practice]                                   │
│  ─────────────────────────────────────────────────  │
│  Q: What is the derivative of sin(x)?               │
│  ○ A) cos(x)   ○ B) -cos(x)                        │
│  ○ C) -sin(x)  ○ D) tan(x)                         │
│  [Submit Answer]                                    │
│  ─────────────────────────────────────────────────  │
│  ✓ Correct! / ✗ Incorrect                          │
│  ┌─────────────────────────────────────────────┐   │
│  │ 📚 Deep Dive Explanation (yellow background) │   │
│  │ What Went Wrong: ...                         │   │
│  │ The Concept: ...                             │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

- Deep Dive explanation rendered in a `bg-yellow-50 border-l-4 border-yellow-400` panel with a 📚 icon to visually distinguish it from standard feedback.
- Radio buttons for MCQ options; submit disabled until an option is selected.

### Path C — Doc-to-Concept (`DocToConcept.tsx`)

```
┌─────────────────────────────────────────────────────┐
│  Doc-to-Concept Agent                               │
│  [PDF file input]  Topic (optional): [input]        │
│  [Upload & Analyse]                                 │
│  ─────────────────────────────────────────────────  │
│  ⏳ Processing... (spinner)                         │
│  ─────────────────────────────────────────────────  │
│  Extracted Formulas (3)                             │
│  ┌─────────────────────────────────────────────┐   │
│  │ 1. x = (-b ± √(b²-4ac)) / 2a               │   │
│  │    Example 1: Find roots of x²-5x+6=0...   │   │
│  │    Example 2: Solve 2x²+3x-2=0...          │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

- File input accepts `.pdf` only (`accept=".pdf"`).
- Formulas rendered in a monospace code block; worked examples in prose below each formula.
- Polls `GET /api/docs/{document_id}/concepts` after upload completes.

### API Client (`src/api/client.ts`)

```typescript
// api/client.ts
// ─────────────────────────────────────────────────────────────────────
// Typed fetch wrappers for all backend endpoints.
// All functions throw an Error with the backend's error message on
// non-2xx responses, which the page components catch and display via
// ErrorBanner.
// ─────────────────────────────────────────────────────────────────────
const BASE_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8080";

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, options);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export const tutorChat = (payload: TutorChatRequest) =>
  apiFetch<TutorChatResponse>("/api/tutor/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

export const practiceStart = (payload: PracticeStartRequest) =>
  apiFetch<PracticeStartResponse>("/api/practice/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

export const practiceAnswer = (payload: PracticeAnswerRequest) =>
  apiFetch<PracticeAnswerResponse>("/api/practice/answer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

export const uploadDoc = (formData: FormData) =>
  apiFetch<{ document_id: string }>("/api/docs/upload", {
    method: "POST",
    body: formData,
  });

export const getDocConcepts = (documentId: string) =>
  apiFetch<DocConceptsResponse>(`/api/docs/${documentId}/concepts`);
```

---

## Error Handling

### Backend Error Strategy

| Scenario | HTTP Status | Response Body |
|---|---|---|
| Pydantic validation failure on request body | 422 | `{"detail": [{"loc": [...], "msg": "...", "type": "..."}]}` |
| LlamaParse API error or non-PDF upload | 422 | `{"detail": "descriptive message"}` |
| Document not found in PostgreSQL | 404 | `{"detail": "Document not found."}` |
| Ollama LLM unreachable | 503 | `{"detail": "LLM service unavailable"}` |
| ChromaDB unreachable | 503 | `{"detail": "Vector store unavailable"}` |
| Unhandled internal error | 500 | `{"detail": "Internal server error"}` |

### Error Handling Patterns

**LlamaParse errors** are caught in `doc_to_concept/router.py` and re-raised as `HTTPException(422)`. The `llamaparse_client.py` raises `ValueError` with a descriptive message that propagates cleanly.

**Pydantic validation** is automatic — FastAPI returns 422 with field-level detail for any request that fails model validation. No additional error handling code is needed at the route level.

**LLM unavailability** is caught in `llm_wrapper.py` with a `try/except` around the `OllamaLLM.invoke()` call. A `503` is raised so the frontend can display a meaningful message rather than a generic 500.

**ChromaDB connection errors** are caught in `retrieve_node` and `store()` with a `try/except` around the ChromaDB client calls.

**PostgreSQL errors** are handled by SQLAlchemy's session context manager — the session is rolled back on exception and the error is re-raised as a 500.

### Frontend Error Handling

All API calls in `client.ts` throw an `Error` with the backend's `detail` message on non-2xx responses. Each page component wraps API calls in `try/catch`:

```typescript
try {
  setIsLoading(true);
  const result = await tutorChat(payload);
  // handle success
} catch (err) {
  setError(err instanceof Error ? err.message : "An unexpected error occurred.");
} finally {
  setIsLoading(false);
}
```

The `ErrorBanner` component renders the error string with a red background. The loading spinner is hidden in the `finally` block regardless of outcome.

---

## Key Design Decisions

### Why LangGraph for all three paths?

LangGraph provides a consistent, inspectable state machine abstraction across all three paths. Students learn one orchestration pattern and see it applied in three different configurations: a conditional RAG graph (Path A), a multi-turn evaluation loop (Path B), and a linear pipeline (Path C). The graph topology is always defined in a single `build_*_graph()` function, making the full execution flow readable in one place.

### Why a single `call_llm()` wrapper?

Centralising all LLM calls in `llm_wrapper.py` means students can find every LLM invocation by reading one file. It also ensures that DEBUG-level logging of task name, model, temperature, and max_tokens is applied consistently — students can set `LOG_LEVEL=DEBUG` and observe every LLM call without modifying any business logic.

### Why XML delimiters in prompts?

XML-style delimiters (`<context>`, `<question>`, `<instructions>`) make the prompt structure visually obvious in source code. They also help the LLM parse the prompt structure reliably, reducing the chance of the model confusing retrieved context with instructions. The delimiter convention is documented once at the top of `prompts.py` and applied consistently across all prompts.

### Why separate `TOP_K` and `TOP_N` constants?

`TOP_K` (default 10) controls how many candidates ChromaDB returns — a larger pool improves recall. `TOP_N` (default 3) controls how many chunks the reranker passes to the LLM — a smaller context reduces hallucination risk. Separating these constants makes the two-stage retrieval trade-off explicit and easy to tune.

### Why `temperature ≤ 0.2` for math tasks?

Math problems have deterministic correct answers. High temperature introduces sampling randomness that can corrupt formula steps or produce inconsistent results across runs. Low temperature keeps the LLM's output close to its highest-probability (most accurate) completion. The rationale is documented as an inline comment next to each `LLMConfig` constant.

### Why `temperature ≥ 0.7` for synthetic example generation?

Synthetic examples benefit from variety — if all examples use the same numbers and context, students see less of the formula's applicability. Higher temperature encourages the LLM to explore different problem setups while still applying the same formula correctly.

### Why Pydantic for all API models?

Pydantic provides automatic JSON serialization/deserialization with field-level validation. This satisfies the round-trip integrity requirement (Req 12.1, 12.2) without any custom serialization code. Validation errors automatically produce 422 responses with field-level detail (Req 12.3).

