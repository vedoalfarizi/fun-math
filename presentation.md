# Agentic Math Learning System
### Presentation Slide Points

---

## Slide 1 — Title & Overview

**Agentic Math Learning System**

- A demo application showcasing how to build **Agentic RAG systems** and **stateful agentic workflows**
- Three independent agentic paths:
  - **Adaptive Tutor** — RAG-based math chat agent
  - **Practice Agent** — Stateful MCQ interaction agent
  - **Doc-to-Concept Agent** — PDF ingestion and concept extraction agent
- Stack: FastAPI + React + Ollama (local LLM) + Docker Compose

---

## Slide 2 — System Architecture

- **Frontend** (React + Tailwind CSS) → **Backend** (FastAPI) → 3 agentic paths
- Each path shares common utilities: embeddings, reranker, LLM wrapper, prompts
- Shared services:
  - **ChromaDB** — vector database for semantic search
  - **PostgreSQL** — session state and interaction history
  - **Ollama** — local LLM inference (`llama3:latest`), no external API dependency
- Single-command deployment: `docker-compose up`

---

## Slide 3 — Feature: Adaptive Tutor (Path A)

**What it does:** A math chat agent that retrieves relevant knowledge and reasons step-by-step before answering

**Concepts implemented:**

- **RAG (Retrieval-Augmented Generation)** — retrieves relevant document chunks from ChromaDB and injects them as context before generating an answer
- **Two-stage retrieval** — vector similarity search (ChromaDB top-K) followed by a cross-encoder reranker (top-N), improving precision over raw similarity alone
- **Query routing** — classifies each query as `knowledge_required` or `direct_answer` to skip retrieval for simple questions
- **Chain-of-Thought (CoT) prompting** — instructs the LLM to list reasoning steps before stating the final answer

---

## Slide 4 — Feature: Practice Agent (Path B)

**What it does:** Generates math MCQs and adapts based on whether the student answers correctly or not

**Concepts implemented:**

- **Stateful LangGraph state machine** — manages multi-turn interaction using a typed state schema that persists across turns
- **Conditional graph branching** — correct answer → advance to next question / incorrect answer → trigger deep dive → retry
- **Tool use** — `Deep_Dive_Tool` is a registered LangGraph tool the agent invokes autonomously when a wrong answer is detected, providing a detailed concept explanation
- **Deterministic generation** — low temperature (≤ 0.2) ensures consistent, well-formed MCQ output

---

## Slide 5 — Feature: Doc-to-Concept Agent (Path C)

**What it does:** Accepts a math PDF, extracts formulas, and auto-generates worked examples

**Concepts implemented:**

- **LlamaParse PDF parsing** — math-aware extraction that preserves LaTeX equations as structured Markdown
- **Embedding pipeline** — sequential `parse → chunk → embed → store` steps, each logged for observability
- **Semantic chunking + LLM classification** — chunks are automatically labeled as `formula`, `explanation`, or `example`
- **Synthetic example generation** — high temperature (≥ 0.7) encourages variety across generated worked examples

---

## Slide 6 — General Concepts (Applied Across All Features)

- **LangGraph orchestration** — all three paths define their workflow as an explicit graph of nodes and edges, making control flow visible and traceable
- **Structured prompt engineering** — all prompts use XML-style delimiters (`<context>`, `<instructions>`, `<output_format>`) and are centralized in a single `prompts.py` module
- **Cross-encoder reranker** (BAAI/bge-reranker-v2-m3) — a two-stage retrieval pattern that reranks raw vector search results for higher relevance
- **Session persistence** — PostgreSQL stores full interaction history for all three paths via SQLAlchemy ORM
- **Local LLM inference** — Ollama runs `llama3:latest` on the host machine with no external API calls
- **Pydantic validation** — all API request and response bodies are typed with Pydantic models, ensuring consistent serialization at every boundary

---

## Slide 7 — Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI (Python 3.11+) |
| Frontend | React + Tailwind CSS |
| Orchestration | LangGraph |
| LLM | Ollama (`llama3:latest`) |
| Vector DB | ChromaDB |
| Relational DB | PostgreSQL |
| PDF Parsing | LlamaParse (LlamaIndex) |
| Reranker | sentence-transformers (`BAAI/bge-reranker-v2-m3`) |
| Deployment | Docker Compose |
