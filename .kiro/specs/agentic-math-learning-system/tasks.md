# Implementation Plan: Agentic Math Learning System

## Overview

Implement the full monorepo stack — FastAPI backend with three LangGraph agentic paths, shared utilities, and a React + Tailwind frontend — following the design document exactly. Each task builds on the previous, ending with all services wired together via Docker Compose.

## Tasks

- [x] 1. Monorepo scaffolding
  - Create the top-level directory structure: `agentic-math-learning-system/`, `backend/`, `frontend/`
  - Write `docker-compose.yml` defining `backend`, `postgres`, and `chromadb` services with correct ports, environment variable references, `depends_on`, and named volumes (`postgres_data`, `chroma_data`)
  - Write `.env.example` documenting every required variable: `LLAMA_PARSE_API_KEY`, `OLLAMA_BASE_URL`, `POSTGRES_DSN`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `CHROMA_HOST`, `CHROMA_PORT`
  - Write `README.md` with a quick-start section (`cp .env.example .env` → `docker-compose up`)
  - Create all `__init__.py` files for every Python package under `backend/`
  - _Requirements: 1.1, 2.1, 2.2, 2.3_

- [x] 2. Backend foundation
  - [x] 2.1 Write `backend/config.py` — define the frozen `LLMConfig` dataclass and the five named config constants (`TUTOR_CONFIG`, `PRACTICE_CONFIG`, `FORMULA_EXTRACT_CONFIG`, `EXAMPLE_GEN_CONFIG`, `CLASSIFY_CONFIG`) with inline comments explaining each temperature choice
    - _Requirements: 3.1, 3.2_
  - [x] 2.2 Write `backend/llm_wrapper.py` — implement `call_llm(prompt, config, task_name)` using `langchain_ollama.OllamaLLM`, reading `OLLAMA_BASE_URL` from the environment, and logging task name, model, temperature, and max_tokens at DEBUG level
    - _Requirements: 3.3, 3.4_
  - [x] 2.3 Write `backend/database.py` — create the SQLAlchemy engine from `POSTGRES_DSN`, a `SessionLocal` factory, and a `get_session()` context-manager helper; include `Base.metadata.create_all()` call so tables are created on startup
    - _Requirements: 1.2_
  - [x] 2.4 Write `backend/main.py` — FastAPI app factory that adds CORS middleware (allowing `http://localhost:3000`), mounts the three routers (`/api/tutor/*`, `/api/practice/*`, `/api/docs/*`), and calls `Base.metadata.create_all()` on startup
    - _Requirements: 1.2, 2.3, 11.1_
  - [x] 2.5 Write `backend/requirements.txt` — pin all dependencies: `fastapi`, `uvicorn`, `langchain-ollama`, `langgraph`, `langchain-core`, `chromadb`, `sentence-transformers`, `sqlalchemy`, `psycopg2-binary`, `pydantic`, `llama-parse`, `python-multipart`
    - _Requirements: 2.1_
  - [x] 2.6 Write `backend/Dockerfile` — Python 3.11-slim base, copy `requirements.txt`, run `pip install`, copy source, expose port 8080, set `CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]`
    - _Requirements: 2.1, 2.3_

- [x] 3. Shared utilities
  - [x] 3.1 Write `backend/embeddings/constants.py` — define `EMBEDDING_MODEL`, `CHUNK_SIZE` (512), `TOP_K` (10), `TOP_N` (3) as named constants with inline comments referencing the requirement numbers they satisfy
    - _Requirements: 4.3, 4.5, 5.3_
  - [x] 3.2 Write `backend/embeddings/pipeline.py` — implement the four sequential pipeline functions `parse()`, `chunk()`, `embed()`, `store()` exactly as designed; each function must log its step name and chunk count at INFO level; `store()` must write `source_document`, `chunk_index`, `topic`, and `chunk_type` metadata to ChromaDB
    - _Requirements: 4.1, 4.2, 4.3, 4.4_
  - [x] 3.3 Write `backend/reranker/reranker.py` — implement `rerank(query, candidates)` using `CrossEncoder(RERANKER_MODEL)` as a module-level singleton; log original and reranked chunk order at DEBUG level; return only `TOP_N` chunks; include inline comment explaining why fewer chunks reduce hallucination risk
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [x] 4. Prompt engineering
  - Write `backend/prompts.py` — define all eight named prompt constants: `TUTOR_ROUTER_PROMPT`, `TUTOR_RAG_PROMPT` (with `<examples>` few-shot block), `TUTOR_DIRECT_PROMPT`, `MCQ_GENERATION_PROMPT`, `DEEP_DIVE_PROMPT`, `FORMULA_EXTRACT_PROMPT`, `EXAMPLE_GEN_PROMPT`, `CHUNK_CLASSIFY_PROMPT`
  - Every prompt must use XML-style delimiters (`<context>`, `<question>`, `<instructions>`, `<examples>`, `<output_format>`) and include a Python comment explaining the purpose of each delimiter section
  - `TUTOR_RAG_PROMPT` must include at least one in-context learning example inside `<examples>` demonstrating the CoT output format
  - _Requirements: 6.5, 6.6, 9.1, 9.2, 9.3, 9.4_

- [x] 5. Data models
  - [x] 5.1 Write `backend/models/db_models.py` — define `Base` (DeclarativeBase), `TutorSession`, `PracticeSession`, and `Document` ORM models with all columns, JSON fields, and `created_at`/`updated_at` timestamps exactly as designed; include a docstring on each class explaining its role
    - _Requirements: 6.7, 7.7, 8.8_
  - [x] 5.2 Write `backend/models/api_models.py` — define all eight Pydantic models: `TutorChatRequest`, `TutorChatResponse`, `PracticeStartRequest`, `PracticeStartResponse`, `PracticeAnswerRequest`, `PracticeAnswerResponse`, `DocConceptsResponse`; use `Field` validators where specified (e.g., `difficulty_level` pattern, `selected_option` pattern, `query` min_length); include module-level comment referencing Req 12.1 and 12.2
    - _Requirements: 12.1, 12.2, 12.3_

- [x] 6. Path A — Adaptive Tutor
  - [x] 6.1 Write `backend/adaptive_tutor/state.py` — define `TutorState` TypedDict with all seven fields (`session_id`, `query`, `topic`, `route`, `candidates`, `context_chunks`, `response`) and an inline comment on each field
    - _Requirements: 6.1_
  - [x] 6.2 Write `backend/adaptive_tutor/tools.py` — define any Path A LangGraph tools (e.g., a retrieval tool wrapper if needed); include complete docstrings and a comment at each registration site explaining how LangGraph exposes the tool to the LLM
    - _Requirements: 10.1, 10.2, 10.3_
  - [x] 6.3 Write `backend/adaptive_tutor/nodes.py` — implement `router_node`, `retrieve_node`, `rerank_node`, `generate_node`, and `persist_node`; `router_node` must log the route decision at INFO level; `retrieve_node` must embed the query and query ChromaDB with `TOP_K` and optional topic filter; `persist_node` must upsert to `tutor_sessions` via `get_session()`
    - _Requirements: 6.2, 6.3, 6.4, 6.7, 6.8_
  - [x] 6.4 Write `backend/adaptive_tutor/graph.py` — implement `build_tutor_graph()` defining all five nodes, the entry point, the conditional edge after `router` (routing on `state["route"]`), and the linear edges through to `END`; include a docstring describing the full graph topology
    - _Requirements: 6.2, 6.3, 6.4, 10.4_
  - [x] 6.5 Write `backend/adaptive_tutor/router.py` — FastAPI `APIRouter` with prefix `/api/tutor`; implement `POST /chat` that builds the initial state, invokes `_graph`, and returns a `TutorChatResponse`; compile the graph once at module level
    - _Requirements: 6.1_

- [x] 7. Path B — Practice Agent
  - [x] 7.1 Write `backend/practice_agent/state.py` — define `PracticeState` TypedDict with all eleven fields; include an inline comment on each field explaining its role in the workflow
    - _Requirements: 7.8_
  - [x] 7.2 Write `backend/practice_agent/tools.py` — implement `deep_dive_tool` as a `@tool`-decorated function with a complete docstring (purpose, when to invoke, args, returns); call `call_llm` with `DEEP_DIVE_PROMPT` and `TUTOR_CONFIG`; include a comment at the registration site explaining how LangGraph exposes it to the LLM
    - _Requirements: 7.4, 7.5, 10.1, 10.2, 10.3_
  - [x] 7.3 Write `backend/practice_agent/nodes.py` — implement `generate_question_node` (calls LLM with `MCQ_GENERATION_PROMPT` and `PRACTICE_CONFIG`, parses question/options/correct from response), `evaluate_answer_node` (compares `selected_option` to `correct_option`, sets `outcome`), `deep_dive_node` (invokes `deep_dive_tool`), and `persist_node` (upserts to `practice_sessions`)
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.6, 7.7_
  - [x] 7.4 Write `backend/practice_agent/graph.py` — implement `build_practice_graph()` with four nodes, entry point at `generate_question`, edge from `generate_question` to `END`, conditional edges from `evaluate_answer` routing on `outcome` (`"correct"` → `persist`, `"incorrect"` → `deep_dive`), and edges `deep_dive` → `persist` → `END`; include topology docstring
    - _Requirements: 7.3, 7.4, 10.4_
  - [x] 7.5 Write `backend/practice_agent/router.py` — FastAPI `APIRouter` with prefix `/api/practice`; implement `POST /start` (invokes graph from `generate_question` entry, returns `PracticeStartResponse`) and `POST /answer` (invokes graph from `evaluate_answer` entry, returns `PracticeAnswerResponse`)
    - _Requirements: 7.1, 7.2_

- [x] 8. Path C — Doc-to-Concept Agent
  - [x] 8.1 Write `backend/doc_to_concept/llamaparse_client.py` — implement `parse_pdf_to_markdown(pdf_bytes, filename)` using `LlamaParse` with `result_type="markdown"` and the math-preserving `parsing_instruction`; write to a temp file, call `parser.load_data()`, join document texts, clean up the temp file; raise `ValueError` with a descriptive message on failure
    - _Requirements: 8.2, 8.9, 12.4_
  - [x] 8.2 Write `backend/doc_to_concept/state.py` — define `DocState` TypedDict with all fields needed by the pipeline nodes: `document_id`, `filename`, `topic`, `pdf_bytes`, `markdown_text`, `chunks`, `formula_chunks`, `formulas`, `examples`
    - _Requirements: 8.1_
  - [x] 8.3 Write `backend/doc_to_concept/tools.py` — implement `extract_formulas_tool` and `generate_examples_tool` as `@tool`-decorated functions with complete docstrings; `extract_formulas_tool` calls LLM with `FORMULA_EXTRACT_PROMPT` and `FORMULA_EXTRACT_CONFIG`; `generate_examples_tool` calls LLM with `EXAMPLE_GEN_PROMPT` and `EXAMPLE_GEN_CONFIG` (temperature ≥ 0.7); include registration-site comments
    - _Requirements: 8.6, 8.7, 10.1, 10.2, 10.3_
  - [x] 8.4 Write `backend/doc_to_concept/nodes.py` — implement all seven pipeline nodes: `parse_node` (calls `parse_pdf_to_markdown`), `chunk_node` (calls `chunk()` from pipeline), `classify_node` (calls LLM with `CHUNK_CLASSIFY_PROMPT` per chunk, sets `chunk_type`), `embed_store_node` (calls `embed()` then `store()` with full metadata), `extract_formulas_node` (invokes `extract_formulas_tool`), `generate_examples_node` (invokes `generate_examples_tool` per formula, ensures ≥ 2 examples each), `persist_node` (writes `Document` record to PostgreSQL)
    - _Requirements: 8.2, 8.3, 8.4, 8.5, 8.6, 8.7_
  - [x] 8.5 Write `backend/doc_to_concept/graph.py` — implement `build_doc_graph()` as a linear pipeline: `parse → chunk → classify → embed_store → extract_formulas → generate_examples → persist → END`; include topology docstring
    - _Requirements: 10.4_
  - [x] 8.6 Write `backend/doc_to_concept/router.py` — FastAPI `APIRouter` with prefix `/api/docs`; implement `POST /upload` (validates `.pdf` extension, generates UUID, invokes graph, catches `ValueError` as HTTP 422) and `GET /{document_id}/concepts` (queries PostgreSQL, returns `DocConceptsResponse`, raises 404 if not found)
    - _Requirements: 8.1, 8.8, 8.9_

- [x] 9. Checkpoint — backend wiring
  - Ensure all backend modules import cleanly (no circular imports)
  - Verify `main.py` mounts all three routers and `Base.metadata.create_all()` runs on startup
  - Confirm all environment variable reads use `os.environ` (no hardcoded secrets)
  - Ask the user if questions arise.
  - _Requirements: 2.4, 1.4_

- [x] 10. Frontend foundation
  - [x] 10.1 Scaffold the React + Vite + TypeScript project under `frontend/` — create `package.json` with dependencies (`react`, `react-dom`, `typescript`, `vite`, `@vitejs/plugin-react`, `tailwindcss`, `postcss`, `autoprefixer`), `vite.config.ts`, `tailwind.config.js`, `index.html`, and `src/main.tsx`
    - _Requirements: 11.1_
  - [x] 10.2 Write `frontend/Dockerfile` — Node 20-alpine base, copy `package.json`, run `npm install`, copy source, expose port 3000, set `CMD ["npm", "run", "dev", "--", "--host"]`
    - _Requirements: 2.1_
  - [x] 10.3 Write `frontend/src/api/client.ts` — implement `apiFetch<T>()` base helper reading `VITE_API_URL` from env (defaulting to `http://localhost:8080`); implement typed wrappers `tutorChat`, `practiceStart`, `practiceAnswer`, `uploadDoc`, `getDocConcepts` matching the Pydantic model shapes exactly
    - _Requirements: 11.4, 11.5, 12.1_
  - [x] 10.4 Write `frontend/src/components/LoadingSpinner.tsx` — Tailwind `animate-spin` spinner shown via an `isLoading` prop
    - _Requirements: 11.4_
  - [x] 10.5 Write `frontend/src/components/ErrorBanner.tsx` — red-background banner with dismissible × button; accepts an `error` string prop; renders nothing when `error` is null
    - _Requirements: 11.5_
  - [x] 10.6 Write `frontend/src/components/TabNav.tsx` — three-tab navigation bar with labels "Adaptive Tutor", "Practice Agent", "Doc-to-Concept"; highlights the active tab; calls an `onSelect` callback with the tab index
    - _Requirements: 11.1_
  - [x] 10.7 Write `frontend/src/App.tsx` — renders `TabNav` and conditionally renders the active page component (`AdaptiveTutor`, `PracticeAgent`, or `DocToConcept`) based on selected tab
    - _Requirements: 11.1_

- [x] 11. Frontend Path A — Adaptive Tutor page
  - Write `frontend/src/pages/AdaptiveTutor.tsx` — implement the chat interface with scrollable chat history, topic filter input, question input, and Send button; call `tutorChat` on submit; render each tutor response with `## Reasoning` in muted text and `## Answer` in bold; display a `route_taken` badge ("RAG" or "Direct") next to each response; show `LoadingSpinner` while awaiting the API; show `ErrorBanner` on non-2xx responses; generate a UUID `session_id` on component mount
  - _Requirements: 11.1, 11.2, 11.4, 11.5_

- [x] 12. Frontend Path B — Practice Agent page
  - Write `frontend/src/pages/PracticeAgent.tsx` — implement topic input, difficulty dropdown (easy/medium/hard), and Start Practice button; call `practiceStart` and render the MCQ with four radio-button options; disable Submit until an option is selected; call `practiceAnswer` on submit; render ✓ Correct or ✗ Incorrect feedback; when `explanation` is present, render it in a `bg-yellow-50 border-l-4 border-yellow-400` panel with a 📚 icon to visually distinguish it from standard feedback; show `LoadingSpinner` and `ErrorBanner` appropriately
  - _Requirements: 11.1, 11.3, 11.4, 11.5_

- [x] 13. Frontend Path C — Doc-to-Concept page
  - Write `frontend/src/pages/DocToConcept.tsx` — implement PDF file input (`accept=".pdf"`), optional topic text input, and Upload & Analyse button; call `uploadDoc` with a `FormData` payload; after upload, poll `getDocConcepts` until the document is ready; render extracted formulas in monospace code blocks with their worked examples in prose below each formula; show `LoadingSpinner` during upload and processing; show `ErrorBanner` on 422 or other errors
  - _Requirements: 11.1, 11.4, 11.5, 11.6_

- [x] 14. Docker Compose wiring — final verification
  - Update `docker-compose.yml` to add the `frontend` service (build `./frontend`, port `3000:3000`, `VITE_API_URL=http://backend:8080`) and ensure `backend` depends on `postgres` and `chromadb`
  - Verify the `backend` service passes all required environment variables (`OLLAMA_BASE_URL`, `POSTGRES_DSN`, `CHROMA_HOST`, `CHROMA_PORT`, `LLAMA_PARSE_API_KEY`) from the `.env` file
  - Confirm `ollama` is documented in `README.md` as a host-side prerequisite (not a compose service) with `OLLAMA_BASE_URL=http://host.docker.internal:11434`
  - Ask the user if questions arise.
  - _Requirements: 2.1, 2.2, 2.3_

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP — no optional tasks are present in this plan per the project brief
- Each task references specific requirements for traceability
- Checkpoints (tasks 9 and 14) ensure incremental validation before moving to the next major phase
- The design document contains complete, copy-ready implementations for every module — use it as the authoritative reference during coding
- All secrets must be read from environment variables; never hardcode credentials
