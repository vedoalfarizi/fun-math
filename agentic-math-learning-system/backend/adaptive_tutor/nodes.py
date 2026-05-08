# adaptive_tutor/nodes.py
# ─────────────────────────────────────────────────────────────────────
# Node functions for the Adaptive Tutor LangGraph (Path A).
#
# Each function is a pure transformation: it receives the current
# TutorState dict, performs one focused task, and returns an updated
# copy of the state.  Keeping nodes as standalone functions (rather
# than methods on a class) makes the graph topology in graph.py easy
# to read — students can see exactly which function handles each step.
#
# Execution order (when route == "knowledge_required"):
#   router_node → retrieve_node → rerank_node → generate_node → persist_node
#
# Execution order (when route == "direct_answer"):
#   router_node → generate_node → persist_node
#
# Satisfies Req 6.2 (router classification), 6.3 (RAG pipeline),
#           6.4 (direct answer path), 6.7 (persist each turn),
#           6.8 (create session on first turn).
# ─────────────────────────────────────────────────────────────────────
import logging
import os

import chromadb

from adaptive_tutor.state import TutorState
from config import TUTOR_CONFIG
from database import get_session
from embeddings.constants import TOP_K
from embeddings.pipeline import embed
from llm_wrapper import call_llm
from models.db_models import TutorSession
from prompts import TUTOR_DIRECT_PROMPT, TUTOR_RAG_PROMPT, TUTOR_ROUTER_PROMPT
from reranker.reranker import rerank

logger = logging.getLogger(__name__)


# ── Node 1: router_node ───────────────────────────────────────────────
def router_node(state: TutorState) -> TutorState:
    """
    Classify the student's query and set the routing decision.

    Calls the LLM with TUTOR_ROUTER_PROMPT to decide whether the query
    requires knowledge-base retrieval or can be answered directly from
    the model's parametric knowledge.  The result is stored in
    state["route"] and used by the LangGraph conditional edge in
    graph.py to choose the next node.

    Logs the route decision at INFO level (Req 6.2) so students can
    observe the routing behaviour at runtime.

    Args:
        state: Current TutorState with session_id, query, and topic set.

    Returns:
        Updated TutorState with 'route' set to one of:
            "knowledge_required" — proceed to retrieve_node
            "direct_answer"      — skip retrieval, go to generate_node
    """
    # Render the router prompt with the student's query.
    prompt = TUTOR_ROUTER_PROMPT.format(query=state["query"])

    # Use TUTOR_CONFIG (temp=0.1) for deterministic binary classification.
    raw_classification = call_llm(prompt, TUTOR_CONFIG, task_name="router")
    classification = raw_classification.strip().lower()

    # Normalise the LLM output to one of the two valid route values.
    # The prompt instructs the LLM to output exactly one label, but we
    # apply a substring check as a safety net against minor formatting
    # variations (e.g., trailing punctuation or extra whitespace).
    if "knowledge" in classification:
        route = "knowledge_required"
    else:
        route = "direct_answer"

    # Log at INFO so students can observe routing decisions in the server logs.
    # Truncate the query to 80 chars to keep log lines readable.
    logger.info(
        "router_node | session=%s query=%r route=%s",
        state["session_id"],
        state["query"][:80],
        route,
    )

    return {**state, "route": route}


# ── Node 2: retrieve_node ─────────────────────────────────────────────
def retrieve_node(state: TutorState) -> TutorState:
    """
    Retrieve the top-K candidate chunks from ChromaDB for the query.

    Embeds the student's query using the same SentenceTransformer model
    used during document ingestion so the query vector lives in the same
    vector space as the stored chunk embeddings.  Applies an optional
    topic metadata filter when state["topic"] is set (Req 6.3).

    This node is only reached when route == "knowledge_required".

    Args:
        state: TutorState with query and optional topic set.

    Returns:
        Updated TutorState with 'candidates' populated — a list of
        TOP_K chunk dicts, each containing 'text' and ChromaDB metadata.
    """
    # Connect to ChromaDB using environment variables (Req 2.4 — no hardcoded addresses).
    chroma_host = os.environ["CHROMA_HOST"]
    chroma_port = int(os.environ.get("CHROMA_PORT", 8000))
    client = chromadb.HttpClient(host=chroma_host, port=chroma_port)

    # get_or_create_collection is idempotent — safe to call on every request.
    collection = client.get_or_create_collection("math_knowledge")

    # Embed the query using the shared embedding model (Req 4.3).
    # We wrap the query string in a chunk dict so embed() can process it
    # using the same code path as document chunks.
    query_embedding = embed([{"text": state["query"]}])[0]["embedding"]

    # Build the optional topic filter for ChromaDB's metadata filtering.
    # Passing where=None retrieves from all topics in the collection.
    # Passing {"topic": topic} restricts results to a specific subject.
    where_filter = {"topic": state["topic"]} if state.get("topic") else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,          # Retrieve TOP_K candidates before reranking (Req 4.5)
        where=where_filter,
    )

    # Unpack ChromaDB's nested result structure into a flat list of chunk dicts.
    # results["documents"][0] is a list of document strings for the first query.
    # results["metadatas"][0] is the corresponding list of metadata dicts.
    candidates = [
        {
            "text": doc,
            "chunk_index": i,
            # Spread any additional metadata (source_document, topic, chunk_type)
            # so downstream nodes (rerank, generate) can access it if needed.
            **(meta or {}),
        }
        for i, (doc, meta) in enumerate(
            zip(results["documents"][0], results["metadatas"][0])
        )
    ]

    logger.info(
        "retrieve_node | session=%s candidates_retrieved=%d topic=%s",
        state["session_id"],
        len(candidates),
        state.get("topic"),
    )

    return {**state, "candidates": candidates}


# ── Node 3: rerank_node ───────────────────────────────────────────────
def rerank_node(state: TutorState) -> TutorState:
    """
    Rerank the TOP_K candidates and keep only the TOP_N most relevant chunks.

    Delegates to the reranker module which uses a cross-encoder
    (BAAI/bge-reranker-v2-m3) to score each (query, candidate) pair
    jointly.  Only the top-N highest-scoring chunks are kept, reducing
    hallucination risk by keeping the LLM context window focused (Req 5.3).

    This node is only reached when route == "knowledge_required".

    Args:
        state: TutorState with query and candidates populated.

    Returns:
        Updated TutorState with 'context_chunks' set to the TOP_N
        reranked chunk dicts.
    """
    top_chunks = rerank(state["query"], state["candidates"])

    logger.info(
        "rerank_node | session=%s context_chunks=%d",
        state["session_id"],
        len(top_chunks),
    )

    return {**state, "context_chunks": top_chunks}


# ── Node 4: generate_node ─────────────────────────────────────────────
def generate_node(state: TutorState) -> TutorState:
    """
    Generate the final Chain-of-Thought answer using the Ollama LLM.

    Selects the appropriate prompt based on whether context chunks are
    available:
      - TUTOR_RAG_PROMPT  — used when context_chunks is non-empty
                            (route == "knowledge_required")
      - TUTOR_DIRECT_PROMPT — used when context_chunks is empty
                              (route == "direct_answer")

    Both prompts include CoT instructions (Req 6.6) and XML-style
    delimiters (Req 6.5) so the output format is consistent regardless
    of the route taken.

    This node is reached from both the RAG path and the direct path,
    making it the single generation step for all tutor responses.

    Args:
        state: TutorState with query and (optionally) context_chunks set.

    Returns:
        Updated TutorState with 'response' set to the full LLM output
        string (including ## Reasoning and ## Answer sections).
    """
    context_chunks = state.get("context_chunks", [])

    if context_chunks:
        # RAG path: inject the reranked chunks as <context> in the prompt.
        # Join chunks with double newlines to visually separate passages.
        context_text = "\n\n".join(c["text"] for c in context_chunks)
        prompt = TUTOR_RAG_PROMPT.format(
            context=context_text,
            question=state["query"],
        )
        task_label = "tutor_generate_rag"
    else:
        # Direct path: no retrieved context; LLM answers from parametric knowledge.
        prompt = TUTOR_DIRECT_PROMPT.format(question=state["query"])
        task_label = "tutor_generate_direct"

    response = call_llm(prompt, TUTOR_CONFIG, task_name=task_label)

    logger.info(
        "generate_node | session=%s route=%s response_length=%d",
        state["session_id"],
        state.get("route"),
        len(response),
    )

    return {**state, "response": response}


# ── Node 5: persist_node ──────────────────────────────────────────────
def persist_node(state: TutorState) -> TutorState:
    """
    Persist the completed conversation turn to the PostgreSQL tutor_sessions table.

    Implements an upsert pattern (Req 6.7, 6.8):
      - If a TutorSession row with the given session_id already exists,
        the new turn is appended to its 'turns' JSON list.
      - If no row exists, a new TutorSession is created first (Req 6.8).

    Each turn dict stored in the JSON column contains:
        query          — the student's original question
        context_chunks — text of the TOP_N chunks used as context (empty for direct path)
        response       — the full LLM response including CoT reasoning

    Uses get_session() from database.py so the connection is properly
    returned to the pool after the write completes.

    Args:
        state: Fully populated TutorState after generate_node has run.

    Returns:
        The state unchanged — persist_node is a write-only side effect.
    """
    with get_session() as db:
        # Attempt to load an existing session row.
        record = (
            db.query(TutorSession)
            .filter_by(session_id=state["session_id"])
            .first()
        )

        if not record:
            # First turn for this session — create a new row (Req 6.8).
            logger.info(
                "persist_node | creating new session session_id=%s",
                state["session_id"],
            )
            record = TutorSession(
                session_id=state["session_id"],
                turns=[],
            )
            db.add(record)

        # Build the turn dict to append.
        # Store only the text of context chunks (not the full metadata dicts)
        # to keep the JSON column compact and human-readable.
        turn = {
            "query": state["query"],
            "context_chunks": [c["text"] for c in state.get("context_chunks", [])],
            "response": state["response"],
            "route": state.get("route", ""),
        }

        # SQLAlchemy does not detect in-place mutations to JSON columns by default.
        # Reassigning the attribute (rather than calling .append()) ensures the
        # ORM marks the column as dirty and includes it in the UPDATE statement.
        record.turns = (record.turns or []) + [turn]

        db.commit()

    logger.info(
        "persist_node | session=%s turns_total=%d",
        state["session_id"],
        len(record.turns),
    )

    # Return state unchanged — this node is a pure side-effect step.
    return state
