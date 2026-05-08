# adaptive_tutor/state.py
# ─────────────────────────────────────────────────────────────────────
# TutorState is the single shared data structure that flows through
# every node in the Adaptive Tutor LangGraph.  Defining it as a
# TypedDict gives students a clear, typed contract for what data each
# node can read and must write.
#
# LangGraph passes the state dict from node to node; each node returns
# a (possibly updated) copy of the state.  Students can trace the full
# data lifecycle by reading this file alongside nodes.py.
#
# Satisfies Req 6.1 — the /api/tutor/chat endpoint accepts session_id,
# query, and optional topic, all of which are fields in this TypedDict.
# ─────────────────────────────────────────────────────────────────────
from typing import List, Optional, TypedDict


class TutorState(TypedDict):
    """
    Shared state schema for the Adaptive Tutor LangGraph (Path A).

    Every node in the graph receives this dict and returns an updated
    copy.  Fields are populated progressively as the graph executes:
    router_node sets 'route', retrieve_node sets 'candidates',
    rerank_node sets 'context_chunks', generate_node sets 'response',
    and persist_node writes the completed turn to PostgreSQL.
    """

    # ── Input fields (populated by the FastAPI route handler) ─────────

    session_id: str
    # UUID string identifying the conversation session.
    # Supplied by the frontend on component mount and reused for every
    # turn in the same conversation so the backend can append turns to
    # the correct PostgreSQL row (Req 6.7, 6.8).

    query: str
    # The student's raw math question as submitted via the chat input.
    # Passed verbatim to the router_node for classification and later
    # used as the retrieval query in retrieve_node.

    topic: Optional[str]
    # Optional subject label (e.g., "calculus", "linear algebra").
    # When provided, retrieve_node applies it as a ChromaDB metadata
    # filter so only chunks tagged with this topic are considered.
    # None means search across all topics in the knowledge base.

    # ── Routing field (set by router_node) ────────────────────────────

    route: str
    # Classification decision produced by router_node.
    # One of two values:
    #   "knowledge_required" — full RAG pipeline (retrieve → rerank → generate)
    #   "direct_answer"      — LLM called directly without retrieval
    # The LangGraph conditional edge reads this field to choose the next node.

    # ── Retrieval fields (set by retrieve_node and rerank_node) ───────

    candidates: List[dict]
    # TOP_K chunk dicts returned by ChromaDB before reranking.
    # Each dict contains at minimum: {"text": str, "chunk_index": int}.
    # Additional ChromaDB metadata keys (topic, source_document, etc.)
    # are preserved so rerank_node and generate_node can use them.
    # Empty list when route == "direct_answer" (retrieval is skipped).

    context_chunks: List[dict]
    # TOP_N chunk dicts selected by rerank_node after cross-encoder scoring.
    # These are the chunks injected into the TUTOR_RAG_PROMPT as <context>.
    # Fewer chunks than candidates reduces hallucination risk (Req 5.3).
    # Empty list when route == "direct_answer".

    # ── Generation field (set by generate_node) ───────────────────────

    response: str
    # The full LLM response string, including the "## Reasoning" and
    # "## Answer" sections produced by the CoT prompt (Req 6.6).
    # Returned to the client in TutorChatResponse.response.
