# adaptive_tutor/router.py
# ─────────────────────────────────────────────────────────────────────
# FastAPI router for Path A — Adaptive Tutor.
#
# This module is the HTTP boundary for the Adaptive Tutor agentic path.
# It is responsible for:
#   1. Accepting the incoming HTTP request and validating it via Pydantic.
#   2. Translating the request into an initial TutorState dict.
#   3. Invoking the compiled LangGraph and waiting for the final state.
#   4. Translating the final state back into a Pydantic response model.
#
# The LangGraph is compiled once at module load time (_graph) so the
# graph structure is validated at startup rather than on the first
# request.  This also avoids the overhead of re-compiling the graph on
# every call.
#
# Satisfies Req 6.1 — exposes POST /api/tutor/chat accepting session_id,
# query, and optional topic.
# ─────────────────────────────────────────────────────────────────────
import logging

from fastapi import APIRouter

from adaptive_tutor.graph import build_tutor_graph
from models.api_models import TutorChatRequest, TutorChatResponse

logger = logging.getLogger(__name__)

# ── FastAPI router ────────────────────────────────────────────────────
# prefix="/api/tutor" means all routes defined below are reachable at
# /api/tutor/<path>.  The tag groups them in the OpenAPI docs UI.
router = APIRouter(prefix="/api/tutor", tags=["adaptive_tutor"])

# ── Compiled LangGraph singleton ──────────────────────────────────────
# build_tutor_graph() constructs and compiles the graph exactly once
# when this module is first imported (at application startup via
# main.py).  Compiling validates the graph topology and pre-allocates
# any internal LangGraph state, so subsequent invocations are fast.
#
# Students can inspect the graph structure by calling:
#   print(_graph.get_graph().draw_ascii())
_graph = build_tutor_graph()

logger.info("adaptive_tutor graph compiled and ready")


# ── POST /api/tutor/chat ──────────────────────────────────────────────
@router.post("/chat", response_model=TutorChatResponse)
async def chat(request: TutorChatRequest) -> TutorChatResponse:
    """
    Accept a student's math question and return a tutored response.

    This endpoint is the single entry point for Path A.  It:
      1. Builds the initial TutorState from the validated request.
      2. Invokes the compiled LangGraph, which runs the full pipeline:
           router → [retrieve → rerank →] generate → persist
      3. Extracts the final state and returns a TutorChatResponse.

    The route_taken field in the response tells the frontend whether
    the RAG pipeline was used ("knowledge_required") or the LLM
    answered directly ("direct_answer"), enabling the UI to display
    the appropriate badge.

    Args:
        request: Validated TutorChatRequest containing session_id,
                 query, and optional topic.

    Returns:
        TutorChatResponse with the LLM answer, route taken, and any
        context chunks that were used to ground the response.
    """
    # Build the initial state dict.  All fields must be present because
    # TutorState is a TypedDict — LangGraph validates keys at each node.
    # Fields that are populated by nodes (route, candidates, etc.) are
    # initialised to safe empty values here.
    initial_state = {
        "session_id": request.session_id,
        "query": request.query,
        "topic": request.topic,          # None if not provided by the client
        "route": "",                     # Set by router_node
        "candidates": [],                # Set by retrieve_node
        "context_chunks": [],            # Set by rerank_node
        "response": "",                  # Set by generate_node
    }

    logger.info(
        "chat | session=%s query=%r topic=%s",
        request.session_id,
        request.query[:80],
        request.topic,
    )

    # Invoke the compiled LangGraph synchronously.
    # _graph.invoke() runs all nodes in sequence (respecting conditional
    # edges) and returns the final state dict after persist_node completes.
    final_state = _graph.invoke(initial_state)

    # Map the final state back to the Pydantic response model.
    # context_chunks contains the TOP_N reranked chunk dicts; we extract
    # only the text so the response stays compact and JSON-serialisable.
    return TutorChatResponse(
        session_id=final_state["session_id"],
        response=final_state["response"],
        route_taken=final_state["route"],
        context_chunks=[
            c["text"] for c in final_state.get("context_chunks", [])
        ],
    )
