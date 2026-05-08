# adaptive_tutor/graph.py
# ─────────────────────────────────────────────────────────────────────
# LangGraph graph definition for the Adaptive Tutor (Path A).
#
# The graph topology is defined in a single function so students can
# read the full execution flow in one place (Req 10.4).  Every node,
# edge, and conditional branch is visible here without needing to trace
# through multiple files.
#
# Key LangGraph concepts demonstrated:
#   - StateGraph    — a graph whose nodes share a typed state dict
#   - add_node      — registers a Python function as a graph node
#   - set_entry_point — designates the first node to execute
#   - add_conditional_edges — routes to different nodes based on state
#   - add_edge      — unconditional transition between two nodes
#   - compile()     — validates the graph and returns an executable object
#
# Satisfies Req 6.2 (router classification), 6.3 (RAG pipeline),
#           6.4 (direct answer path), 10.4 (topology in one function).
# ─────────────────────────────────────────────────────────────────────
from langgraph.graph import END, StateGraph

from adaptive_tutor.nodes import (
    generate_node,
    persist_node,
    rerank_node,
    retrieve_node,
    router_node,
)
from adaptive_tutor.state import TutorState


def build_tutor_graph() -> StateGraph:
    """
    Construct and compile the Adaptive Tutor LangGraph.

    Graph topology
    ──────────────
    Entry point: router

    RAG path (route == "knowledge_required"):
        router → retrieve → rerank → generate → persist → END

    Direct path (route == "direct_answer"):
        router → generate → persist → END

    The conditional edge after 'router' reads state["route"] to choose
    between the two paths.  This is the key pattern students should
    study: a single classification step gates an entire sub-pipeline.

    Node responsibilities:
        router   — classifies the query; sets state["route"]
        retrieve — embeds the query; fetches TOP_K chunks from ChromaDB
        rerank   — cross-encoder reranking; keeps TOP_N chunks
        generate — builds the CoT prompt; calls the Ollama LLM
        persist  — upserts the completed turn to PostgreSQL

    Returns:
        A compiled LangGraph StateGraph ready to be invoked with an
        initial TutorState dict.
    """
    # Initialise the graph with the TutorState TypedDict as the state schema.
    # LangGraph uses the TypedDict to validate state keys at each node boundary.
    graph = StateGraph(TutorState)

    # ── Register nodes ────────────────────────────────────────────────
    # Each node is a pure function defined in nodes.py.
    # The string name is used in edge definitions below and appears in
    # LangGraph's execution trace, making it easy to follow in logs.
    graph.add_node("router", router_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("generate", generate_node)
    graph.add_node("persist", persist_node)

    # ── Entry point ───────────────────────────────────────────────────
    # Every request starts at the router node regardless of route.
    graph.set_entry_point("router")

    # ── Conditional edge: router → retrieve | generate ────────────────
    # This is the core routing decision of Path A.  LangGraph evaluates
    # the lambda after router_node completes and uses the returned string
    # to look up the next node in the mapping dict.
    #
    # The lambda reads state["route"] which was set by router_node.
    # The mapping keys must exactly match the values router_node writes:
    #   "knowledge_required" → retrieve (full RAG pipeline)
    #   "direct_answer"      → generate (skip retrieval entirely)
    graph.add_conditional_edges(
        "router",                          # source node
        lambda state: state["route"],      # function that returns the routing key
        {
            "knowledge_required": "retrieve",   # RAG path
            "direct_answer": "generate",        # Direct path
        },
    )

    # ── RAG path edges ────────────────────────────────────────────────
    # Linear pipeline: retrieve → rerank → generate → persist → END
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "generate")

    # ── Shared tail edges ─────────────────────────────────────────────
    # Both paths converge at generate → persist → END.
    # The direct path arrives at generate via the conditional edge above;
    # the RAG path arrives via rerank → generate.
    graph.add_edge("generate", "persist")
    graph.add_edge("persist", END)

    # ── Compile ───────────────────────────────────────────────────────
    # compile() validates the graph structure (checks for unreachable
    # nodes, missing edges, etc.) and returns an executable object.
    # The compiled graph is stored as a module-level singleton in
    # router.py so it is built only once at application startup.
    return graph.compile()
