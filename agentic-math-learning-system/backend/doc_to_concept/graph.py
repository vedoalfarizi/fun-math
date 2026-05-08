# doc_to_concept/graph.py
# ─────────────────────────────────────────────────────────────────────
# LangGraph pipeline graph for the Doc-to-Concept Agent (Path C).
#
# This graph is a linear pipeline — no conditional edges, no loops.
# Every uploaded PDF flows through the same seven steps in sequence.
# Students can trace the full execution flow by reading build_doc_graph()
# top-to-bottom, which is why the topology is defined in a single
# function (Req 10.4).
#
# Contrast with Path A (adaptive_tutor/graph.py) which has a conditional
# edge after the router node, and Path B (practice_agent/graph.py) which
# has a conditional edge after the evaluate_answer node. Path C
# demonstrates the simplest LangGraph pattern: a straight pipeline.
#
# Requirement satisfied: 10.4 (graph topology in a single, clearly
#                               commented function).
# ─────────────────────────────────────────────────────────────────────
from langgraph.graph import END, StateGraph

from doc_to_concept.nodes import (
    chunk_node,
    classify_node,
    embed_store_node,
    extract_formulas_node,
    generate_examples_node,
    parse_node,
    persist_node,
)
from doc_to_concept.state import DocState


def build_doc_graph() -> StateGraph:
    """
    Construct and compile the Doc-to-Concept LangGraph pipeline.

    Graph topology (linear — no conditional edges):

        parse
          │  LlamaParse: PDF bytes → Markdown string
          ▼
        chunk
          │  Word-boundary splitting → list of chunk dicts
          ▼
        classify
          │  LLM classifies each chunk as formula/explanation/example
          │  Populates formula_chunks for the extract step
          ▼
        embed_store
          │  SentenceTransformer embeds all chunks
          │  ChromaDB stores chunks with full metadata (source, topic, type)
          ▼
        extract_formulas
          │  LLM extracts LaTeX formulas from formula-classified chunks
          │  Deduplicates across chunks
          ▼
        generate_examples
          │  LLM generates ≥2 worked examples per formula (temp=0.8)
          ▼
        persist
          │  Writes Document record to PostgreSQL
          ▼
         END

    Each node is a pure function defined in nodes.py. The graph is
    compiled once at module import time (in router.py) and reused for
    every upload request, avoiding repeated compilation overhead.

    Returns:
        A compiled LangGraph StateGraph ready to invoke with an initial
        DocState dict.
    """
    graph = StateGraph(DocState)

    # ── Register nodes ────────────────────────────────────────────────
    # Each node is a pure function from nodes.py. Registering them here
    # by name makes the graph topology readable as a sequence of labels.
    graph.add_node("parse", parse_node)
    graph.add_node("chunk", chunk_node)
    graph.add_node("classify", classify_node)
    graph.add_node("embed_store", embed_store_node)
    graph.add_node("extract_formulas", extract_formulas_node)
    graph.add_node("generate_examples", generate_examples_node)
    graph.add_node("persist", persist_node)

    # ── Set entry point ───────────────────────────────────────────────
    # The pipeline always starts at parse_node, which receives the raw
    # PDF bytes from the initial state dict built in router.py.
    graph.set_entry_point("parse")

    # ── Define linear edges ───────────────────────────────────────────
    # add_edge(A, B) means: after node A completes, always go to node B.
    # There are no conditional edges in this pipeline — every document
    # follows the same path regardless of its content.
    graph.add_edge("parse", "chunk")
    graph.add_edge("chunk", "classify")
    graph.add_edge("classify", "embed_store")
    graph.add_edge("embed_store", "extract_formulas")
    graph.add_edge("extract_formulas", "generate_examples")
    graph.add_edge("generate_examples", "persist")
    graph.add_edge("persist", END)

    # ── Compile and return ────────────────────────────────────────────
    # compile() validates the graph structure (no orphan nodes, no
    # missing edges) and returns an executable CompiledGraph object.
    return graph.compile()
