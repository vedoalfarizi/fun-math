# adaptive_tutor/tools.py
# ─────────────────────────────────────────────────────────────────────
# LangGraph tools for Path A — Adaptive Tutor.
#
# In Path A the retrieval and reranking logic is implemented directly
# as LangGraph *nodes* (see nodes.py) rather than as @tool-decorated
# functions.  This is an intentional design choice that demonstrates
# two different patterns students should understand:
#
#   Pattern 1 — Node-based retrieval (used here in Path A):
#     The graph explicitly calls retrieve_node and rerank_node as
#     sequential steps.  The LLM never "decides" to retrieve; the graph
#     topology enforces retrieval whenever route == "knowledge_required".
#     This pattern is appropriate when retrieval is always deterministic
#     and the graph controls the flow.
#
#   Pattern 2 — Tool-based retrieval (used in Path B / Path C):
#     The LLM is given a @tool-decorated function and decides at runtime
#     whether and when to call it.  This pattern is appropriate for
#     open-ended agentic loops where the LLM must reason about which
#     tool to invoke.
#
# The retrieve_knowledge tool below is provided as an *optional*
# educational reference showing how the same retrieval logic would look
# if exposed as a LangGraph tool.  It is NOT wired into the graph in
# graph.py — the node-based approach is used there instead.
#
# Satisfies Req 10.1 (tools in dedicated tools.py), 10.2 (complete
# docstrings), 10.3 (registration-site comments).
# ─────────────────────────────────────────────────────────────────────
import os
from typing import List

import chromadb
from langchain_core.tools import tool

from embeddings.constants import TOP_K
from embeddings.pipeline import embed
from reranker.reranker import rerank


# ── Tool: retrieve_knowledge ─────────────────────────────────────────
# Registration note (Req 10.3):
#   The @tool decorator from langchain_core.tools registers this
#   function as a LangGraph-compatible tool.  LangGraph reads the
#   function's docstring and type annotations to build the tool schema
#   that is passed to the LLM.  The LLM uses the docstring to decide
#   *when* to call the tool and the type annotations to know *what
#   arguments* to supply.  Writing a clear, precise docstring is
#   therefore critical for correct tool invocation.
@tool
def retrieve_knowledge(query: str, topic: str = "") -> List[str]:
    """
    Retrieve and rerank the most relevant knowledge chunks for a math query.

    Use this tool when you need to look up specific mathematical concepts,
    theorems, formulas, or proofs from the knowledge base before answering
    a student's question.  Do NOT use this tool for straightforward
    arithmetic calculations that can be answered from general knowledge.

    The tool performs two-stage retrieval:
      1. Embeds the query and fetches the top-K candidates from ChromaDB
         using cosine similarity (fast but approximate).
      2. Reranks the candidates with a cross-encoder for precise relevance
         scoring and returns only the top-N most relevant chunks.

    Args:
        query: The student's math question or the specific concept to look up.
               Should be a natural-language string (e.g., "What is the
               fundamental theorem of calculus?").
        topic: Optional subject label to restrict retrieval to a specific
               topic (e.g., "calculus", "linear algebra").  Pass an empty
               string to search across all topics.

    Returns:
        A list of text strings — the top-N reranked knowledge chunks most
        relevant to the query.  Each string is a passage from the knowledge
        base that can be used as context for answering the question.
        Returns an empty list if no relevant chunks are found.
    """
    # Connect to ChromaDB using environment variables — never hardcode addresses.
    chroma_host = os.environ["CHROMA_HOST"]
    chroma_port = int(os.environ.get("CHROMA_PORT", 8000))
    client = chromadb.HttpClient(host=chroma_host, port=chroma_port)

    collection = client.get_or_create_collection("math_knowledge")

    # Embed the query using the same model used during document ingestion
    # so the query vector lives in the same vector space as the stored chunks.
    query_chunks = embed([{"text": query}])
    query_embedding = query_chunks[0]["embedding"]

    # Build the optional topic filter for ChromaDB's metadata filtering.
    # Passing where=None retrieves from all topics; passing a topic dict
    # restricts results to chunks tagged with that subject label.
    where_filter = {"topic": topic} if topic else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        where=where_filter,
    )

    # Unpack ChromaDB results into the chunk dict format expected by rerank().
    candidates = [
        {"text": doc, "chunk_index": i, **(meta or {})}
        for i, (doc, meta) in enumerate(
            zip(results["documents"][0], results["metadatas"][0])
        )
    ]

    # Rerank with the cross-encoder and return only the text of the top-N chunks.
    top_chunks = rerank(query, candidates)
    return [c["text"] for c in top_chunks]
