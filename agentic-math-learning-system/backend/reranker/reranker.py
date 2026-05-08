# reranker/reranker.py
# ─────────────────────────────────────────────────────────────────────
# Two-stage retrieval pattern (Req 5.1 – 5.4):
#
#   Stage 1 — ChromaDB returns TOP_K candidates ranked by cosine
#             similarity (fast approximate nearest-neighbour search).
#             Cosine similarity is computed in the embedding space, so
#             it captures broad semantic overlap but can miss fine-grained
#             relevance distinctions.
#
#   Stage 2 — This module re-scores each candidate against the query
#             using a cross-encoder, which reads the query and candidate
#             *together* and produces a single relevance score. Cross-
#             encoders are slower than bi-encoders but significantly more
#             precise because they model query-document interaction.
#
# Only TOP_N chunks survive to the LLM context window. Fewer chunks
# reduce hallucination risk: the LLM cannot be distracted by loosely-
# related passages that happen to share vocabulary with the query.
# ─────────────────────────────────────────────────────────────────────
import logging
from typing import List, Optional

from sentence_transformers import CrossEncoder

from embeddings.constants import TOP_N

logger = logging.getLogger(__name__)

# ── Model constant ───────────────────────────────────────────────────
# Named constant so students can swap the reranker model in one place
# without hunting through the codebase (Req 5.2).
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# ── Module-level singleton ───────────────────────────────────────────
# The CrossEncoder is loaded once when the module is first imported and
# reused for every rerank() call. This avoids repeated disk I/O and
# model initialisation overhead on each request (Req 5.1).
_cross_encoder: Optional[CrossEncoder] = None


def _get_cross_encoder() -> CrossEncoder:
    """
    Lazily initialise and return the module-level CrossEncoder singleton.

    Using a singleton means the ~500 MB model weights are loaded from
    disk exactly once per process lifetime, keeping request latency low
    after the first call.
    """
    global _cross_encoder
    if _cross_encoder is None:
        # Loading is deferred until the first rerank() call so the
        # application starts up quickly even if the model is large.
        logger.info("reranker loading model=%s", RERANKER_MODEL)
        _cross_encoder = CrossEncoder(RERANKER_MODEL)
    return _cross_encoder


def rerank(query: str, candidates: List[dict]) -> List[dict]:
    """
    Rerank candidate chunks against the query using a cross-encoder (Req 5.1).

    The cross-encoder scores each (query, candidate) pair jointly, which
    captures fine-grained relevance that cosine similarity alone misses.
    Only the top-N highest-scoring chunks are returned to keep the LLM
    context window focused and reduce hallucination risk (Req 5.3).

    Args:
        query:      The student's question string.
        candidates: List of chunk dicts from ChromaDB. Each dict must
                    contain a 'text' key with the chunk content. Any
                    additional keys (e.g., 'chunk_index', 'topic') are
                    preserved in the returned dicts.

    Returns:
        A list of at most TOP_N chunk dicts sorted by cross-encoder
        relevance score in descending order (most relevant first).
    """
    if not candidates:
        # Guard against empty input — nothing to rerank.
        logger.debug("reranker received empty candidates list; returning []")
        return []

    model = _get_cross_encoder()

    # Build (query, document) pairs for the cross-encoder.
    # The cross-encoder reads both strings together, enabling it to model
    # the interaction between the question and each candidate passage.
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)

    # ── Observability logging (Req 5.4) ─────────────────────────────
    # Log the original retrieval order (by chunk_index if available,
    # otherwise by list position) and the reranked order so students
    # can observe how the cross-encoder changes the ranking at runtime.
    # Run with LOG_LEVEL=DEBUG to see these lines.
    original_order = [c.get("chunk_index", i) for i, c in enumerate(candidates)]
    scored_pairs = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    reranked_order = [c.get("chunk_index", i) for i, (_, c) in enumerate(scored_pairs)]
    logger.debug(
        "reranker original_order=%s reranked_order=%s",
        original_order,
        reranked_order,
    )

    # ── Truncate to TOP_N (Req 5.3) ──────────────────────────────────
    # Returning only TOP_N chunks reduces hallucination risk: the LLM
    # cannot be distracted by loosely-related retrieved passages that
    # happen to share vocabulary with the query but do not actually
    # answer it. A focused context window also keeps prompt length
    # predictable and within the model's effective attention span.
    top_chunks = [c for _, c in scored_pairs[:TOP_N]]
    return top_chunks
