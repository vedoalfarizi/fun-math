# embeddings/constants.py
# ─────────────────────────────────────────────────────────────────────
# All embedding-related constants are defined here as named values so
# students can change them in one place and see the effect everywhere.
# Each constant references the requirement number it satisfies.
# ─────────────────────────────────────────────────────────────────────

# The shared embedding model used for all vectorisation (Req 4.3).
# all-MiniLM-L6-v2 produces 384-dimensional vectors — small enough to
# run on CPU without a GPU, making it ideal for a local demo stack.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Target token count per chunk produced by the chunking step (Req 4.3, 8.3).
# 512 tokens balances context richness (enough for a full formula + explanation)
# against retrieval precision (smaller chunks are more topically focused).
CHUNK_SIZE = 512

# Number of candidate chunks retrieved from ChromaDB before reranking (Req 4.5).
# A wider initial net (10 candidates) ensures the best chunks are not missed
# even when cosine similarity alone is imprecise.
TOP_K = 10

# Number of chunks passed to the LLM after reranking (Req 5.3).
# Keeping only the top 3 most relevant chunks reduces hallucination risk:
# the LLM cannot be distracted by loosely-related retrieved passages.
TOP_N = 3
