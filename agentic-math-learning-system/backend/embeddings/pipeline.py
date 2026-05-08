# embeddings/pipeline.py
# ─────────────────────────────────────────────────────────────────────
# The four pipeline steps are intentionally separate functions so
# students can read each step in isolation and understand its role.
# Step names match the conceptual model: parse → chunk → embed → store
#
# Requirements satisfied: 4.1 (sequential steps), 4.2 (INFO logging),
#                         4.3 (shared embedding model), 4.4 (metadata).
# ─────────────────────────────────────────────────────────────────────
import logging
from typing import List

import chromadb
from sentence_transformers import SentenceTransformer

from embeddings.constants import EMBEDDING_MODEL, CHUNK_SIZE

logger = logging.getLogger(__name__)


# ── Step 1: Parse ────────────────────────────────────────────────────
def parse(raw_text: str) -> str:
    """
    Normalise raw text before chunking.

    Strips leading/trailing whitespace and collapses runs of blank lines
    so downstream chunking operates on clean, consistent input.

    Args:
        raw_text: The raw string extracted from a document (e.g., LlamaParse
                  Markdown output or plain text).

    Returns:
        A cleaned string ready for chunking.
    """
    # Log the step name at INFO so students can trace the pipeline at runtime.
    logger.info("pipeline_step=parse")
    # Strip outer whitespace; additional normalisation (e.g., Unicode fixes)
    # can be added here without touching any other pipeline step.
    cleaned = raw_text.strip()
    return cleaned


# ── Step 2: Chunk ────────────────────────────────────────────────────
def chunk(text: str, chunk_size: int = CHUNK_SIZE) -> List[dict]:
    """
    Segment text into chunks of approximately chunk_size words.

    Each chunk is represented as a dict with:
        - text        : the chunk string
        - chunk_index : zero-based position in the document
        - chunk_type  : placeholder value "explanation"; the Doc-to-Concept
                        path overrides this with LLM-assisted classification.

    Args:
        text:       Cleaned text from the parse() step.
        chunk_size: Target word count per chunk (default: CHUNK_SIZE = 512).

    Returns:
        List of chunk dicts ordered by position in the source document.
    """
    # Word-boundary splitting is a simple, transparent strategy that students
    # can easily follow. The Doc-to-Concept path replaces this with semantic
    # chunking for richer metadata (see doc_to_concept/nodes.py).
    words = text.split()
    chunks: List[dict] = []
    for i in range(0, len(words), chunk_size):
        chunk_text = " ".join(words[i : i + chunk_size])
        chunks.append(
            {
                "text": chunk_text,
                "chunk_index": len(chunks),
                # Default type; overridden by classify_node in Path C.
                "chunk_type": "explanation",
            }
        )

    # Log step name and output count so students can observe the pipeline (Req 4.2).
    logger.info("pipeline_step=chunk chunks_produced=%d", len(chunks))
    return chunks


# ── Step 3: Embed ────────────────────────────────────────────────────
def embed(chunks: List[dict]) -> List[dict]:
    """
    Vectorise each chunk using the shared embedding model (Req 4.3).

    Loads the SentenceTransformer model, encodes all chunk texts in a
    single batch for efficiency, and adds an 'embedding' key (list of
    floats) to each chunk dict.

    Args:
        chunks: List of chunk dicts from chunk() (or any list of dicts
                with a 'text' key).

    Returns:
        The same list of chunk dicts, each augmented with an 'embedding'
        key containing the vector as a plain Python list of floats.
    """
    # Using the module-level constant ensures every call site uses the
    # same model, so query embeddings and document embeddings are
    # always in the same vector space (Req 4.3).
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [c["text"] for c in chunks]
    # show_progress_bar=False keeps logs clean; students can enable it
    # locally by passing show_progress_bar=True for large batches.
    vectors = model.encode(texts, show_progress_bar=False)
    for c, v in zip(chunks, vectors):
        # Convert numpy array to a plain list so it is JSON-serialisable
        # and compatible with ChromaDB's Python client.
        c["embedding"] = v.tolist()

    # Log step name and count (Req 4.2).
    logger.info("pipeline_step=embed chunks_embedded=%d", len(chunks))
    return chunks


# ── Step 4: Store ────────────────────────────────────────────────────
def store(
    chunks: List[dict],
    collection_name: str,
    source_document: str,
    topic: str,
    chroma_client: chromadb.Client,
) -> None:
    """
    Persist embedded chunks to ChromaDB with full metadata (Req 4.4).

    Each chunk is stored with four metadata fields required by Req 4.4:
        - source_document : filename or document identifier
        - chunk_index     : position within the source document
        - topic           : subject label (e.g., "calculus")
        - chunk_type      : semantic category ("formula" | "explanation" | "example")

    These metadata fields enable filtered retrieval in the Adaptive Tutor
    (e.g., retrieve only "formula" chunks for a given topic).

    Args:
        chunks:           List of chunk dicts, each must have 'text',
                          'chunk_index', 'embedding', and optionally 'chunk_type'.
        collection_name:  ChromaDB collection to upsert into.
        source_document:  Identifier for the originating document (used in IDs
                          and metadata so chunks can be traced back to their source).
        topic:            Subject label stored in metadata for filtered queries.
        chroma_client:    An initialised chromadb.Client (HttpClient or EphemeralClient).
    """
    # get_or_create_collection is idempotent — safe to call on every ingestion.
    collection = chroma_client.get_or_create_collection(collection_name)

    # Build parallel lists required by the ChromaDB Python client API.
    ids = [f"{source_document}_{c['chunk_index']}" for c in chunks]
    embeddings = [c["embedding"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {
            # Req 4.4: all four metadata fields must be present on every chunk.
            "source_document": source_document,
            "chunk_index": c["chunk_index"],
            "topic": topic,
            "chunk_type": c.get("chunk_type", "explanation"),
        }
        for c in chunks
    ]

    # add() upserts by ID, so re-ingesting the same document is safe.
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    # Log step name and stored count (Req 4.2).
    logger.info(
        "pipeline_step=store chunks_stored=%d collection=%s",
        len(chunks),
        collection_name,
    )
