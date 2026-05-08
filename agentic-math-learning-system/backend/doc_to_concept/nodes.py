# doc_to_concept/nodes.py
# ─────────────────────────────────────────────────────────────────────
# LangGraph node functions for the Doc-to-Concept pipeline (Path C).
#
# Each function is a pure transformation: it receives the current
# DocState dict, performs one pipeline step, and returns a dict
# containing only the fields it modified. LangGraph merges the returned
# dict back into the running state automatically.
#
# Pipeline order (defined in graph.py):
#   parse_node → chunk_node → classify_node → embed_store_node
#   → extract_formulas_node → generate_examples_node → persist_node
#
# Requirements satisfied:
#   8.2 (parse_node — PDF → Markdown via LlamaParse)
#   8.3 (chunk_node — semantic chunking with CHUNK_SIZE constant)
#   8.4 (classify_node — LLM classifies each chunk)
#   8.5 (embed_store_node — embed + store with full metadata)
#   8.6 (extract_formulas_node — extract formulas from formula chunks)
#   8.7 (generate_examples_node — ≥2 examples per formula)
#   8.8 (persist_node — write Document record to PostgreSQL)
# ─────────────────────────────────────────────────────────────────────
import logging
import os
from typing import List

import chromadb

from config import CLASSIFY_CONFIG
from database import get_session
from doc_to_concept.llamaparse_client import parse_pdf_to_markdown
from doc_to_concept.state import DocState
from doc_to_concept.tools import extract_formulas_tool, generate_examples_tool
from embeddings.pipeline import chunk, embed, store
from llm_wrapper import call_llm
from models.db_models import Document
from prompts import CHUNK_CLASSIFY_PROMPT

logger = logging.getLogger(__name__)


# ── Node 1: parse_node ────────────────────────────────────────────────
def parse_node(state: DocState) -> dict:
    """
    Convert the uploaded PDF bytes to a Markdown string via LlamaParse.

    This is the entry point of the pipeline. It delegates to
    ``parse_pdf_to_markdown`` which handles the LlamaParse API call,
    temp-file management, and error wrapping.

    Reads:  state["pdf_bytes"], state["filename"]
    Writes: state["markdown_text"]

    Raises:
        ValueError: Propagated from llamaparse_client if LlamaParse fails.
                    The router catches this and returns HTTP 422 (Req 8.9).
    """
    logger.info(
        "doc_pipeline node=parse document_id=%s filename=%s",
        state["document_id"],
        state["filename"],
    )
    markdown_text = parse_pdf_to_markdown(
        pdf_bytes=state["pdf_bytes"],
        filename=state["filename"],
    )
    logger.info(
        "doc_pipeline node=parse complete markdown_chars=%d",
        len(markdown_text),
    )
    return {"markdown_text": markdown_text}


# ── Node 2: chunk_node ────────────────────────────────────────────────
def chunk_node(state: DocState) -> dict:
    """
    Segment the Markdown text into chunks of approximately CHUNK_SIZE words.

    Delegates to the shared ``chunk()`` function from embeddings/pipeline.py
    which applies word-boundary splitting. The chunk_type field on each
    chunk is initialised to "explanation" and will be overridden by
    classify_node in the next step.

    Reads:  state["markdown_text"]
    Writes: state["chunks"]
    """
    logger.info(
        "doc_pipeline node=chunk document_id=%s",
        state["document_id"],
    )
    # chunk() logs pipeline_step=chunk and chunks_produced=N at INFO level (Req 4.2)
    chunks = chunk(state["markdown_text"])
    logger.info(
        "doc_pipeline node=chunk complete chunks=%d",
        len(chunks),
    )
    return {"chunks": chunks}


# ── Node 3: classify_node ─────────────────────────────────────────────
def classify_node(state: DocState) -> dict:
    """
    Classify each chunk as "formula", "explanation", or "example" using the LLM.

    Calls the LLM once per chunk with CHUNK_CLASSIFY_PROMPT and CLASSIFY_CONFIG
    (temperature=0.1 for deterministic, consistent labels). The chunk_type field
    on each chunk dict is updated in-place with the classification result.

    After classification, formula_chunks is populated with the subset of chunks
    whose chunk_type is "formula" — these are passed to extract_formulas_node.

    Reads:  state["chunks"]
    Writes: state["chunks"] (with updated chunk_type), state["formula_chunks"]

    Satisfies Req 8.4 (classify each chunk; store classification in metadata).
    """
    logger.info(
        "doc_pipeline node=classify document_id=%s chunks_to_classify=%d",
        state["document_id"],
        len(state["chunks"]),
    )

    classified_chunks: List[dict] = []
    for chunk_dict in state["chunks"]:
        prompt = CHUNK_CLASSIFY_PROMPT.format(chunk_text=chunk_dict["text"])
        # CLASSIFY_CONFIG: temperature=0.1 — deterministic classification
        # prevents the same chunk from being labelled differently on re-runs.
        raw_label = call_llm(prompt, CLASSIFY_CONFIG, task_name="classify_chunk").strip().lower()

        # Normalise the label to one of the three valid values.
        # The prompt instructs the LLM to output only the label, but we
        # guard against extra whitespace or punctuation just in case.
        if "formula" in raw_label:
            chunk_type = "formula"
        elif "example" in raw_label:
            chunk_type = "example"
        else:
            # Default to "explanation" for any ambiguous or unexpected output.
            chunk_type = "explanation"

        classified_chunks.append({**chunk_dict, "chunk_type": chunk_type})

    # Collect formula chunks for the extract_formulas_node.
    # Filtering here avoids re-scanning all chunks in the next node.
    formula_chunks = [c for c in classified_chunks if c["chunk_type"] == "formula"]

    logger.info(
        "doc_pipeline node=classify complete formula=%d explanation=%d example=%d",
        len(formula_chunks),
        sum(1 for c in classified_chunks if c["chunk_type"] == "explanation"),
        sum(1 for c in classified_chunks if c["chunk_type"] == "example"),
    )
    return {"chunks": classified_chunks, "formula_chunks": formula_chunks}


# ── Node 4: embed_store_node ──────────────────────────────────────────
def embed_store_node(state: DocState) -> dict:
    """
    Vectorise all chunks and persist them to ChromaDB with full metadata.

    Calls ``embed()`` to add an 'embedding' key to each chunk dict, then
    calls ``store()`` to write all chunks to the "math_knowledge" ChromaDB
    collection. The metadata stored per chunk includes source_document,
    chunk_index, topic, and chunk_type (Req 4.4).

    Reads:  state["chunks"], state["document_id"], state["filename"], state["topic"]
    Writes: (side-effect only — ChromaDB is updated; state is returned unchanged)

    Satisfies Req 8.5 (embed and store each chunk with full metadata).
    """
    logger.info(
        "doc_pipeline node=embed_store document_id=%s chunks=%d",
        state["document_id"],
        len(state["chunks"]),
    )

    # embed() adds an 'embedding' key (list of floats) to each chunk dict.
    # It logs pipeline_step=embed and chunks_embedded=N at INFO level (Req 4.2).
    embedded_chunks = embed(state["chunks"])

    # Build the ChromaDB client from environment variables (Req 2.4).
    chroma_client = chromadb.HttpClient(
        host=os.environ["CHROMA_HOST"],
        port=int(os.environ.get("CHROMA_PORT", 8000)),
    )

    # store() persists chunks with all four required metadata fields (Req 4.4):
    # source_document, chunk_index, topic, chunk_type.
    # It logs pipeline_step=store and chunks_stored=N at INFO level (Req 4.2).
    store(
        chunks=embedded_chunks,
        collection_name="math_knowledge",
        source_document=state["document_id"],   # Use UUID as the stable identifier
        topic=state.get("topic") or "",
        chroma_client=chroma_client,
    )

    logger.info(
        "doc_pipeline node=embed_store complete document_id=%s",
        state["document_id"],
    )
    # Return the updated chunks (now with embeddings) so downstream nodes
    # have access to the full chunk data if needed.
    return {"chunks": embedded_chunks}


# ── Node 5: extract_formulas_node ─────────────────────────────────────
def extract_formulas_node(state: DocState) -> dict:
    """
    Extract all distinct mathematical formulas from the formula-classified chunks.

    Calls ``extract_formulas_tool`` once per formula chunk. The tool uses
    FORMULA_EXTRACT_PROMPT with FORMULA_EXTRACT_CONFIG (temperature=0.1) to
    extract LaTeX-formatted formulas as structured JSON objects.

    Deduplicates formulas across chunks by formula string to avoid storing
    the same formula multiple times when it appears in several chunks.

    Reads:  state["formula_chunks"]
    Writes: state["formulas"]

    Satisfies Req 8.6 (extract formulas from formula-classified chunks).
    """
    logger.info(
        "doc_pipeline node=extract_formulas document_id=%s formula_chunks=%d",
        state["document_id"],
        len(state["formula_chunks"]),
    )

    all_formulas: List[dict] = []
    seen_formulas: set = set()

    for formula_chunk in state["formula_chunks"]:
        # extract_formulas_tool is a @tool-decorated function; call it directly
        # by passing the chunk text. LangGraph exposes it to the LLM via its
        # docstring, but here we invoke it directly as a Python function.
        extracted = extract_formulas_tool.invoke({"chunk_text": formula_chunk["text"]})

        for formula_obj in extracted:
            formula_str = formula_obj.get("formula", "").strip()
            # Deduplicate: skip formulas we've already seen in earlier chunks.
            if formula_str and formula_str not in seen_formulas:
                seen_formulas.add(formula_str)
                all_formulas.append(formula_obj)

    logger.info(
        "doc_pipeline node=extract_formulas complete formulas_found=%d",
        len(all_formulas),
    )
    return {"formulas": all_formulas}


# ── Node 6: generate_examples_node ────────────────────────────────────
def generate_examples_node(state: DocState) -> dict:
    """
    Generate at least two synthetic worked examples for each extracted formula.

    Calls ``generate_examples_tool`` once per formula. The tool uses
    EXAMPLE_GEN_PROMPT with EXAMPLE_GEN_CONFIG (temperature=0.8) to encourage
    variety across examples (Req 8.7).

    The resulting examples dict maps each formula string to a list of example
    strings. The minimum of two examples per formula is enforced by the tool
    itself (with a placeholder appended if the LLM returns fewer).

    Reads:  state["formulas"], state["topic"]
    Writes: state["examples"]

    Satisfies Req 8.7 (≥2 synthetic worked examples per formula).
    """
    logger.info(
        "doc_pipeline node=generate_examples document_id=%s formulas=%d",
        state["document_id"],
        len(state["formulas"]),
    )

    examples: dict = {}
    topic = state.get("topic") or "mathematics"

    for formula_obj in state["formulas"]:
        formula_str = formula_obj.get("formula", "")
        description = formula_obj.get("description", "")

        if not formula_str:
            continue

        # generate_examples_tool is a @tool-decorated function; invoke directly.
        # High temperature (0.8) in EXAMPLE_GEN_CONFIG ensures variety (Req 8.7).
        formula_examples = generate_examples_tool.invoke({
            "formula": formula_str,
            "description": description,
            "topic": topic,
        })

        # The tool guarantees len(formula_examples) >= 2 (Req 8.7).
        examples[formula_str] = formula_examples
        logger.info(
            "doc_pipeline node=generate_examples formula=%r examples_generated=%d",
            formula_str[:60],
            len(formula_examples),
        )

    logger.info(
        "doc_pipeline node=generate_examples complete total_formulas_with_examples=%d",
        len(examples),
    )
    return {"examples": examples}


# ── Node 7: persist_node ──────────────────────────────────────────────
def persist_node(state: DocState) -> dict:
    """
    Write the completed Document record to the PostgreSQL ``documents`` table.

    Creates a new Document ORM instance with all extracted content and
    commits it via the get_session() context manager. The document_id
    (UUID generated in the router) is used as the primary key so the
    GET /api/docs/{document_id}/concepts endpoint can look it up directly.

    Reads:  state["document_id"], state["filename"], state["topic"],
            state["formulas"], state["examples"], state["chunks"]
    Writes: (side-effect only — PostgreSQL is updated; state is returned unchanged)

    Satisfies Req 8.8 (persist results; expose via GET endpoint).
    """
    logger.info(
        "doc_pipeline node=persist document_id=%s formulas=%d",
        state["document_id"],
        len(state["formulas"]),
    )

    # Extract just the formula strings for the formulas column.
    # The full formula objects (with descriptions) are not needed in the
    # DB because the API response only returns formula strings + examples.
    formula_strings = [
        f.get("formula", "") for f in state["formulas"] if f.get("formula")
    ]

    with get_session() as db:
        document = Document(
            document_id=state["document_id"],
            filename=state["filename"],
            topic=state.get("topic") or None,
            formulas=formula_strings,
            examples=state["examples"],
            # Store chunk count as a string (matches the String column type
            # in db_models.py; cast to int in the API response model).
            chunk_count=str(len(state["chunks"])),
        )
        db.add(document)
        db.commit()

    logger.info(
        "doc_pipeline node=persist complete document_id=%s",
        state["document_id"],
    )
    # No state fields are modified by this node — return empty dict.
    return {}
