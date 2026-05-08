# doc_to_concept/router.py
# ─────────────────────────────────────────────────────────────────────
# FastAPI router for the Doc-to-Concept Agent (Path C).
# Mounted at /api/docs/* in main.py.
#
# Endpoints:
#   POST /api/docs/upload
#       Accepts a PDF file upload and an optional topic label.
#       Validates the file extension, generates a UUID document_id,
#       builds the initial DocState, and invokes the LangGraph pipeline.
#       Returns the document_id so the client can poll for results.
#       Catches ValueError (from LlamaParse or invalid PDF) and returns
#       HTTP 422 with a descriptive message (Req 8.9).
#
#   GET /api/docs/{document_id}/concepts
#       Queries PostgreSQL for the processed document and returns the
#       extracted formulas and synthetic examples (Req 8.8).
#       Returns HTTP 404 if the document_id is not found.
#
# Requirements satisfied: 8.1 (upload endpoint), 8.8 (concepts endpoint),
#                         8.9 (422 on invalid PDF / LlamaParse error).
# ─────────────────────────────────────────────────────────────────────
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from database import get_session
from doc_to_concept.graph import build_doc_graph
from models.api_models import DocConceptsResponse
from models.db_models import Document

# ── Router setup ──────────────────────────────────────────────────────
# prefix="/api/docs" means all routes below are relative to /api/docs.
# tags=["doc_to_concept"] groups these endpoints in the OpenAPI docs UI.
router = APIRouter(prefix="/api/docs", tags=["doc_to_concept"])

# ── Compile the graph once at module load time ────────────────────────
# build_doc_graph() validates the graph structure and returns a compiled
# CompiledGraph. Compiling once here (rather than per-request) avoids
# repeated validation overhead on every upload.
_graph = build_doc_graph()


# ── POST /upload ──────────────────────────────────────────────────────
@router.post("/upload")
async def upload_pdf(
    file: UploadFile = File(..., description="PDF file to analyse"),
    topic: str = Form(default="", description="Optional math topic label (e.g., 'calculus')"),
) -> dict:
    """
    Accept a PDF upload, run the Doc-to-Concept pipeline, and return the document_id.

    The pipeline runs synchronously inside this request handler:
        1. Validate the file extension (.pdf required).
        2. Generate a UUID document_id.
        3. Read the raw PDF bytes from the upload.
        4. Build the initial DocState dict.
        5. Invoke the LangGraph pipeline (_graph.invoke).
        6. Return {"document_id": <uuid>} on success.

    The client uses the returned document_id to call
    GET /api/docs/{document_id}/concepts once processing is complete.

    Error handling:
        - Non-.pdf extension → HTTP 422 (Req 8.9)
        - LlamaParse failure → HTTP 422 via ValueError catch (Req 8.9)
        - Pydantic validation errors → HTTP 422 (automatic, Req 12.3)

    Args:
        file:  The uploaded PDF file (multipart/form-data).
        topic: Optional subject label stored in ChromaDB chunk metadata
               and the PostgreSQL documents table.

    Returns:
        JSON dict: {"document_id": "<uuid>"}
    """
    # ── Validate file extension ───────────────────────────────────────
    # Check the filename rather than the MIME type because browsers
    # sometimes send "application/octet-stream" for PDFs. A filename
    # check is simple, transparent, and sufficient for this demo.
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Uploaded file '{file.filename}' is not a PDF. "
                "Please upload a file with a .pdf extension."
            ),
        )

    # ── Generate a stable document identifier ────────────────────────
    # UUID4 is random and collision-resistant. It is used as:
    #   - The PostgreSQL documents.document_id primary key
    #   - The prefix for ChromaDB chunk IDs (document_id_<chunk_index>)
    #   - The path parameter in GET /api/docs/{document_id}/concepts
    document_id = str(uuid.uuid4())

    # ── Read PDF bytes ────────────────────────────────────────────────
    # await file.read() loads the entire file into memory. For a demo
    # system this is acceptable; a production system would stream to
    # object storage instead.
    pdf_bytes = await file.read()

    # ── Build initial state and invoke the pipeline ───────────────────
    initial_state = {
        "document_id": document_id,
        "filename": file.filename,
        "topic": topic.strip() or None,
        "pdf_bytes": pdf_bytes,
        "markdown_text": "",
        "chunks": [],
        "formula_chunks": [],
        "formulas": [],
        "examples": {},
    }

    try:
        # _graph.invoke() runs the full pipeline synchronously:
        # parse → chunk → classify → embed_store → extract_formulas
        # → generate_examples → persist
        _graph.invoke(initial_state)
    except ValueError as exc:
        # ValueError is raised by llamaparse_client.py for:
        #   - LlamaParse returning no content (empty/scanned PDF)
        #   - LlamaParse API errors (auth failure, quota exceeded, etc.)
        # We surface these as HTTP 422 with the descriptive message from
        # the ValueError so the student sees a useful error (Req 8.9).
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {"document_id": document_id}


# ── GET /{document_id}/concepts ───────────────────────────────────────
@router.get("/{document_id}/concepts", response_model=DocConceptsResponse)
async def get_concepts(document_id: str) -> DocConceptsResponse:
    """
    Return the extracted formulas and synthetic examples for a processed document.

    Queries the PostgreSQL ``documents`` table by document_id and returns
    the full DocConceptsResponse. The frontend renders formulas in monospace
    code blocks and worked examples in prose below each formula (Req 11.6).

    Args:
        document_id: UUID string returned by POST /upload.

    Returns:
        DocConceptsResponse with formulas, examples, and metadata.

    Raises:
        HTTPException(404): If no document with the given ID exists in
                            PostgreSQL (e.g., pipeline not yet complete,
                            or invalid ID).
    """
    with get_session() as db:
        # Query by primary key — filter_by is equivalent to WHERE document_id = ?
        doc = db.query(Document).filter_by(document_id=document_id).first()

        if not doc:
            raise HTTPException(
                status_code=404,
                detail=f"Document '{document_id}' not found. "
                       "The document may still be processing or the ID may be invalid.",
            )

        # Build the response inside the session context so SQLAlchemy
        # can access lazy-loaded attributes before the session closes.
        return DocConceptsResponse(
            document_id=doc.document_id,
            filename=doc.filename or "",
            topic=doc.topic,
            formulas=doc.formulas or [],
            examples=doc.examples or {},
            # chunk_count is stored as String in the DB; cast to int here.
            chunk_count=int(doc.chunk_count or 0),
        )
