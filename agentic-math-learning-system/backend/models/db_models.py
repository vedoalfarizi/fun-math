# models/db_models.py
# ─────────────────────────────────────────────────────────────────────
# SQLAlchemy ORM models for all three agentic paths.
# Each model maps to a PostgreSQL table and stores session/document
# state as JSON columns so the schema stays flexible while the
# relational layer handles identity and timestamps.
#
# Base.metadata.create_all() is called in main.py on startup so tables
# are created automatically — no manual migration needed for the demo.
# ─────────────────────────────────────────────────────────────────────
from sqlalchemy import Column, String, Integer, JSON, DateTime, func
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """
    Shared declarative base for all ORM models.

    All models inherit from this class so that Base.metadata.create_all()
    in main.py and database.py can discover and create every table in a
    single call.
    """
    pass


class TutorSession(Base):
    """
    Persists each Adaptive Tutor (Path A) conversation session.

    A session is identified by a UUID session_id supplied by the frontend.
    Each call to POST /api/tutor/chat appends a turn dict to the `turns`
    JSON column, keeping the full conversation history in one row.

    Satisfies Req 6.7 (persist each turn) and Req 6.8 (create session on
    first turn).

    Columns:
        session_id   — Primary key; UUID string supplied by the client.
        turns        — JSON list of {query, context_chunks, response} dicts.
        created_at   — Server-side timestamp set on INSERT.
        updated_at   — Server-side timestamp refreshed on every UPDATE.
    """

    __tablename__ = "tutor_sessions"

    # Primary key — UUID string provided by the frontend on session start
    session_id = Column(String, primary_key=True, nullable=False)

    # JSON list of turn dicts: [{query, context_chunks, response}, ...]
    # default=list ensures a fresh list per row rather than a shared mutable default
    turns = Column(JSON, nullable=False, default=list)

    # Timestamps — server_default lets PostgreSQL set these without Python involvement
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=True,
    )


class PracticeSession(Base):
    """
    Persists the full state machine history for a Practice Agent (Path B) session.

    One row is created when POST /api/practice/start is called. Each answer
    submission via POST /api/practice/answer appends a history entry containing
    the question, the student's answer, the outcome, and any Deep Dive
    explanation that was generated.

    Satisfies Req 7.7 (persist full state machine history).

    Columns:
        session_id       — Primary key; UUID string.
        topic            — Math topic used for MCQ generation.
        difficulty_level — "easy" | "medium" | "hard".
        history          — JSON list of {question, answer, outcome, explanation} dicts.
        created_at       — Server-side timestamp set on INSERT.
    """

    __tablename__ = "practice_sessions"

    # Primary key — UUID string
    session_id = Column(String, primary_key=True, nullable=False)

    # Topic and difficulty are stored so the session can be resumed or reviewed
    topic = Column(String, nullable=True)
    difficulty_level = Column(String, nullable=True)

    # JSON list of turn dicts:
    # [{question_text, selected_option, correct_option, outcome, explanation}, ...]
    history = Column(JSON, nullable=False, default=list)

    # Timestamp — set once on INSERT; no updated_at needed (append-only history)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class Document(Base):
    """
    Persists Doc-to-Concept Agent (Path C) results for a processed PDF.

    One row is created at the end of the LangGraph pipeline after all
    formulas have been extracted and synthetic examples generated. The
    GET /api/docs/{document_id}/concepts endpoint reads from this table.

    Satisfies Req 8.8 (expose GET endpoint returning formulas + examples).

    Columns:
        document_id  — Primary key; UUID string generated at upload time.
        filename     — Original PDF filename for display purposes.
        topic        — Optional topic label supplied by the student on upload.
        formulas     — JSON list of LaTeX formula strings extracted from the PDF.
        examples     — JSON dict mapping each formula to a list of worked examples:
                       {formula_string: [example_1, example_2, ...]}.
        chunk_count  — Number of chunks produced during the chunking step,
                       stored as a string to avoid a schema migration if the
                       type needs to change (cast to int at read time).
        created_at   — Server-side timestamp set on INSERT.
    """

    __tablename__ = "documents"

    # Primary key — UUID string generated in doc_to_concept/router.py
    document_id = Column(String, primary_key=True, nullable=False)

    # Metadata about the source file
    filename = Column(String, nullable=True)
    topic = Column(String, nullable=True)

    # Extracted content — both stored as JSON for schema flexibility
    formulas = Column(JSON, nullable=False, default=list)   # [latex_string, ...]
    examples = Column(JSON, nullable=False, default=dict)   # {formula: [ex1, ex2, ...]}

    # Chunk count stored as String; cast to int in the API response model
    chunk_count = Column(String, nullable=True)

    # Timestamp — set once on INSERT
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
