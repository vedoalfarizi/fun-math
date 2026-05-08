# database.py
# ─────────────────────────────────────────────────────────────────────
# SQLAlchemy database setup for the Agentic Math Learning System.
#
# This module provides three things students need to understand:
#   1. engine       — the connection pool to PostgreSQL
#   2. SessionLocal — a factory that creates individual DB sessions
#   3. get_session  — a context-manager helper used in every node that
#                     needs to read or write persistent state
#
# All three agentic paths (Tutor, Practice, Doc-to-Concept) import
# get_session() from here so database access is consistent.
# ─────────────────────────────────────────────────────────────────────
import logging
import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)


# ── Declarative base ──────────────────────────────────────────────────
# All ORM models (TutorSession, PracticeSession, Document) inherit from
# Base so that Base.metadata.create_all() can create every table in one
# call at application startup.
class Base(DeclarativeBase):
    """
    SQLAlchemy declarative base shared by all ORM models.

    Import Base in db_models.py and inherit from it for each table.
    Calling Base.metadata.create_all(engine) at startup ensures all
    tables exist before the first request is handled.
    """
    pass


# ── Engine ────────────────────────────────────────────────────────────
# Read the DSN from the environment — never hardcode credentials.
# POSTGRES_DSN format: postgresql://user:password@host:port/dbname
# In the Docker Compose network the host is the service name "postgres".
def _create_engine():
    dsn = os.environ["POSTGRES_DSN"]
    # pool_pre_ping=True makes SQLAlchemy test connections before use,
    # which prevents "server closed the connection unexpectedly" errors
    # after the Postgres container restarts.
    return create_engine(dsn, pool_pre_ping=True)


engine = _create_engine()

# ── Session factory ───────────────────────────────────────────────────
# SessionLocal is a class; calling SessionLocal() creates a new session.
# autocommit=False means we must call db.commit() explicitly — this
# makes transaction boundaries visible and educational.
# autoflush=False prevents SQLAlchemy from issuing unexpected SQL during
# attribute access, keeping behaviour predictable for students.
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)


# ── Session context manager ───────────────────────────────────────────
@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Provide a transactional database session as a context manager.

    Usage in node functions:
        from database import get_session
        with get_session() as db:
            record = db.query(TutorSession).filter_by(...).first()
            db.commit()

    The session is automatically closed (and rolled back on exception)
    when the `with` block exits, preventing connection leaks.
    """
    db: Session = SessionLocal()
    try:
        yield db
    except Exception:
        # Roll back any partial writes so the DB stays consistent.
        db.rollback()
        logger.exception("Database session rolled back due to exception")
        raise
    finally:
        # Always close the session to return the connection to the pool.
        db.close()


# ── Table creation ────────────────────────────────────────────────────
def create_tables() -> None:
    """
    Create all tables defined by ORM models that inherit from Base.

    Called once at application startup (from main.py) so the schema
    is always in sync with the ORM definitions without requiring a
    separate migration step for this demo application.

    Note: In production you would use Alembic migrations instead.
    """
    # Import models here to ensure they are registered with Base.metadata
    # before create_all() is called.  The import is intentionally deferred
    # to avoid circular imports at module load time.
    #
    # IMPORTANT: db_models.Base is the declarative base that all ORM models
    # (TutorSession, PracticeSession, Document) inherit from.  We must call
    # create_all() on *that* Base — not the one defined above in this module —
    # otherwise no tables are created because no models are registered here.
    from models.db_models import Base as ModelsBase  # noqa: F401

    ModelsBase.metadata.create_all(bind=engine)
    logger.info("database tables created (or already exist)")
