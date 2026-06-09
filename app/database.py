"""
database.py
─────────────────────────────────────────────────────────────────────────────
Responsibility: PostgreSQL storage using SQLAlchemy.

- ORM model defines the schema as a Python class
- AsyncEngine + AsyncSession for non-blocking DB writes
- One row per prediction in api_logs (features stored as JSON column)
- Alembic-ready: models are defined via DeclarativeBase
"""

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    create_async_engine,
    async_sessionmaker,
)
from app.logger import logger
from app.models import Base, LoanApiLog
import os
import pathlib
from dotenv import load_dotenv
# Load .env from project root
load_dotenv(pathlib.Path(__file__).resolve().parents[1] / ".env")

# ── Connection URLs ───────────────────────────────────────────────────────────
def _build_url(async_: bool = False) -> str:
    host     = os.getenv("PGHOST")
    port     = os.getenv("PGPORT","5432")
    dbname   = os.getenv("PGDATABASE")
    user     = os.getenv("PGUSER")
    password = os.getenv("PGPASSWORD")
    # psycopg (v3) supports both async and sync, but SQLAlchemy's async engine specifically requires the psycopg v3 driver string.
    #  psycopg2 is the classic sync-only driver that Alembic and sync scripts expect.
    driver   = "postgresql+psycopg" if async_ else "postgresql+psycopg2"
    return f"{driver}://{user}:{password}@{host}:{port}/{dbname}"


# Create an Async engine — used by FastAPI (for non-blocking) 
async_engine = create_async_engine(
    _build_url(async_=True),
    echo=True,          # set True to log all SQL statements (debug only)
    pool_size=5,
    max_overflow=10,
)
# Cereate an async session factory — used by store_record() to get sessions
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# DB initialisation ─────────────────────────────────────────────────────────
async def init_db():
    """
    Create tables from ORM models.
    Called once at FastAPI startup (inside lifespan).
    Safe to call multiple times — uses checkfirst=True.
    """
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all, checkfirst=True)
    logger.info("db_init", extra={"extra": {"event": "db_init", "status": "ok"}})


# Write one prediction record ───────────────────────────────────────────────
async def store_record(
    *,
    request_id:    str,
    loan_id:       int,
    proba_default: float | None,
    proba_class:   str   | None,
    decision:      str   | None,
    inference_ms:  float,
    total_ms:      float,
    status_code:   int,
    error_message: str   | None,
    features:      dict  | None,
    shap_values:   dict  | None,
    model_runtime: str   | None,
):
    """
    Async insert — called as a FastAPI BackgroundTask so it runs after the response is already sent to the user.
    A failure here never affects API latency or availability.
    """
    record = LoanApiLog(
        request_id=    request_id,
        loan_id=       loan_id,
        proba_default= proba_default,
        proba_class=   proba_class,
        decision=      decision,
        inference_ms=  inference_ms,
        total_ms=      total_ms,
        status_code=   status_code,
        error_message= error_message,
        features=      features,
        shap_values=   shap_values,
        model_runtime= model_runtime,
    )
    try:
        # insert info into PostgreSQL — if this fails, we log the error but do not re-raise (DB failure should not affect API availability)
        async with AsyncSessionLocal() as session:
            session.add(record)
            await session.commit()
    except Exception as exc:
        # DB failure is logged but never re-raised
        logger.error("db_write_failed", extra={"extra": {
            "event":      "db_error",
            "request_id": request_id,
            "error":      str(exc),
        }})
        await session.rollback()
        raise exc
