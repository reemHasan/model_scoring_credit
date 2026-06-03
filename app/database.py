"""
db_store.py
─────────────────────────────────────────────────────────────────────────────
Responsibility: PostgreSQL storage using SQLAlchemy.

- ORM model defines the schema as a Python class
- AsyncEngine + AsyncSession for non-blocking DB writes
- One row per prediction in api_logs (features stored as JSON column)
- Alembic-ready: models are defined via DeclarativeBase
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy.orm import Session
from app.logger import logger
from app.models import Base, LoanApiLog
import pandas as pd
import os
import json
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

# Create Sync engine — used by analysis.py / alembic 
sync_engine = create_engine(
    _build_url(async_=False),
    echo=False,
    pool_size=5,
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


# Read helpers for analysis.py ──────────────────────────────────────────────
def load_logs_df():
    """
    Load all loan_api_logs into a pandas DataFrame (sync — for analysis scripts).
    """
    # with analmysis.py we can use a sync Session since it's not latency-sensitive and allows easier integration with Alembic for migrations.
    with Session(sync_engine) as session:
        rows = session.query(LoanApiLog).order_by(LoanApiLog.timestamp).all()

    if not rows:
        return pd.DataFrame()

    records = [
        {
            "request_id":    r.request_id,
            "timestamp":     r.timestamp,
            "loan_id":       r.loan_id,
            "proba_default": r.proba_default,
            "proba_class":   r.proba_class,
            "decision":      r.decision,
            "inference_ms":  r.inference_ms,
            "total_ms":      r.total_ms,
            "status_code":   r.status_code,
            "error_message": r.error_message,
        }
        for r in rows
    ]
    base_df = pd.DataFrame(records)
    base_df["timestamp"] = pd.to_datetime(base_df["timestamp"], utc=True)
    return base_df


def load_features_wide_df():
    """
    Load features JSON column and expand into wide format for Evidently.
    Returns one row per prediction, one column per feature.
    """

    with Session(sync_engine) as session:
        rows = session.query(
            LoanApiLog.request_id,
            LoanApiLog.timestamp,
            LoanApiLog.loan_id,
            LoanApiLog.proba_default,
            LoanApiLog.features,
        ).filter(LoanApiLog.features.isnot(None)).order_by(LoanApiLog.timestamp).all()

    if not rows:
        return pd.DataFrame()

    records = [
        {"request_id": r.request_id,
         "timestamp":  r.timestamp,
         "loan_id":    r.loan_id,
         "proba_default": r.proba_default,
         "proba_class":   r.proba_class,
         **r.features}          # unpack all feature keys as columns
        for r in rows
    ]
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def parse_and_store_log_file(log_path: str = "logs/api.log"):
    """
    Batch job: read the JSON log file line by line and insert into PostgreSQL.
    Use this when running analysis locally after downloading logs from HF Space.

    Usage:
        python db_store.py
    """

    path = pathlib.Path(log_path)
    if not path.exists():
        logger.warning(f"Log file not found: {log_path}")
        return

    inserted = 0
    skipped  = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                # Only process prediction events
                if record.get("event") != "prediction":
                    continue

                store_record(
                    request_id=    record.get("request_id", "unknown"),
                    loan_id=       record.get("loan_id", -1),
                    proba_default= record.get("proba"),
                    proba_class=   record.get("class"),
                    decision=      record.get("decision"),
                    inference_ms=  record.get("inference_ms", 0),
                    total_ms=      record.get("total_ms", 0),
                    status_code=   record.get("status_code", 200),
                    error_message= record.get("error"),
                    features=      record.get("features"),
                    shap_values=   record.get("shap_values"),
                )
                inserted += 1
            except (json.JSONDecodeError, KeyError) as exc:
                skipped += 1
                logger.warning(f"Skipped malformed log line: {exc}")

    logger.info(f"Log import complete: {inserted} inserted, {skipped} skipped")


if __name__ == "__main__":
    init_db()
    parse_and_store_log_file()