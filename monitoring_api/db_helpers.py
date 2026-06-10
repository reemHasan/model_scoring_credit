"""
Database connection and storage functions for PostgreSQL using SQLAlchemy.
─────────────────────────────────────────────────────────────────────────────
Responsibility: PostgreSQL storage using SQLAlchemy.

- ORM model defines the schema as a Python class
- AsyncEngine + AsyncSession for non-blocking DB writes
- One row per prediction in api_logs (features stored as JSON column)
- Alembic-ready: models are defined via DeclarativeBase
"""
import pathlib
from sqlalchemy import create_engine,text
from sqlalchemy.orm import Session
import pandas as pd
import os
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
    print("db url", f"{driver}://{user}:{password}@{host}:{port}/{dbname}")
    return f"{driver}://{user}:{password}@{host}:{port}/{dbname}"

# Create Sync engine — used by analysis.py / alembic 
sync_engine = create_engine(
    _build_url(async_=False),
    echo=False,
    pool_size=5,
)

#from sqlalchemy import inspect
#inspector = inspect(sync_engine)
#print("Table names:", inspector.get_table_names())

# Read helpers for analysis.py ──────────────────────────────────────────────
def load_logs_df():
    """
    Load all loan_api_logs into a pandas DataFrame (sync — for analysis scripts).
    """
    # with analmysis.py we can use a sync Session since it's not latency-sensitive and allows easier integration with Alembic for migrations.
    with Session(sync_engine) as session:
        rows = session.execute(
        text("SELECT * FROM loan_api_logs ORDER BY timestamp")).fetchall()
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
            "model_runtime": r.model_runtime,
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
        rows = session.execute(
            text("SELECT request_id, status_code,  timestamp, loan_id, proba_default, proba_class, features FROM loan_api_logs WHERE features IS NOT NULL ORDER BY timestamp")
        ).fetchall()
    if not rows:
        return pd.DataFrame()

    records = [
        {"request_id": r.request_id,
        "status_code": r.status_code,
         "timestamp":  r.timestamp,
         "loan_id":    r.loan_id,
         "proba_default": r.proba_default,
         "proba_class":   r.proba_class,
         **r.features}          # unpack all feature keys as columns
        for r in rows
         if r.status_code==200
    ]
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df