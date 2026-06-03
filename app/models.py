

from sqlalchemy import (
    Integer, Float, Text, DateTime, JSON)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from datetime import datetime, timezone
"""
This file defines the ORM model for the API logs

"""
# ORM Model ─────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


class LoanApiLog(Base):
    """
    One row per API prediction call.
    features and shap_values stored as JSON — schema-flexible,
    """
    __tablename__ = "loan_api_logs"

    id:            Mapped[int]            = mapped_column(Integer,  primary_key=True, autoincrement=True)
    request_id:    Mapped[str]            = mapped_column(Text,     nullable=False)
    timestamp:     Mapped[datetime]       = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    loan_id:       Mapped[int]            = mapped_column(Integer,  nullable=False)
    proba_default: Mapped[float | None]   = mapped_column(Float,    nullable=True)
    proba_class:   Mapped[str | None]     = mapped_column(Text,     nullable=True)
    decision:      Mapped[str | None]     = mapped_column(Text,     nullable=True)
    inference_ms:  Mapped[float | None]   = mapped_column(Float,    nullable=True)
    total_ms:      Mapped[float | None]   = mapped_column(Float,    nullable=True)
    status_code:   Mapped[int]            = mapped_column(Integer,  nullable=False, default=200)
    error_message: Mapped[str | None]     = mapped_column(Text,     nullable=True)
    features:      Mapped[dict | None]    = mapped_column(JSON,     nullable=True)   # all input features in one row
    shap_values:   Mapped[dict | None]    = mapped_column(JSON,     nullable=True)

