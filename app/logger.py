"""
logger.py
─────────────────────────────────────────────────────────────────────────────
Responsibility: structured JSON logging to stdout and file.

Every API call emits one JSON line like:
{
  "ts": "2026-05-14T10:23:01Z",
  "level": "INFO",
  "event": "prediction",
  "request_id": "abc123",
  "loan_id": 42,
  "proba": 0.34,
  "class": "Accept loan application",
  "decision": "Accept loan application",
  "inference_ms": 12.4,
  "total_ms": 18.1,
  "status_code": 200
}
"""

import os
import json
import logging
from datetime import datetime, timezone

class JsonFormatter(logging.Formatter):
    """Emit every log record as a single JSON line."""
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "ts":      datetime.now(timezone.utc).isoformat(),
            "level":   record.levelname,
            "logger":  record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra"):
            log_obj.update(record.extra)
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj, default=str)


def get_logger(name: str = "api") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:          # avoid duplicate handlers on reload
        return logger
    logger.setLevel(logging.DEBUG)

    # Console handler
    # The StreamHandler class, sends logging output to streams such as sys.stdout, sys.stderr
    ch = logging.StreamHandler()
    ch.setFormatter(JsonFormatter())
    # The addHandler() method adds the specified handler to the logger. 
    # Handler objects are responsible for dispatching the appropriate log messages (based on the log messages’ severity) to the handler’s specified destination.
    logger.addHandler(ch)

    # File handler — keeps a rolling log on disk as backup
    # The FileHandler class sends logging output to a disk file.
    # if not os.getenv("SPACE_ID"): apply if we don't want to write logs in Hugging Face Space (ephemeral storage)
    os.makedirs("logs", exist_ok=True)
    fh = logging.FileHandler("logs/api.log", encoding="utf-8")
    fh.setFormatter(JsonFormatter())
    logger.addHandler(fh)
    return logger


logger = get_logger("api")

def log_prediction(
    *,
    request_id:      str,
    loan_id:         int,
    proba_default:   float | None,
    proba_class:     str | None,
    decision:        str | None,
    inference_ms:    float,
    total_ms:        float,
    status_code:     int,
    error_message:   str | None,
    client_features: dict | None,
    shap_values:     dict | None,
):
    """Insert one row into api_logs """
    # python logger (JSON to stdout/file) ────────────────────────────────
    log_level = logging.ERROR if status_code >= 400 else logging.INFO
    logger.log(log_level, "prediction_request", extra={"extra": {
        "request_id":    request_id,
        "loan_id":       loan_id,
        "proba_default": proba_default,
        "class":         proba_class,
        "decision":      decision,
        "inference_ms":  inference_ms,
        "total_ms":      total_ms,
        "status_code":   status_code,
        "error":         error_message,
        "features":      client_features,
        "shap_values":   shap_values,
    }})
