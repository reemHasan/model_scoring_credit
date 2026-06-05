# ── Stage 1: builder — install dependencies with uv ──────────────────────────
FROM python:3.12-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Copy dependency files first (layer caching — only re-runs if deps change)
COPY pyproject.toml uv.lock* ./

# Install dependencies into /app/.venv
# --frozen: respect uv.lock exactly
# --no-dev:  skip dev/test tools in production image
RUN uv sync --frozen --no-dev

# ── Stage 2: runtime — lean final image ──────────────────────────────────────
FROM python:3.12-slim AS runtime

# HuggingFace Spaces runs as a non-root user (uid=1000)
RUN useradd -m -u 1000 appuser

# Install OpenMP library for LightGBM
#RUN apt-get update && apt-get install -y libgomp1 && apt-get clean && rm -rf /var/lib/apt/lists/*
# Install OpenMP runtime — required by LightGBM, numba, scikit-learn
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY app/ ./app/
# Copy model and data (large files — kept in final image for HF Spaces)
# paths relative to build context (project root)
COPY ml/model/lgbm_bestmodel_fbeta10_bundle.pkl ./ml/model/lgbm_bestmodel_fbeta10_bundle.pkl
COPY data/prod_data/ ./data/prod_data/

# COPY app/gradio_app.py  ./app/gradio_app.py

# Create logs directory (writable by appuser)
#RUN mkdir -p logs && chown -R appuser:appuser /app
#USER appuser

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app/app"
    
# HuggingFace Spaces exposes port 7860
EXPOSE 7860
#EXPOSE 8000
# Start FastAPI + Gradio on port 7860
# api.py mounts Gradio via mount_gradio_app — one process, one port
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "7860"]
# test docker in local with uvicorn on 8000
#CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]