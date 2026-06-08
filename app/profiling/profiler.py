# app/profiling/profiler_pyInstrument .py
import os
from pathlib import Path
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from pyinstrument import Profiler
"""
Usage:
1. run api : $env:ENABLE_PROFILING="true"; uvicorn app.api:app --port 8000 --reload
2. call api with profiling enabled:  1..20 | ForEach-Object {Invoke-RestMethod -Method POST -Uri "http://127.0.0.1:8000/predict/$_"}
"""

class ProfilerMiddleware(BaseHTTPMiddleware):
    """
    Profiles every /predict request with PyInstrument.
    Only active when ENABLE_PROFILING=true env var is set.
    Saves both HTML (flame graph) and text reports.
    """
    async def dispatch(self, request: Request, call_next):

        if os.getenv("ENABLE_PROFILING") != "true":
            return await call_next(request)

        if "/predict" not in request.url.path:
            return await call_next(request)

        # PyInstrument — async=True understands await correctly
        profiler = Profiler(async_mode="enabled")

        with profiler:
            response = await call_next(request)

        # ── Save reports ──────────────────────────────────────────────────────
        loan_id  = request.url.path.split("/")[-1]
        out_dir  = Path("profiling_output/profiles")
        out_dir.mkdir(parents=True, exist_ok=True)

        # HTML flame graph — open in browser
        html_path = out_dir / f"predict_{loan_id}.html"
        html_path.write_text(profiler.output_html())

        # Text summary — printed to stdout
        print(f"\n── PyInstrument: /predict/{loan_id} ──────────────")
        print(profiler.output_text(unicode=True, color=True, show_all=False))

        return response