# scripts/run_profiling.py
import subprocess
import requests
import time
import sys

# Step 1 — start the API in background
print("Starting API with profiling enabled...")
process = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "app.api:app", "--port", "8000", "--workers", "1",# ← single worker, sequential requests
        "--loop", "asyncio",],
    env={**__import__("os").environ, "ENABLE_PROFILING": "true"},
)

# Wait for API to be ready
print("Waiting for API to start...")
for _ in range(30):
    try:
        requests.get("http://localhost:8000/health")
        print("API is ready.")
        break
    except Exception:
        time.sleep(1)

# Step 2 — make 20 predictions
print("Making 20 predictions...")
for i in range(1, 3):
    resp = requests.post(f"http://localhost:8000/predict/{i}")
    print(f"  predict/{i} → {resp.status_code}")
    time.sleep(0.1)  

# Step 3 — stop the API
print("Stopping API...")
process.terminate()
print("Done. Check logs/profiles/ for .prof files.")