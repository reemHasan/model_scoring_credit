# src/benchmark.py
import time
import joblib
import numpy as np
import pandas as pd
import onnxruntime as rt
from pathlib import Path

BASE_DIR   = Path(__file__).parent.parent
BASE_DIR_DATA   = Path(__file__).parent.parent.parent
print(BASE_DIR_DATA)
N_RUNS     = 1000
LOAN_IDS   = list(range(1, N_RUNS + 1))

# Load artifacts
bundle      = joblib.load(BASE_DIR / "model/lgbm_bestmodel_fbeta10_bundle.pkl")
model       = bundle["model"]
features    = bundle["feature_names"]
client_data = pd.read_parquet(
    BASE_DIR_DATA / "data/prod_data/new_test_data_20features.parquet")[features]

# ── Benchmark 1: original LightGBM ───────────────────────────────────────────
print("Benchmarking original LightGBM...")
# LightGBM warmup
model.predict_proba(client_data.iloc[[0]])

lgbm_times = []
for lid in LOAN_IDS:
    X = client_data.iloc[[lid - 1]]
    t = time.perf_counter()
    model.predict_proba(X)
    lgbm_times.append((time.perf_counter() - t) * 1000)

# ── Benchmark 2: ONNX float32 ────────────────────────────────────────────────
print("Benchmarking ONNX float32...")
sess_f32   = rt.InferenceSession(
    str(BASE_DIR/"model/lgbm_model.onnx"),
    providers=["CPUExecutionProvider"],
)
in_name    = sess_f32.get_inputs()[0].name
out_name   = sess_f32.get_outputs()[1].name

# ONNX warmup
X0 = client_data.iloc[[0]].values.astype(np.float32)
sess_f32.run([out_name], {in_name: X0})

onnx_times = []
for lid in LOAN_IDS:
    X = client_data.iloc[[lid - 1]].values.astype(np.float32)
    t = time.perf_counter()
    sess_f32.run([out_name], {in_name: X})
    onnx_times.append((time.perf_counter() - t) * 1000)

# ── Benchmark 3: ONNX quantized INT8 ─────────────────────────────────────────
print("Benchmarking ONNX quantized INT8...")
sess_int8  = rt.InferenceSession(
    str(BASE_DIR/"model/lgbm_model_quantized.onnx"),
    providers=["CPUExecutionProvider"],
)
# ONNX int warmup
sess_int8.run([out_name], {in_name: X0})
quant_times = []
for lid in LOAN_IDS:
    X = client_data.iloc[[lid - 1]].values.astype(np.float32)
    t = time.perf_counter()
    sess_int8.run([out_name], {in_name: X})
    quant_times.append((time.perf_counter() - t) * 1000)

# ── Results ───────────────────────────────────────────────────────────────────
results = pd.DataFrame({
    "version":   ["LightGBM (original)", "ONNX float32", "ONNX INT8 quantized"],
    "mean_ms":   [np.mean(lgbm_times),   np.mean(onnx_times),  np.mean(quant_times)],
    "p50_ms":    [np.median(lgbm_times), np.median(onnx_times), np.median(quant_times)],
    "p95_ms":    [np.percentile(lgbm_times, 95), np.percentile(onnx_times, 95), np.percentile(quant_times, 95)],
    "p99_ms":    [np.percentile(lgbm_times, 99), np.percentile(onnx_times, 99), np.percentile(quant_times, 99)],
})

print("\n── Benchmark Results ──────────────────────────────")
print(results.to_string(index=False))
print(f"\nONNX speedup vs LightGBM   : {np.mean(lgbm_times)/np.mean(onnx_times):.2f}x")
print(f"INT8  speedup vs LightGBM  : {np.mean(lgbm_times)/np.mean(quant_times):.2f}x")
print(f"INT8  speedup vs ONNX f32  : {np.mean(onnx_times)/np.mean(quant_times):.2f}x")

results.to_csv(BASE_DIR/"artifacts/benchmark_results.csv", index=False)
print("\n✅ Results saved to artifacts/benchmark_results.csv")
# Verify if prediction of all model is the same
lgbm_pred = model.predict_proba(client_data.iloc[[0]])

onnx_pred = sess_f32.run(
    [out_name],
    {in_name: client_data.iloc[[0]].values.astype(np.float32)}
)[0]
onnx_int8_pred = sess_int8.run(
    [out_name],
    {in_name: client_data.iloc[[0]].values.astype(np.float32)}
)[0]

print(lgbm_pred)
print(onnx_pred)
print(onnx_int8_pred)