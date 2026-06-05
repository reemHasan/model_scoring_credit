"""
analysis.py
─────────────────────────────────────────────────────────────────────────────
Automated analysis of production logs stored in PostgreSQL:
  1. Operational anomalies  — error rate, latency spikes
  2. Score distribution drift — predicted probability shift
  3. Feature data drift       — Evidently AI report per feature
  4. Prints a summary + saves drift_report.html
"""
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timezone, timedelta
from evidently import Report
from evidently.presets  import DataDriftPreset, DataSummaryPreset
# from evidently.metrics import DatasetDriftMetric,DatasetMissingValuesMetric, ColumnDriftMetric
from db_helpers import load_logs_df, load_features_wide_df
import warnings
warnings.filterwarnings("ignore")

# ── Tunable thresholds ────────────────────────────────────────────────────────
ERROR_RATE_THRESHOLD    = 0.05    # alert if > 5 % errors
LATENCY_Z_THRESHOLD     = 2.5    # alert if latency > mean + 2.5σ
SCORE_DRIFT_THRESHOLD   = 0.10   # alert if mean proba shifts > 10 pp
REFERENCE_DAYS          = 7      # first N days = reference window
CURRENT_DAYS            = 1      # last N days  = current window

# ═════════════════════════════════════════════════════════════════════════════
# 2. OPERATIONAL ANOMALY DETECTION
# ═════════════════════════════════════════════════════════════════════════════
def analyse_operations(logs: pd.DataFrame) -> dict:
    print("\n" + "═"*60)
    print("  OPERATIONAL ANALYSIS")
    print("═"*60)

    results = {}
    n = len(logs)
    if n == 0:
        print("  No logs found.")
        return results

    print(f"  Total calls logged : {n}")

    # ── Error rate ────────────────────────────────────────────────────────────
    errors     = logs[logs["status_code"] >= 400]
    error_rate = len(errors) / n
    results["error_rate"] = error_rate
    flag = "⚠️  ALERT" if error_rate > ERROR_RATE_THRESHOLD else "✅ OK"
    print(f"  Error rate         : {error_rate:.2%}   {flag}")
    if len(errors):
        print("Most common errors:")
        for code, cnt in errors["status_code"].value_counts().head(3).items():
            print(f"      HTTP {code} : {cnt} calls")

    # ── Latency ───────────────────────────────────────────────────────────────
    lat       = logs["total_ms"].dropna()
    mean_lat  = lat.mean()
    std_lat   = lat.std()
    p95_lat   = lat.quantile(0.95)
    threshold = mean_lat + LATENCY_Z_THRESHOLD * std_lat
    anomalous = logs[logs["total_ms"] > threshold]

    results.update({"mean_latency_ms": mean_lat, "p95_latency_ms": p95_lat,
                    "latency_anomalies": len(anomalous)})

    print("\n  Latency (total_ms)")
    print(f"    Mean             : {mean_lat:.1f} ms")
    print(f"    P95              : {p95_lat:.1f} ms")
    print(f"    Anomaly threshold: {threshold:.1f} ms  (mean + {LATENCY_Z_THRESHOLD}σ)")
    print(f"    Anomalous calls  : {len(anomalous)}  ", end="")
    print("⚠️  ALERT" if len(anomalous) > 0 else "✅ OK")

    # ── Inference time ────────────────────────────────────────────────────────
    inf = logs["inference_ms"].dropna()
    print("\n  Inference time (model only)")
    print(f"    Mean             : {inf.mean():.1f} ms")
    print(f"    P95              : {inf.quantile(0.95):.1f} ms")

    return results


# ═════════════════════════════════════════════════════════════════════════════
# 3. SCORE DISTRIBUTION DRIFT
# ═════════════════════════════════════════════════════════════════════════════
def analyse_score_drift(logs: pd.DataFrame, artifact_dir: Path) -> dict:
    print("\n" + "═"*60)
    print("  SCORE DISTRIBUTION DRIFT")
    print("═"*60)

    results = {}
    scores  = logs["proba_default"].dropna()
    if len(scores) < 10:
        print("  Not enough data yet.")
        return results

    now      = logs["timestamp"].max()
    ref_mask = logs["timestamp"] <= now - timedelta(days=CURRENT_DAYS)
    cur_mask = logs["timestamp"] >  now - timedelta(days=CURRENT_DAYS)

    ref_scores = logs.loc[ref_mask, "proba_default"].dropna()
    cur_scores = logs.loc[cur_mask, "proba_default"].dropna()

    if len(ref_scores) == 0 or len(cur_scores) == 0:
        print("  Insufficient data in one window — using full dataset stats.")
        ref_scores = scores[:len(scores)//2]
        cur_scores = scores[len(scores)//2:]

    mean_shift = abs(cur_scores.mean() - ref_scores.mean())
    results["score_mean_ref"]     = ref_scores.mean()
    results["score_mean_current"] = cur_scores.mean()
    results["score_mean_shift"]   = mean_shift

    print(f"  Reference mean score  : {ref_scores.mean():.4f}")
    print(f"  Current mean score    : {cur_scores.mean():.4f}")
    print(f"  Absolute mean shift   : {mean_shift:.4f}  ", end="")
    print("⚠️  ALERT" if mean_shift > SCORE_DRIFT_THRESHOLD else "✅ OK")

    default_rate_ref = (logs.loc[ref_mask, "proba_class"] == "default").mean()
    default_rate_cur = (logs.loc[cur_mask, "proba_class"] == "default").mean()
    print(f"  Default rate (ref)    : {default_rate_ref:.2%}")
    print(f"  Default rate (current): {default_rate_cur:.2%}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(ref_scores, bins=30, alpha=0.6, label=f"Reference (n={len(ref_scores)})", color="#4299E1")
    ax.hist(cur_scores, bins=30, alpha=0.6, label=f"Current (n={len(cur_scores)})",   color="#FC8181")
    ax.axvline(ref_scores.mean(), color="#2B6CB0", linestyle="--", linewidth=1.5)
    ax.axvline(cur_scores.mean(), color="#C53030", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Default probability")
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution: Reference vs Current")
    ax.legend()
    fig.tight_layout()
    fig.savefig(artifact_dir / "score_drift.png", dpi=130)
    plt.close(fig)
    print(f"  → Chart saved to {artifact_dir / 'score_drift.png'}")

    return results


# ═════════════════════════════════════════════════════════════════════════════
# 4. FEATURE DATA DRIFT  (Evidently AI)
# ═════════════════════════════════════════════════════════════════════════════
def analyse_feature_drift(features_wide: pd.DataFrame, artifact_dir: Path) -> None:
    print("\n" + "═"*60)
    print("  FEATURE DATA DRIFT  (Evidently AI)")
    print("═"*60)

    feature_cols = [c for c in features_wide.columns
                    if c not in ("request_id","timestamp","loan_id")]

    if len(feature_cols) == 0:
        print("  No feature columns found.")
        return

    now      = features_wide["timestamp"].max()
    ref_mask = features_wide["timestamp"] <= now - timedelta(days=CURRENT_DAYS)
    cur_mask = features_wide["timestamp"] >  now - timedelta(days=CURRENT_DAYS)

    ref_df = features_wide.loc[ref_mask, feature_cols]
    cur_df = features_wide.loc[cur_mask, feature_cols]

    if len(ref_df) < 5 or len(cur_df) < 5:
        print("  Splitting dataset in half for reference/current windows.")
        mid    = len(features_wide) // 2
        ref_df = features_wide.iloc[:mid][feature_cols]
        cur_df = features_wide.iloc[mid:][feature_cols]

    print(f"  Reference rows : {len(ref_df)}")
    print(f"  Current rows   : {len(cur_df)}")

    # Build Evidently report
    report = Report([
        DataDriftPreset(),
        DataSummaryPreset()
    ])

    report_run = report.run(reference_data=ref_df, current_data=cur_df)

    out_path = artifact_dir/"drift_report.html"
    report_run.save_html(str(out_path))
    print(f"  ✅ Evidently report saved → {out_path}")
    #print(report_run.json())
    """
    # Print quick per-feature drift summary
    report_dict = report_run.json()
    try:
        drift_results = report_dict["metrics"][0]["result"]["drift_by_columns"]
        print("\n  Per-feature drift detected:")
        for feat, info in drift_results.items():
            drifted = info.get("drift_detected", False)
            score   = info.get("drift_score", None)
            icon    = "⚠️ " if drifted else "  "
            score_s = f"{score:.4f}" if score is not None else "n/a"
            print(f"  {icon} {feat:<35} drift_score={score_s}  drifted={drifted}")
    except (KeyError, IndexError):
        print("  (Could not parse per-feature summary from report dict)")
    """

# ═════════════════════════════════════════════════════════════════════════════
# 5. LATENCY OVER TIME CHART
# ═════════════════════════════════════════════════════════════════════════════
def plot_latency_over_time(logs: pd.DataFrame, artifact_dir: Path) -> None:
    if len(logs) < 2:
        return
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(logs["timestamp"], logs["total_ms"], linewidth=0.8,
            alpha=0.7, color="#4299E1", label="total_ms")
    ax.plot(logs["timestamp"], logs["inference_ms"], linewidth=0.8,
            alpha=0.7, color="#68D391", label="inference_ms")
    threshold = logs["total_ms"].mean() + LATENCY_Z_THRESHOLD * logs["total_ms"].std()
    ax.axhline(threshold, color="#FC8181", linestyle="--",
               linewidth=1, label=f"anomaly threshold ({threshold:.0f} ms)")
    ax.set_ylabel("ms")
    ax.set_title("API Latency Over Time")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(artifact_dir / "latency_over_time.png", dpi=130)
    plt.close(fig)
    print(f"  → Latency chart saved to {artifact_dir / 'latency_over_time.png'}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # Create artifacts folder 
    BASE_DIR = Path(__file__).resolve().parent
    artifact_dir = BASE_DIR / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    print("\n🔍  Home Credit API — Production Log Analysis")
    print(f"    Run at: {datetime.now(timezone.utc).isoformat()}\n")
    # load logs and features from PostgreSQL (via SQLAlchemy) and run analyses
    logs          = load_logs_df()
    features_wide = load_features_wide_df()
    op_results    = analyse_operations(logs)
    score_results = analyse_score_drift(logs, artifact_dir)
    plot_latency_over_time(logs, artifact_dir)
    analyse_feature_drift(features_wide, artifact_dir)

    print("\n" + "═"*60)
    print("  DONE — check logs/ folder for charts and drift_report.html")
    print("═"*60 + "\n")
