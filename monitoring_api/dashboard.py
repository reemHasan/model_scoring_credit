"""
dashboard.py
─────────────────────────────────────────────────────────────────────────────
Production monitoring dashboard for the Home Credit Scoring API.
Shows: KPIs, latency charts, error distribution, score drift, Evidently reports.

Run:
    streamlit run dashboard.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import tempfile
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from datetime import timedelta
# ── Evidently imports ─────────────────────────────────────────────────────────
from evidently import Report
from evidently.presets  import DataDriftPreset, DataSummaryPreset
# ── DB loaders ────────────────────────────────────────────────────────────────
from db_helpers import load_logs_df, load_features_wide_df

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Monitoring LOAN SCORING SYSTEM",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .dashboard-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2rem;
        font-weight: 600;
        letter-spacing: -1px;
        color: #F0F4F8;
        padding-left: 16px;
        margin-bottom: 4px;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .dashboard-subtitle {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        color: #64748B;
        letter-spacing: 3px;
        text-transform: uppercase;
        padding-left: 20px;
        margin-bottom: 32px;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .kpi-card {
        background: #0F1117;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 20px 24px;
        text-align: center;
    }
    .kpi-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #475569;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2rem;
        font-weight: 600;
        color: #F0F4F8;
    }
    .kpi-value.alert { color: #EF4444; }
    .kpi-value.ok    { color: #10B981; }
    .kpi-value.warn  { color: #F59E0B; }
    .section-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.1rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #ff850b;
        margin: 32px 0 16px 0;
        border-bottom: 1px solid #1E293B;
        padding-bottom: 8px;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    div[data-testid="stMetric"] {
        background: #0F1117;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 16px;
    }
    .stRadio > div { gap: 12px; }
</style>
""", unsafe_allow_html=True)
# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    # refresh_sec = st.slider("Auto-refresh (sec)", 10, 300, 60)
    refresh_hours = st.sidebar.slider("Auto-refresh (hours)", 1, 24, 3)
    lookback    = st.selectbox("Lookback window", ["Last 24h", "Last 7 days", "Last 30 days", "All time"], index=1)
    st.divider()
    # to be used after when we have more data in the DB and want to adjust the reference window for drift analysis
    st.markdown("### 📊 Reference data ")
    ref_split = st.slider("Reference split (%)", 10, 50, 30,
                          help="% of earliest data used as Evidently reference")
    st.divider()
    st.caption("🔄 Auto-refreshes every {} hours".format(refresh_hours))
# Auto-refresh
# st_autorefresh(interval=refresh_sec * 1000, key="dashboard_refresh")
st_autorefresh(interval=refresh_hours * 3600 * 1000, key="dashboard_refresh")
# ── Load data functions ──────────────────────────────────────────────────────────
#BASE_DIR = Path(__file__).parent.parent
#@st.cache_data
#def get_ref__data():
#    reference_data = pd.read_parquet(BASE_DIR / "data/training_data/test_data_bestFeatures.parquet")
#    return reference_data

@st.cache_data(ttl="1d")
def get_log_data():
    return load_logs_df(), load_features_wide_df()

@st.cache_data(ttl="1d", show_spinner="Generating drift report...")
def build_drift_report(ref_data, cur_data,metric_name):
    if metric_name == "drift":
        metric = DataDriftPreset()
    elif metric_name == "summary":
        metric = DataSummaryPreset()
    report = Report(metrics=[metric])
    report_result = report.run(reference_data=ref_data, current_data=cur_data)
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        report_result.save_html(f.name)
        return Path(f.name)
# ==================================================================================================
# ── Load data ─────────────────────────────────────────────────────────────────────
# Get reference data for Evidently reports

#reference_data = get_ref__data()
#print("Reference data info:", reference_data.shape[0], "rows,", reference_data.shape[1], "columns")
#print("Reference data columns:", reference_data.columns)"""
# Get Api data from DB
logs, features_wide = get_log_data()
features_wide["Target"] = (features_wide["proba_class"] == "default").astype(int)
logs["Target"] = (logs["proba_class"] == "default").astype(int)
#print("Logs info:", logs.shape[0], "rows,", logs.shape[1], "columns")
print("Client api info:", features_wide.shape[0], "rows,", features_wide.shape[1], "columns")
# print("Client api data columns:", features_wide.columns)
# ===========================================================================================
# Apply lookback filter
#==========================================================================================
if not logs.empty:
    now = pd.Timestamp.now(tz="UTC")
    windows = {
        "Last 24h":    timedelta(hours=24),
        "Last 7 days": timedelta(days=7),
        "Last 30 days":timedelta(days=30),
        "All time":    None,
    }
    w = windows[lookback]
    if w:
        logs = logs[logs["timestamp"] >= now - w]
        if not features_wide.empty:
            features_wide = features_wide[features_wide["timestamp"] >= now - w]

# ── TITLE ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="dashboard-title">🏦 Monitoring LOAN SCORING SYSTEM</div>', unsafe_allow_html=True)
st.markdown('<div class="dashboard-subtitle">Home Credit Risk · Production API · Real-time monitoring</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="dashboard-subtitle">
        <p>
            <a href="https://reemmathbout-loan-scoring-app.hf.space" target="_blank">Production API</a>
            <a href="https://github.com/reemHasan/model_scoring_credit" target="_blank">GitHub</a> 
        </p></div>""",unsafe_allow_html=True)

if logs.empty:
    st.warning("No logs found in the selected window. Make some predictions first.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# KPI ROW
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Key Metrics</div>', unsafe_allow_html=True)

total_calls   = len(logs)
error_rate    = (logs["status_code"] >= 400).mean()
max_latency = logs["total_ms"].max()
avg_latency   = logs["total_ms"].mean()
avg_inference = logs["inference_ms"].mean()
rejected_rate = (logs["proba_class"] == "default").mean() if "proba_class" in logs.columns else 0.0

c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:
    st.metric("Total Calls", f"{total_calls:,}",
              delta="API High Demand" if total_calls > 1000 else "API normal Demand",
              delta_color="inverse" if total_calls > 1000 else "normal")
with c2:
    delta_color = "inverse" if error_rate > 0.05 else "normal"
    st.metric("Error Rate", f"{error_rate:.1%}",
              delta="above threshold" if error_rate > 0.05 else "normal",
              delta_color=delta_color)
with c3:
    st.metric("Max Latency", f"{max_latency:.1f} ms",
              delta="high" if max_latency > 3000 else "normal",
              delta_color="inverse" if max_latency > 3000 else "normal")
with c4:
    st.metric("Avg Latency", f"{avg_latency:.1f} ms",
              delta="high" if avg_latency > 500 else "normal",
              delta_color="inverse" if avg_latency > 500 else "normal")
with c5:
    st.metric("Avg Inference", f"{avg_inference:.1f} ms",
              delta="high" if avg_inference > 300 else "normal",
              delta_color="inverse" if avg_inference > 300 else "normal")
with c6:
    st.metric("Rejection Rate", f"{rejected_rate:.1%}",
              delta="high" if rejected_rate > 0.4 else "normal",
              delta_color="inverse" if rejected_rate > 0.4 else "normal")

# ══════════════════════════════════════════════════════════════════════════════
# LATENCY CHARTS — side by side
# ══════════════════════════════════════════════════════════════════════════════
#st.markdown('<div class="section-header">Latency Over Time</div>', unsafe_allow_html=True)
col_lat, col_inf = st.columns(2)
with col_lat:
    #st.subheader('<div class="section-header">API Latency Over Time</div>')
    st.markdown('<div class="section-header">API Latency Over Time</div>', unsafe_allow_html=True)
    fig_lat = px.line(
        logs, x="timestamp", y="total_ms",
        #title="API Latency (total_ms)",
        labels={"total_ms": "ms", "timestamp": ""},
        template="plotly_dark",
        color_discrete_sequence=["#3B82F6"],)
    # anomaly threshold line
    threshold = logs["total_ms"].mean() + 2 * logs["total_ms"].std()
    fig_lat.add_hline(
        y=threshold, line_dash="dash", line_color="#EF4444",
        annotation_text=f"threshold {threshold:.0f}ms",
        annotation_position="top right",
    )
    fig_lat.update_layout(
        plot_bgcolor="#0F1117", paper_bgcolor="#0F1117",
        font_color="#94A3B8", height=300,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig_lat, width='content')

with col_inf:
    st.markdown('<div class="section-header">Model Inference Time Distribution</div>', unsafe_allow_html=True)
    fig_inf = px.line(
        logs, x="timestamp", y="inference_ms",
        #title="Model Inference Time (inference_ms)",
        labels={"inference_ms": "ms", "timestamp": ""},
        template="plotly_dark",
        color_discrete_sequence=["#10B981"],
    )
    fig_inf.update_layout(
        plot_bgcolor="#0F1117", paper_bgcolor="#0F1117",
        font_color="#94A3B8", height=300,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig_inf, width='content')

# ══════════════════════════════════════════════════════════════════════════════
# STATUS CODE DISTRIBUTION — only if errors exist
# ══════════════════════════════════════════════════════════════════════════════
if error_rate > 0:
    st.markdown('<div class="section-header">Error Distribution</div>', unsafe_allow_html=True)
    error_logs = logs.copy()
    error_logs["hour"] = error_logs["timestamp"].dt.floor("h")
    status_over_time = (
        error_logs.groupby(["hour", "status_code"])
        .size()
        .reset_index(name="count")
    )
    status_over_time["status_code"] = status_over_time["status_code"].astype(str)
    fig_err = px.bar(
        status_over_time, x="hour", y="count", color="status_code",
        #title=f"Status Code Distribution Over Time  (error rate: {error_rate:.1%})",
        labels={"hour": "", "count": "Requests", "status_code": "HTTP Status"},
        template="plotly_dark",
        color_discrete_map={"200": "#10B981", "400": "#F59E0B", "404": "#F97316", "500": "#EF4444"},
    )
    fig_err.update_layout(
        plot_bgcolor="#0F1117", paper_bgcolor="#0F1117",
        font_color="#94A3B8", height=320,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig_err, width='content')
# ══════════════════════════════════════════════════════════════════════════════
# SCORE DISTRIBUTION + SCORE DRIFT — side by side
# ══════════════════════════════════════════════════════════════════════════════
#st.markdown('<div class="section-header">Score Distribution & Drift</div>', unsafe_allow_html=True)
col_score, col_drift = st.columns(2)
score_logs = logs.dropna(subset=["proba_default"])
with col_score:
    st.markdown('<div class="section-header">Avg Predicted Score per Class Over Time</div>', unsafe_allow_html=True)
    if not score_logs.empty and "proba_class" in score_logs.columns:
        # Resample by hour — mean proba per class per hour
        score_time = score_logs.copy()
        score_time["hour"] = score_time["timestamp"].dt.floor("h")
        score_over_time = (
            score_time.groupby(["hour", "proba_class"])["proba_default"]
            .mean()
            .reset_index()
        )
        score_over_time["proba_class_label"] = score_over_time["proba_class"].map({
            "default":    "Reject Loan",
            "no default": "Approve Loan",
        })
        fig_score = px.line(
            score_over_time,
            x="hour", y="proba_default",
            color="proba_class_label",
            #title="Avg Predicted Score per Class Over Time",
            labels={
                "hour":             "",
                "proba_default":    "Avg Default Probability",
                "proba_class_label": "",
            },
            template="plotly_dark",
            color_discrete_map={
                "Reject Loan":    "#EF4444",
                "Approve Loan": "#10B981",
            },
            markers=True,
        )
        fig_score.update_layout(
            plot_bgcolor="#0F1117", paper_bgcolor="#0F1117",
            font_color="#94A3B8", height=320,
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            yaxis=dict(range=[0, 1], tickformat=".0%"),
        )
        st.plotly_chart(fig_score, width="content")
    else:
        st.info("No score data available yet.")
with col_drift:
    if len(score_logs) >= 10:
        split_idx   = max(1, int(len(score_logs) * ref_split / 100))
        ref_data_api    = score_logs.iloc[:split_idx]
        cur_data_api    = score_logs.iloc[split_idx:]

        # Count class distribution for reference and current
        def class_counts(df):
            counts = df["proba_class"].value_counts(normalize=True).reset_index()
            counts.columns = ["class", "rate"]
            counts["class_label"] = counts["class"].map({
                "default":    "Class Default — Reject Loan",
                "no default": "Class No Default — Approve Loan",
            })
            return counts
        st.markdown('<div class="section-header">Predicted Class Drift Reference vs Current</div>', unsafe_allow_html=True)
        ref_counts = class_counts(ref_data_api)
        cur_counts = class_counts(cur_data_api)

        ref_counts["window"] = f"Reference (first {ref_split}%)"
        cur_counts["window"] = "Current"

        combined = pd.concat([ref_counts, cur_counts], ignore_index=True)

        fig_drift = px.bar(
            combined,
            x="class_label", y="rate",
            color="window", barmode="group",
            #title="Predicted Class Distribution — Reference vs Current",
            labels={"rate": "Rate", "class_label": "Class", "window": "Window"},
            template="plotly_dark",
            color_discrete_map={
                f"Reference (first {ref_split}%)": "#3B82F6",
                "Current":                         "#F59E0B",
            },
            text=combined["rate"].apply(lambda x: f"{x:.1%}"),
        )

        # Detect drift — flag if class rate shifts more than 10%
        ref_default = ref_data_api["proba_class"].eq("default").mean()
        cur_default = cur_data_api["proba_class"].eq("default").mean()
        shift       = abs(cur_default - ref_default)

        if shift > 0.10:
            fig_drift.add_annotation(
                text=f"⚠️ Class drift detected (+{shift:.1%})",
                xref="paper", yref="paper",
                x=0.98, y=0.95, showarrow=False,
                font=dict(color="#EF4444", size=12),
            )

        fig_drift.update_traces(textposition="outside")
        fig_drift.update_layout(
            plot_bgcolor="#0F1117", paper_bgcolor="#0F1117",
            font_color="#94A3B8", height=320,
            margin=dict(l=0, r=0, t=40, b=0),
            yaxis_tickformat=".0%",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_drift, width='content')
    else:
        st.info("Need at least 10 predictions to show class drift.")

# ══════════════════════════════════════════════════════════════════════════════
# EVIDENTLY REPORTS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header"> 📊 Evidently AI Reports</div>', unsafe_allow_html=True)

feature_cols = [c for c in features_wide.columns
                if c not in ("request_id", "timestamp", "loan_id", "proba_default","proba_class","Target")]

if features_wide.empty or len(feature_cols) == 0:
    st.info("No feature data available yet for drift analysis.")
else:
    #ref_df    = reference_data[feature_cols]
    #cur_df    = features_wide[feature_cols]
    # use reference datafrom api logs instead of training data for more relevant drift analysis, and split it into reference vs current based on the slider
    split_idx = max(1, int(len(features_wide) * ref_split / 100))
    ref_df    = features_wide.iloc[:split_idx][feature_cols]
    cur_df    = features_wide.iloc[split_idx:][feature_cols]

    if len(ref_df) < 5 or len(cur_df) < 5:
        st.warning(f"Not enough data for Evidently analysis. Need at least 5 rows in each window. "
                   f"Currently: reference={len(ref_df)}, current={len(cur_df)}. "
                   f"Try adjusting the reference split slider.")
    else:
        # ── Data Drift report (always shown) ─────────────────────────────────
        #st.markdown("#### 📊 Data Drift Report")
        drift_html = build_drift_report(
            ref_data=ref_df,
            cur_data=cur_df,
            metric_name="drift",
        )
        with open(drift_html, encoding="utf-8") as f:
                    st.iframe(f.read(), height=800)

        # ── Radio button for additional report ───────────────────────────────
        st.markdown("#### 📋 Additional Reports")
        report_choice = st.radio(
            "Select report to display",
            options=["None", "Data Summary"],
            horizontal=True,
        )
        if report_choice == "Data Summary":
            summary_html = build_drift_report(
                ref_data=ref_df,
                cur_data=cur_df,
                metric_name="summary",
            )
            with open(summary_html, encoding="utf-8") as f:
                    st.iframe(f.read(), height=800)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    f"Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} · "
    f"Showing {len(logs):,} requests · "
    f"Auto-refresh every {refresh_hours}h"
)