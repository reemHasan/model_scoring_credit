import gradio as gr
import requests
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap

# CONFIG
# Deploy in local
API_BASE = "http://127.0.0.1:8000"
# change if deployed on HF Spaces   
# API_BASE = "https://home-credit-scoring.hf.space"

# Load custom CSS for styling
def load_css():
    with open('./css/style.css', 'r') as file:
        css_content = file.read()
    return css_content
# SHAP WATERFALL PLOT
def plot_waterfall(shap_values_dict: dict, expected_value: float, proba: float) -> plt.Figure:
    """Build a clean SHAP waterfall chart from the API response."""
    features  = list(shap_values_dict.keys())
    sv        = np.array([shap_values_dict[f] for f in features], dtype=float)
    # Sort by absolute contribution
    order     = np.argsort(np.abs(sv))
    features  = [features[i] for i in order]
    sv        = sv[order]
    # Keep top 15 for readability
    features  = features[-15:]
    sv        = sv[-15:]
    cumulative = expected_value + sv.cumsum()
    starts     = np.concatenate([[expected_value], cumulative[:-1]])
    colors = ["#E53E3E" if v > 0 else "#38A169" for v in sv]
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#0F1117")
    for i, (feat, val, start, color) in enumerate(zip(features, sv, starts, colors)):
        ax.barh(i, val, left=start, color=color, height=0.55,
                edgecolor="#0F1117", linewidth=0.5)
        sign = "+" if val > 0 else ""
        ax.text(start + val + (0.002 if val > 0 else -0.002),
                i, f"{sign}{val:.3f}",
                va="center", ha="left" if val > 0 else "right",
                fontsize=7.5, color="white", fontweight="bold")
    # Expected value line
    ax.axvline(expected_value, color="#718096", linewidth=1.2,
               linestyle="--", label=f"E[f(x)] = {expected_value:.3f}")
    # Final prediction line
    ax.axvline(proba, color="#F6E05E", linewidth=1.8,
               linestyle="-", label=f"f(x) = {proba:.3f}")
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=8.5, color="#E2E8F0")
    ax.set_xlabel("SHAP value (impact on default probability)", color="#A0AEC0", fontsize=9)
    ax.tick_params(axis="x", colors="#A0AEC0")
    ax.spines[:].set_color("#2D3748")
    red_patch   = mpatches.Patch(color="#E53E3E", label="Increases default risk")
    green_patch = mpatches.Patch(color="#38A169", label="Decreases default risk")
    ax.legend(handles=[red_patch, green_patch,
                        mpatches.Patch(color="#718096", label=f"Baseline E[f(x)]={expected_value:.3f}"),
                        mpatches.Patch(color="#F6E05E", label=f"Prediction f(x)={proba:.3f}")],
              loc="lower right", fontsize=7.5,
              facecolor="#1A202C", edgecolor="#2D3748", labelcolor="white")
    ax.set_title("SHAP Waterfall — Feature Contributions", color="white",
                 fontsize=11, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig


def plot_shap_waterfall(shap_values_dict: dict, client_data: dict, expected_value: float) -> plt.Figure:
    # Create fresh figure
    plt.close("all")
    fig = plt.figure(figsize=(6, 6))
    features  = list(shap_values_dict.keys())
    shap_values = np.array([shap_values_dict[f] for f in features])
    client_values = np.array([client_data[f] for f in features])

    explanation = shap.Explanation(
        values=shap_values,
        base_values=expected_value,
        data=client_values,
        feature_names=features,
    )
    shap.plots.waterfall(explanation, show=False)
    plt.tight_layout()
    return fig

# Create empty plot 
def empty_plot():
    fig, ax = plt.subplots()
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.text(0.5, 0.5, "Waiting for SHAP output...",
            ha="center", va="center", fontsize=14, color="white", fontstyle="italic")
    ax.axis("off")
    return fig

# MAIN PREDICT FUNCTION
def predict_client(loan_id: int):
    try:
        resp = requests.post(f"{API_BASE}/predict/{loan_id}", timeout=15)
    except requests.exceptions.ConnectionError:
        return (
            "<div style='color:#FC8181;font-size:25px'>❌ Cannot reach the API. "
            "Make sure it is running at <code>" + API_BASE + "</code></div>",
            None, ""
        )
    if resp.status_code == 404:
        return (
            "<div style='color:#FC8181;font-size:30px'>❌ "
            + resp.json().get("detail", "Client not found") + "</div>",
            None, ""
        )
    if resp.status_code != 200:
        return (
            f"<div style='color:#FC8181;font-size:30px'>❌ API error {resp.status_code}</div>",
            None, ""
        )

    data     = resp.json()
    proba    = data["Client default probability"]
    decision = data["Decision"]
    cls      = data["Class"]

    # ── Decision badge
    if cls == "default":
        badge_color = "#2D1515"   # red
        icon        = "🚫"
        bg          = "#F1CDCD"
        border      = "#E53E3E"
    else:
        badge_color = "#152D1E"   # green
        icon        = "✅"
        bg          = "#A0DDB7"
        border      = "#38A169"

    decision_html = f"""
    <div style="
        background:{bg};
        border:2px solid {border};
        border-radius:12px;
        padding:20px 28px;
        font-family:'Courier New',monospace;
        margin-bottom:8px;
        text-align: center
    ">
        <div style="font-size:25px;font-weight:900;color:{badge_color};letter-spacing:1px">
            {icon} {decision}
        </div>
        <div style="margin-top:10px;font-size:17px;color:#29333f;font-weight:500">
            Client ID &nbsp;<span style="color:#29333f;font-weight:700">{loan_id}</span>
            &nbsp;·&nbsp;
            Default probability &nbsp;<span style="color:{badge_color};font-weight:700">{proba:.1%}</span>
            &nbsp;·&nbsp;
            Class &nbsp;<span style="color:{badge_color};font-weight:700">{cls.upper()}</span>
        </div>
    </div>
    """

    # ── Client info table 
    client_info = json.loads(data["Client_info"])[0]
    rows = "".join(
        f"<tr><td style='color:white;padding:4px 12px 4px 0'>{k}</td>"
        f"<td style='color:white;font-weight:600'>{round(v,4) if isinstance(v,float) else v}</td></tr>"
        for k, v in client_info.items()
    )
    info_html = f"""
    <div style='text-align:center;color:#ffffff;font-size:20px;margin-bottom:3px'>CLIENT INFORMATION </div>
            <div style="text-align:center;width:300px;height:1px;background:#3182CE;margin:3px auto 0"></div>
    <table style="font-family:'Courier New',monospace;font-size:15px;border-collapse:collapse">
        {rows}
    </table>
    """

    # ── SHAP waterfall
    shap_dict     = json.loads(data["Shap_values_client"])[0]
    # expected_value is returned by API — if not, default to 0.5
    expected_val  = data.get("Expected_Shap_Value")
    fig = plot_waterfall(shap_dict, expected_val, proba)
    #fig = plot_shap_waterfall(shap_dict, client_info, expected_val)    

    return decision_html,fig, info_html

# ── GRADIO UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(css=load_css(), title="Home Credit Scoring") as demo:

    # ── Header ─────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div style="
        text-align:center;
        padding:32px 0 16px;
        font-family:'Courier New',monospace;
    ">
        <div style="font-size:18px;font-weight:700;letter-spacing:6px;color:#ffffff;margin-bottom:6px">
            HOME CREDIT RISK GROUP
        </div>
        <div style="font-size:35px;font-weight:900;color:#E2E8F0;letter-spacing:2px">
            LOAN SCORING SYSTEM
        </div>
        <div style="font-size:15px;color:#f7f8f9;margin-top:8px">
            LightGBM · SHAP Explanations · Real-time Risk Assessment
        </div>
        <div style="width:60px;height:3px;background:#3182CE;margin:16px auto 0"></div>
    </div>
    """)

    # ── Input row ──────────────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=2):
            loan_id_input = gr.Slider(
                minimum=1, maximum=48744, step=1, value=1,
                label="Client ID  (1 – 48744)",
                info="Drag or type a client application ID"
            )
        with gr.Column(scale=1):
            predict_btn = gr.Button("▶  RUN PREDICTION", variant="primary", size="lg")

    # ── Outputs ────────────────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=0, min_width=800):
            decision_out = gr.HTML(label="Decision")
    with gr.Row():
        with gr.Column(scale=2):
            shap_plot = gr.Plot(value=empty_plot, label="SHAP Waterfall — Feature Contributions")
        with gr.Column(scale=1):
            client_info_out = gr.HTML(label="Information about the current client application")

    # ── Footer ─────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div style="text-align:center;padding:20px 0 8px;
                font-size:12px;color:#ffffff;font-family:'Courier New',monospace">
        LightGBM · FastAPI · Gradio &nbsp;|&nbsp; For educational use
    </div>
    """)

    # ── Wiring ─────────────────────────────────────────────────────────────────
    predict_btn.click(
        fn=predict_client,
        inputs=[loan_id_input],  # loan_id_input-1 to convert from 1-based to 0-based ID
        outputs=[decision_out, shap_plot, client_info_out],
    )

"""
    # Also trigger on slider release (optional)
    loan_id_input.release(
        fn=predict_client,
        inputs=[loan_id_input-1],  # loan_id_input-1 to convert from 1-based to 0-based ID
        outputs=[decision_out, results_row, shap_plot, client_info_out],
    )
"""

if __name__ == "__main__":
    demo.launch(server_port=7860)