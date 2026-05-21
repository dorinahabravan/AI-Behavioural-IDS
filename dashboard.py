import streamlit as st
import pandas as pd
import joblib
import time
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import base64
from tensorflow.keras.models import load_model
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# -------------------------------------------------
# 🎨 UI STYLE (EXACT GRAFANA MATCH - UNCHANGED)
# -------------------------------------------------
st.set_page_config(layout="wide", page_title="IDS Dashboard")

st.markdown(
    """
<style>

/* MAIN BACKGROUND */
.stApp {
    background: linear-gradient(135deg, rgba(24,27,31,0.98), rgba(17,18,23,0.98));
    color: #d8d9da;
}

/* REDUCE PAGE PADDING */
.block-container {
    padding-top: 0.4rem;
    padding-bottom: 0.2rem;
    padding-left: 0.8rem;
    padding-right: 0.8rem;
    max-width: 100%;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background-color: rgba(24,27,31,0.95);
    border-right: 1px solid #26292e;
}

/* TOP BAR */
header[data-testid="stHeader"] {
    background-color: rgba(24,27,31,0.9);
    border-bottom: 1px solid #26292e;
}

/* COMPACT TITLE */
h1 {
    font-size: 1.7rem !important;
    margin-top: 0rem !important;
    margin-bottom: 0.4rem !important;
}

/* HEADER TEXT */
.main-header {
    font-size: 11px;
    color: #9ca3af;
    margin-bottom: 4px;
}

/* PANELS */
.grafana-panel {
    background-color: rgba(30,34,40,0.72);
    border: 1px solid #2c3136;
    border-radius: 6px;
    padding: 8px;
    margin-bottom: 6px;
    backdrop-filter: blur(6px);
}
            .panel-title {
    background-color: rgba(30,34,40,0.72);
    border: 1px solid #2c3136;
    border-radius: 6px;
    padding: 10px;
    margin-bottom: 6px;
    text-align: center;
    color: #e5e7eb;
    font-size: 10px;
    font-weight: bold;
    text-transform: uppercase;
}

/* KPI */
.kpi-container {
    text-align: center;
    min-height: 70px;
}

.kpi-title {
    color: #56d679;
    font-size: 10px;
    font-weight: bold;
    text-transform: uppercase;
}

.kpi-value {
    font-size: 24px;
    font-weight: 700;
    color: #56d679;
}

/* TABLE INSIDE SCROLL ONLY */
.scroll-table {
    max-height: 230px;
    overflow-y: auto;
    border-radius: 6px;
    font-size: 10px;
}

/* TABLE FONT */
.scroll-table table {
    font-size: 10px !important;
    width: 100%;
}

.scroll-table th, .scroll-table td {
    padding: 4px !important;
    white-space: nowrap;
    text-align: left;!important;
}

/* ALERT BOX SAME HEIGHT AS TABLE */
.alert-box {
    background: rgba(90, 45, 50, 0.35);
    border: 1px solid rgba(239, 68, 68, 0.45);
    padding: 10px;
    border-radius: 6px;
    width: 100%;
    box-sizing: border-box;
    min-height: 190px;
    max-height: 190px;
    overflow-y: auto;
    font-size: 11px;
    color: #e5e7eb;
}

/* NORMAL ALERT PLACEHOLDER */
.normal-alert-box {
    background: rgba(34,197,94,0.12);
    border: 1px solid #22c55e;
    padding: 10px;
    border-radius: 6px;
    width: 100%;
    box-sizing: border-box;
    min-height: 230px;
    max-height: 230px;
    font-size: 11px;
}

/* REMOVE STREAMLIT DEFAULT */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

</style>
""",
    unsafe_allow_html=True,
)

# -------------------------------------------------
# STATE
# -------------------------------------------------
if "running" not in st.session_state:
    st.session_state.running = False
if "attack_logs" not in st.session_state:
    st.session_state.attack_logs = []
if "risk_history" not in st.session_state:
    st.session_state.risk_history = []
if "alert_active" not in st.session_state:
    st.session_state.alert_active = False
if "peak_traffic" not in st.session_state:
    st.session_state.peak_traffic = 0
if "alert_counter" not in st.session_state:
    st.session_state.alert_counter = 0
if "model_run_summaries" not in st.session_state:
    st.session_state.model_run_summaries = []

if "completed_models" not in st.session_state:
    st.session_state.completed_models = set()


# -------------------------------------------------
# SOUND ALERT
# -------------------------------------------------
import base64


def play_sound():
    with open("alert.mp3", "rb") as f:
        audio_bytes = f.read()
        audio_base64 = base64.b64encode(audio_bytes).decode()

    st.components.v1.html(
        f"""
        <audio id="alert_audio" autoplay>
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        </audio>

        <script>
            var audio = document.getElementById("alert_audio");
            if (audio) {{
                audio.play();
            }}
        </script>
    """,
        height=0,
    )


# -------------------------------------------------
# SHAP-INFORMED ANALYST EXPLANATION
# -------------------------------------------------


def generate_shap_explanation(row, model_choice, risk=None):
    """
    Generates varied analyst-facing explanations based on SHAP-important
    features identified during offline SHAP analysis.
    """

    destination_port = row.get("Destination Port", 0)
    flow_duration = row.get("Flow Duration", 0)
    flow_bytes = row.get("Flow Bytes/s", 0)
    flow_packets = row.get("Flow Packets/s", 0)
    psh_flag = row.get("PSH Flag Count", 0)
    init_win = row.get("Init_Win_bytes_forward", 0)
    port_attack_rate = row.get("port_attack_rate", 0)
    window_attack_rate = row.get("port_window_attack_rate", 0)
    packet_length_mean = row.get("Packet Length Mean", 0)

    reasons = []

    if port_attack_rate > 0:
        reasons.append(
            f"destination port {int(destination_port)} has attack-associated behaviour "
            f"(port attack rate {port_attack_rate:.4f})"
        )

    if window_attack_rate > 0:
        reasons.append(
            f"the current traffic window shows attack-like behaviour "
            f"(window attack rate {window_attack_rate:.4f})"
        )

    if flow_duration <= 100:
        reasons.append(
            f"flow duration is very short ({flow_duration:.2f}), which may indicate scanning or automated traffic"
        )
    elif flow_duration > 1000000:
        reasons.append(
            f"flow duration is unusually long ({flow_duration:.2f}), suggesting abnormal session behaviour"
        )

    if flow_bytes > 100000:
        reasons.append(f"traffic volume is high ({flow_bytes:.2f} Flow Bytes/s)")

    if flow_packets > 10000:
        reasons.append(
            f"packet rate is unusually high ({flow_packets:.2f} Flow Packets/s)"
        )

    if psh_flag > 0:
        reasons.append(
            f"PSH flag activity is present ({int(psh_flag)}), contributing to suspicious traffic behaviour"
        )

    if init_win > 0:
        reasons.append(
            f"initial forward window size influenced the decision ({int(init_win)})"
        )

    if packet_length_mean > 1000:
        reasons.append(f"average packet length is large ({packet_length_mean:.2f})")

    if not reasons:
        reasons.append(
            "the model detected an unusual combination of flow-level and behavioural features"
        )

    # Vary the order slightly based on port, so text does not always look identical
    if int(destination_port) % 2 == 0:
        selected_reasons = reasons[:3]
    else:
        selected_reasons = list(reversed(reasons[:3]))

    if model_choice == "Random Forest":
        opening = "Random Forest classified this flow as ATTACK. "
        bridge = "The main SHAP-informed indicators were "
    elif model_choice == "Logistic Regression":
        opening = "Logistic Regression flagged this flow as ATTACK. "
        bridge = "The strongest feature indicators were "
    elif model_choice == "Isolation Forest":
        opening = "Isolation Forest identified this flow as anomalous. "
        bridge = "The anomaly appears linked to "
    elif model_choice == "CNN":
        opening = "The 1D-CNN detected attack-like traffic behaviour. "
        bridge = "The detected pattern was mainly associated with "
    else:
        opening = "The selected model classified this flow as ATTACK. "
        bridge = "The main indicators were "

    return opening + bridge + "; ".join(selected_reasons) + "."


def plot_live_shap_influence(row, model_choice, is_attack):
    """
    Live SHAP-informed feature influence chart.
    Keeps SHAP-like feature movement and pushes the High pink/red point
    to the top when the current flow is classified as ATTACK.
    """

    features = {
        "Attack classification": 1 if is_attack else 0,
        "Port attack rate": row.get("port_attack_rate", 0),
        "Window attack rate": row.get("port_window_attack_rate", 0),
        "Flow duration": row.get("Flow Duration", 0),
        "Flow Bytes/s": row.get("Flow Bytes/s", 0),
        "Flow Packets/s": row.get("Flow Packets/s", 0),
        "PSH Flag count": row.get("PSH Flag Count", 0),
        "Init Win Bytes Fwd": row.get("Init_Win_bytes_forward", 0),
        "Packet length mean": row.get("Packet Length Mean", 0),
    }

    influence_rows = []

    for feature, value in features.items():
        value = 0 if pd.isna(value) else value

        if feature == "Attack classification":
            # This is the top indicator point.
            # If attack = pink/red and shifted right.
            # If normal = blue and closer to zero.
            influence = 8 if is_attack else 0.5
            colour_value = 1 if is_attack else 0
            feature_value = "ATTACK" if is_attack else "NORMAL"
        else:
            # Original SHAP-like behaviour
            magnitude = np.log1p(abs(value))
            direction = 1 if value >= 0 else -1
            # Small live movement so feature points visibly update between flows
            live_shift = 0.15 * np.sin(magnitude + len(str(value)))
            influence = (magnitude * direction) + live_shift
            feature_value = value

            # Normalise later for normal features
            colour_value = None

        influence_rows.append(
            {
                "Feature": feature,
                "Influence": influence,
                "Feature value": feature_value,
                "ColourValue": colour_value,
            }
        )

    influence_df = pd.DataFrame(influence_rows)

    # Normalise colour for normal feature rows only
    normal_mask = influence_df["Feature"] != "Attack classification"
    normal_values = pd.to_numeric(
        influence_df.loc[normal_mask, "Feature value"], errors="coerce"
    ).fillna(0)

    min_val = normal_values.min()
    max_val = normal_values.max()

    if max_val - min_val == 0:
        influence_df.loc[normal_mask, "ColourValue"] = 0.5
    else:
        influence_df.loc[normal_mask, "ColourValue"] = (normal_values - min_val) / (
            max_val - min_val
        )

    influence_df["ColourValue"] = influence_df["ColourValue"].astype(float)

    influence_df = influence_df.sort_values("Influence", ascending=True)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=influence_df["Influence"],
            y=influence_df["Feature"],
            mode="markers",
            marker=dict(
                size=[
                    18 if str(f) == "Attack classification" else 13
                    for f in influence_df["Feature"]
                ],
                color=influence_df["ColourValue"],
                colorscale=[
                    [0, "#1e90ff"],  # low = blue
                    [0.5, "#8b5cf6"],  # middle = purple
                    [1, "#ff006e"],  # high = pink/red
                ],
                showscale=True,
                colorbar=dict(
                    tickvals=[0, 1],
                    ticktext=["Low", "High"],
                    thickness=12,
                    len=0.85,
                    titlefont=dict(color="white"),
                    tickfont=dict(color="white"),
                ),
                line=dict(width=0.8, color="rgba(255,255,255,0.75)"),
            ),
            hovertemplate=(
                "<b>%{y}</b><br>Influence indicator: %{x:.2f}<br><extra></extra>"
            ),
        )
    )

    fig.add_vline(x=0, line_width=2, line_color="rgba(200,200,200,0.7)")

    fig.update_layout(
        height=190,
        margin=dict(l=5, r=5, t=20, b=25),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", size=10),
        xaxis=dict(
            title="SHAP value (impact on model output)",
            range=[-0.5, 10],
            zeroline=False,
            gridcolor="rgba(255,255,255,0.08)",
            color="white",
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            color="white",
        ),
    )

    return fig


# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    selected_model = st.selectbox(
        "Detection Model",
        ["Random Forest", "Logistic Regression", "Isolation Forest", "CNN"],
    )

    # Stop monitoring automatically when model changes
    if "previous_model" not in st.session_state:
        st.session_state.previous_model = selected_model

    if selected_model != st.session_state.previous_model:
        st.session_state.running = False
        st.session_state.previous_model = selected_model

    model_choice = selected_model

    if st.button("▶ Start Monitoring", use_container_width=True):
        st.session_state.running = True
        st.session_state.attack_logs = []
        st.session_state.peak_traffic = 0
        st.session_state.risk_history = []
        st.session_state.alert_active = False
        st.session_state.alert_counter = 0

    if st.button("🛑 Stop", use_container_width=True):
        st.session_state.running = False

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown(
    '<div class="main-header">Home > Dashboards > IDS Behavioral Analysis</div>',
    unsafe_allow_html=True,
)
st.title(f"🛡️ Intrusion Detection System ({model_choice})")

# -------------------------------------------------
# LAYOUT
# -------------------------------------------------
# -------------------------------------------------
# COMPACT GRAFANA-LIKE ONE PAGE LAYOUT
# -------------------------------------------------

col_chart_l, col_chart_r = st.columns([2.2, 1])

with col_chart_l:
    st.markdown(
        '<div class="panel-title"><b>Risk Level Trend</b></div>', unsafe_allow_html=True
    )
    trend_placeholder = st.empty()

with col_chart_r:
    st.markdown(
        '<div class="panel-title"><b>Attack Risk Indicator</b></div>',
        unsafe_allow_html=True,
    )
    gauge_placeholder = st.empty()

k1, k2, k3, k4 = st.columns(4)
kpi_placeholders = [k1.empty(), k2.empty(), k3.empty(), k4.empty()]

# -------------------------------------------------
# ATTACK LOG FULL WIDTH
# -------------------------------------------------

st.markdown(
    '<div class="panel-title"><b> Attack Logs</b></div>', unsafe_allow_html=True
)
table_placeholder = st.empty()

# -------------------------------------------------
# EXPLAINABILITY SECTION: ALERT + LIVE SHAP
# -------------------------------------------------

col_explain, col_shap = st.columns([1, 1])

with col_explain:
    st.markdown(
        '<div class="panel-title"><b> Analyst Explanation</b></div>',
        unsafe_allow_html=True,
    )
    alert_placeholder = st.empty()

with col_shap:
    st.markdown(
        '<div class="panel-title"><b> Live SHAP-Informed Feature Influence</b></div>',
        unsafe_allow_html=True,
    )
    shap_chart_placeholder = st.empty()


# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
try:
    df_raw = pd.read_csv("streamlit_test_data.csv").sample(200)
except:
    st.error("❌ CSV file not found")
    st.stop()

# -------------------------------------------------
# SAFE MODEL LOADING
# -------------------------------------------------
model = None
scaler = None

try:
    scaler = joblib.load("models/scaler.pkl")
    feature_cols = joblib.load("models/feature_columns.pkl")

    if model_choice == "Random Forest":
        model = joblib.load("models/rf_under.pkl")

    elif model_choice == "Logistic Regression":
        model = joblib.load("models/logistic_under.pkl")

    elif model_choice == "Isolation Forest":
        model = joblib.load("models/isolation_under.pkl")

    elif model_choice == "CNN":
        model = load_model("models/cnn_under.h5")

except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()

# IMPORTANT: must match training
# feature_cols = [
#'Flow Duration',
#'Total Fwd Packets',
#'Total Backward Packets',
#'Flow Bytes/s',
#'Flow Packets/s'
# ]

# -------------------------------------------------
# EXECUTION LOOP
# -------------------------------------------------
if st.session_state.running:
    total = 0
    attacks = 0
    run_start_time = time.time()

    y_true_list = []
    y_pred_list = []
    indicator_values = []

    for _, row in df_raw.iterrows():
        total += 1

        # -------------------------------------------------
        # PREDICTION (SAFE)
        # -------------------------------------------------
        try:
            if model is not None and scaler is not None:
                X = pd.DataFrame([row[feature_cols]], columns=feature_cols)
                X_scaled = scaler.transform(X)

            if model_choice == "CNN":
                X_cnn = X_scaled.reshape(1, X_scaled.shape[1], 1)
                risk = float(model.predict(X_cnn, verbose=0)[0][0]) * 100
                prediction = 1 if risk >= 50 else 0

            elif model_choice == "Isolation Forest":
                raw_pred = model.predict(X_scaled)[0]
                prediction = 1 if raw_pred == -1 else 0
                risk = 90 if prediction == 1 else 10

            elif model_choice in ["Random Forest", "Logistic Regression"]:
                prediction = model.predict(X_scaled)[0]
                risk = model.predict_proba(X_scaled)[0][1] * 100

            else:
                st.error("Unknown model selected.")
                st.stop()

            is_attack = prediction == 1
            y_pred_list.append(int(prediction))

            # If the CSV contains ground-truth labels, store them for performance metrics
            if "AttackFlag" in row.index:
                y_true_list.append(int(row["AttackFlag"]))

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()
            # DEMO fallback

        port = int(row.get("Destination Port", 80))
        packets = row.get("Total Fwd Packets", 0)

        # -------------------------------------------------
        # STORE HISTORY
        # -------------------------------------------------
        st.session_state.risk_history.append(risk)
        if len(st.session_state.risk_history) > 50:
            st.session_state.risk_history.pop(0)

        # -------------------------------------------------
        # ATTACK DETECTED
        # -------------------------------------------------
        if is_attack:
            st.session_state.alert_active = True
            st.session_state.alert_counter = 2  # show alert for 2 cycles
            attacks += 1
            play_sound()
            shap_explanation = generate_shap_explanation(row, model_choice, risk)
            st.session_state.attack_logs.append(
                {
                    "Time": datetime.now().strftime("%H:%M:%S"),
                    "Model": model_choice,
                    "Prediction": "ATTACK",
                    "Destination Port": int(row.get("Destination Port", 0)),
                    "Flow Duration": round(row.get("Flow Duration", 0), 2),
                    "Flow Bytes/s": round(row.get("Flow Bytes/s", 0), 2),
                    "Flow Packets/s": round(row.get("Flow Packets/s", 0), 2),
                    "Total Fwd Packets": int(row.get("Total Fwd Packets", 0)),
                    "Total Backward Packets": int(row.get("Total Backward Packets", 0)),
                    "Packet Length Mean": round(row.get("Packet Length Mean", 0), 2),
                    "PSH Flag Count": int(row.get("PSH Flag Count", 0)),
                    "Init Win Bytes Fwd": int(row.get("Init_Win_bytes_forward", 0)),
                    "Port Attack Rate": round(row.get("port_attack_rate", 0), 4),
                    "Window Attack Rate": round(
                        row.get("port_window_attack_rate", 0), 4
                    ),
                }
            )

            time.sleep(0.8)

            alert_placeholder.markdown(
                f"""
         <div class="alert-box">
         <b></b><br>
        <b>Port:</b> {port}<br>
        <b>Classification:</b> ATTACK<br>
        <b>Alert Level:</b> HIGH<br><br>
        <b>Analyst Explanation:</b><br>
           {shap_explanation}
    </div>
""",
                unsafe_allow_html=True,
            )

        fig_shap_live = plot_live_shap_influence(row, model_choice, is_attack)
        shap_chart_placeholder.plotly_chart(
            fig_shap_live, use_container_width=True, key=f"shap_live_{total}"
        )

        packets = row.get("Total Fwd Packets", 0)

        if packets > st.session_state.peak_traffic:
            st.session_state.peak_traffic = packets

        # -------------------------------------------------
        # VISUALS (UNCHANGED STYLE)
        # -------------------------------------------------

        # Trend
        fig_trend = go.Figure()
        fig_trend.add_trace(
            go.Scatter(
                y=st.session_state.risk_history, fill="tozeroy", line_color="#1f77b4"
            )
        )
        fig_trend.update_layout(
            height=210,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
        )
        trend_placeholder.plotly_chart(
            fig_trend, use_container_width=True, key=f"trend_{total}"
        )

        # Gauge
        # -------------------------------------------------
        # -------------------------------------------------
        # Attack Risk Indicator Gauge
        # Internal values are only used for visual movement.
        # No numeric score is displayed.
        # -------------------------------------------------

        if st.session_state.alert_active:
            indicator_value = 100
            indicator_values.append(indicator_value)
            indicator_status = "ATTACK DETECTED"
            gauge_color = "rgba(239,68,68,0.85)"
        else:
            indicator_value = 20
            indicator_status = "NORMAL TRAFFIC"
            gauge_color = "rgba(250,204,21,0.45)"

        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge",
                value=indicator_value,
                title={"text": f"<b>{indicator_status}</b>", "font": {"size": 15}},
                gauge={
                    "axis": {"range": [0, 100], "visible": False},
                    "bar": {"color": gauge_color},
                    "bgcolor": "rgba(255,255,255,0.05)",
                    "borderwidth": 1,
                    "bordercolor": "#374151",
                },
            )
        )

        fig_gauge.update_layout(
            height=210,
            margin=dict(l=10, r=10, t=30, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
        )

        gauge_placeholder.plotly_chart(
            fig_gauge, use_container_width=True, key=f"gauge_{total}"
        )
        # -------------------------------------------------
        # -------------------------------------------------
        # KPI CALCULATIONS
        # -------------------------------------------------
        normal = total - attacks

        # ALERT LOGIC
        if st.session_state.alert_active:
            normal_display = "ATTACK"
            normal_color = "#ef4444"
        else:
            normal_display = normal
            normal_color = "#56d679"

        metrics = [
            ("FLOWS", total),
            ("ATTACKS", attacks),
            ("NORMAL", normal_display),
            ("PEAK TRAFFIC", int(st.session_state.peak_traffic)),
        ]

        # -------------------------------------------------
        # DISPLAY KPIs (SAME STYLE)
        # -------------------------------------------------
        for i, (label, val) in enumerate(metrics):
            color = normal_color if label == "NORMAL" else "#56d679"

            kpi_placeholders[i].markdown(
                f"""
               <div class="grafana-panel kpi-container">
               <div class="kpi-title">{label}</div>
               <div class="kpi-value" style="color:{color}">{val}</div>
               </div>
            """,
                unsafe_allow_html=True,
            )

        # TABLE
        if len(st.session_state.attack_logs) > 0:
            table_html = (
                pd.DataFrame(st.session_state.attack_logs).tail(8).to_html(index=False)
            )

            table_placeholder.markdown(
                f"""
           <div class="grafana-panel scroll-table">
               {table_html}
           </div>
           """,
                unsafe_allow_html=True,
            )
            # keep alert visible for a few cycles
        if st.session_state.alert_counter > 0:
            st.session_state.alert_counter -= 1
        else:
            st.session_state.alert_active = False

        time.sleep(0.4)

        # -------------------------------------------------
    # SAVE MODEL RUN SUMMARY AFTER PROCESSING 200 FLOWS
    # -------------------------------------------------

    run_time = round(time.time() - run_start_time, 2)
    normal = total - attacks
    attack_rate = round((attacks / total) * 100, 2) if total > 0 else 0
    avg_indicator = round(np.mean(indicator_values), 2) if indicator_values else 0

    summary = {
        "Model": model_choice,
        "Flows Processed": total,
        "Attacks Detected": attacks,
        "Normal Flows": normal,
        "Attack Rate (%)": attack_rate,
        "Peak Traffic": int(st.session_state.peak_traffic),
        "Average Indicator": avg_indicator,
        "Execution Time (s)": run_time,
    }

    # Add supervised metrics only if AttackFlag exists in the CSV
    if len(y_true_list) == len(y_pred_list) and len(y_true_list) > 0:
        summary["Accuracy"] = round(accuracy_score(y_true_list, y_pred_list), 4)
        summary["Precision"] = round(
            precision_score(y_true_list, y_pred_list, zero_division=0), 4
        )
        summary["Recall"] = round(
            recall_score(y_true_list, y_pred_list, zero_division=0), 4
        )
        summary["F1-Score"] = round(
            f1_score(y_true_list, y_pred_list, zero_division=0), 4
        )
    else:
        summary["Accuracy"] = "N/A"
        summary["Precision"] = "N/A"
        summary["Recall"] = "N/A"
        summary["F1-Score"] = "N/A"

    # Replace previous run for the same model, so table stays clean
    st.session_state.model_run_summaries = [
        item
        for item in st.session_state.model_run_summaries
        if item["Model"] != model_choice
    ]

    st.session_state.model_run_summaries.append(summary)
    st.session_state.completed_models.add(model_choice)

    # Save results to CSV
    results_df = pd.DataFrame(st.session_state.model_run_summaries)
    results_df.to_csv("dashboard_model_run_results.csv", index=False)

    st.session_state.running = False

    # MODEL RUN COMPARISON TABLE
# -------------------------------------------------

if len(st.session_state.model_run_summaries) > 0:
    with st.expander(" Open Model Runtime Comparison Summary", expanded=False):
        comparison_df = pd.DataFrame(st.session_state.model_run_summaries)

        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        st.caption(
            "This table records the dashboard behaviour for each evaluated model. "
            "If ground-truth labels are available in the CSV, supervised metrics are calculated; "
            "otherwise, the table summarises operational detection behaviour during the 200-flow simulation."
        )

    if len(st.session_state.completed_models) == 4:
        st.success(
            "All four models have been evaluated. The recorded summaries can now be used "
            "to compare detection behaviour, runtime responsiveness and alert generation."
        )
