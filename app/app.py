import os
import sys
import tempfile
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Allow importing backend pipeline from scripts folder
sys.path.append(os.path.abspath("scripts"))
from step8b_final_combined_pipeline import run_combined_pipeline
from attack_intelligence import get_attack_intelligence

# ---------------------------------------------------
# Page Config
# ---------------------------------------------------
st.set_page_config(
    page_title="VPN-Aware Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
)

# ---------------------------------------------------
# Custom CSS — UPDATED: converted from dark SOC theme to clean light theme
# ---------------------------------------------------
st.markdown("""
<style>
    /* UPDATED: pure white background, removed dark gradients and cyber grid */
    .stApp {
        background: #FFFFFF;
        color: #111111;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1500px;
    }

    /* UPDATED: hero card — light blue gradient, dark text, subtle shadow */
    .hero-card {
        position: relative;
        background: linear-gradient(135deg, #eef2ff, #dbeafe);
        border: 1px solid #c7d2fe;
        border-radius: 20px;
        padding: 30px 32px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
        margin-bottom: 20px;
        overflow: hidden;
    }

    /* UPDATED: removed neon radial glow overlay */
    .hero-card::before {
        content: "";
        position: absolute;
        top: -40%;
        right: -10%;
        width: 280px;
        height: 280px;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.08) 0%, transparent 70%);
        pointer-events: none;
    }

    /* UPDATED: dark text for hero title */
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1e293b;
        margin-bottom: 8px;
        letter-spacing: 0.2px;
    }

    /* UPDATED: dark subtitle text */
    .hero-subtitle {
        font-size: 1rem;
        color: #475569;
        line-height: 1.7;
        max-width: 920px;
    }

    /* UPDATED: glass card → light card with subtle shadow */
    .glass-card {
        background: #FFFFFF;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 18px 18px 12px 18px;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.05);
        margin-bottom: 18px;
    }

    /* UPDATED: metric box — white background, soft border and shadow */
    .metric-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 14px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }

    /* UPDATED: footer card — light background, dark text */
    .footer-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 18px;
        text-align: center;
        color: #475569;
        margin-top: 20px;
    }

    /* UPDATED: small note — medium gray for readability */
    .small-note {
        color: #64748b;
        font-size: 0.92rem;
    }

    /* UPDATED: metric label — dark gray text */
    div[data-testid="stMetric"] {
        background: transparent;
        border-radius: 12px;
        padding: 6px 2px 2px 2px;
    }

    div[data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-weight: 600;
    }

    /* UPDATED: metric value — near-black text */
    div[data-testid="stMetricValue"] {
        color: #1e293b !important;
        font-weight: 800;
    }

    /* UPDATED: buttons — white with border, no neon glow */
    .stButton > button {
        width: 100%;
        background: #FFFFFF;
        color: #1e293b;
        border: 1px solid #cbd5e1;
        border-radius: 12px;
        padding: 0.8rem 1rem;
        font-weight: 700;
        font-size: 1rem;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.06);
        transition: background 0.2s;
    }

    .stButton > button:hover {
        background: #f1f5f9;
        color: #1e293b;
    }

    /* UPDATED: download button — green with white text, no glow */
    .stDownloadButton > button {
        width: 100%;
        background: #059669;
        color: #FFFFFF;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 1rem;
        font-weight: 700;
        font-size: 1rem;
        box-shadow: 0 2px 8px rgba(5, 150, 105, 0.15);
    }

    .stDownloadButton > button:hover {
        background: #047857;
        color: #FFFFFF;
    }

    /* UPDATED: file uploader — light background, clean border */
    [data-testid="stFileUploader"] {
        background: #f8fafc;
        border: 1px dashed #94a3b8;
        border-radius: 14px;
        padding: 10px;
    }

    /* UPDATED: force file uploader inner area to light theme */
    [data-testid="stFileUploader"] section {
        background: #f8fafc !important;
        border-radius: 12px !important;
    }

    [data-testid="stFileUploader"] section > div {
        background: #f8fafc !important;
        color: #1e293b !important;
    }

    /* UPDATED: drag-drop zone — light bg, dark text */
    [data-testid="stFileDropzoneInstructions"] {
        color: #1e293b !important;
    }

    [data-testid="stFileDropzoneInstructions"] div,
    [data-testid="stFileDropzoneInstructions"] span,
    [data-testid="stFileDropzoneInstructions"] small {
        color: #475569 !important;
    }

    /* UPDATED: uploaded file name text — black */
    [data-testid="stFileUploader"] [data-testid="stMarkdownContainer"],
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] span {
        color: #1e293b !important;
    }

    [data-testid="stFileUploader"] button {
        background-color: #FFFFFF !important;
        color: #1e293b !important;
        border-radius: 12px !important;
        border: 1px solid #cbd5e1 !important;
        font-weight: 700;
    }

    [data-testid="stFileUploader"] button:hover {
        background-color: #f1f5f9 !important;
        color: #1e293b !important;
    }

    /* UPDATED: file uploader SVG icon — dark color */
    [data-testid="stFileUploader"] svg {
        fill: #64748b !important;
        color: #64748b !important;
    }

    /* UPDATED: threat banners — light tinted backgrounds, no neon glow */
    .threat-low {
        padding: 14px 18px;
        border-radius: 14px;
        border: 1px solid #86efac;
        background: #f0fdf4;
        box-shadow: 0 2px 8px rgba(34, 197, 94, 0.08);
        font-weight: 700;
        color: #166534;
        margin-bottom: 16px;
    }

    .threat-medium {
        padding: 14px 18px;
        border-radius: 14px;
        border: 1px solid #fcd34d;
        background: #fffbeb;
        box-shadow: 0 2px 8px rgba(245, 158, 11, 0.08);
        font-weight: 700;
        color: #92400e;
        margin-bottom: 16px;
    }

    /* UPDATED: removed pulsing animation from medium threat */

    .threat-high {
        padding: 14px 18px;
        border-radius: 14px;
        border: 1px solid #fca5a5;
        background: #fef2f2;
        box-shadow: 0 2px 8px rgba(239, 68, 68, 0.10);
        font-weight: 800;
        color: #991b1b;
        margin-bottom: 16px;
        animation: pulseRed 1.8s infinite;
    }

    /* UPDATED: subtle pulse — shadow only, no neon glow */
    @keyframes pulseRed {
        0% { box-shadow: 0 2px 8px rgba(239, 68, 68, 0.10); }
        50% { box-shadow: 0 4px 16px rgba(239, 68, 68, 0.18); }
        100% { box-shadow: 0 2px 8px rgba(239, 68, 68, 0.10); }
    }

    /* UPDATED: all text dark for light theme */
    h1, h2, h3, h4, h5, h6, p, label, div {
        color: #111111;
    }

    /* UPDATED: dataframe header and text styling */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
    }

    /* UPDATED: expander styling for light theme */
    .streamlit-expanderHeader {
        color: #1e293b !important;
        font-weight: 600;
    }

    /* UPDATED: dataframe — light theme via Streamlit's built-in theming */
    [data-testid="stDataFrame"] {
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        overflow: hidden;
    }

    /* UPDATED: force all Streamlit alert/success text to dark */
    [data-testid="stAlert"] p,
    [data-testid="stAlert"] div {
        color: #1e293b !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Helper Functions
# ---------------------------------------------------
def calculate_soc_metrics(results: dict) -> dict:
    total_records = results.get("total_records", 0)
    input_type = results.get("input_type", "unsupported")

    attack_counts = results.get("attack_counts", {})
    vpn_counts = results.get("vpn_counts", {})
    drift = results.get("concept_drift", {})

    total_attacks = sum(attack_counts.values()) if attack_counts else 0
    attack_percentage = (total_attacks / total_records * 100) if total_records > 0 and attack_counts else 0

    vpn_traffic = vpn_counts.get("VPN", 0) + vpn_counts.get("vpn", 0)
    vpn_percentage = (vpn_traffic / total_records * 100) if total_records > 0 and vpn_counts else 0

    filtered_attacks = {
        k: v for k, v in attack_counts.items()
        if str(k).lower() not in ["normal", "unknown"]
    }

    if filtered_attacks:
        top_attack = max(filtered_attacks, key=filtered_attacks.get)
    elif attack_counts:
        top_attack = max(attack_counts, key=attack_counts.get)
    else:
        top_attack = "N/A"

    js_distance = drift.get("js_distance", 0.0)
    drift_detected = drift.get("drift_detected", False)

    risk_score = min(
        100,
        round((attack_percentage * 0.5) + (vpn_percentage * 0.2) + (js_distance * 100 * 0.3), 2)
    )

    if risk_score < 35:
        threat_level = "Low"
    elif risk_score < 70:
        threat_level = "Medium"
    else:
        threat_level = "High"

    if drift_detected and threat_level == "Low":
        threat_level = "Medium"

    if input_type == "unsupported":
        threat_level = "N/A"
        top_attack = "N/A"
        risk_score = 0

    return {
        "total_attacks": total_attacks,
        "attack_percentage": round(attack_percentage, 2),
        "vpn_percentage": round(vpn_percentage, 2),
        "top_attack": top_attack,
        "risk_score": risk_score,
        "threat_level": threat_level,
    }


def build_attack_chart(attack_counts: dict):
    attack_df = pd.DataFrame(
        list(attack_counts.items()),
        columns=["Attack Type", "Count"]
    ).sort_values(by="Count", ascending=False)

    attack_df["Frame"] = "Current Window"

    fig = px.bar(
        attack_df,
        x="Attack Type",
        y="Count",
        color="Attack Type",
        text="Count",
        title="Attack Type Distribution",
        animation_frame="Frame"
    )

    # UPDATED: chart styling — white background, black text on axes/ticks/labels
    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#f8fafc",
        font=dict(color="#111111"),
        title_font=dict(size=20, color="#111111"),
        showlegend=False,
        xaxis_title="Attack Type",
        yaxis_title="Count",
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(
            gridcolor="#e2e8f0",
            linecolor="#cbd5e1",
            tickfont=dict(color="#111111", size=12),
            title_font=dict(color="#111111", size=14),
        ),
        yaxis=dict(
            gridcolor="#e2e8f0",
            linecolor="#cbd5e1",
            tickfont=dict(color="#111111", size=12),
            title_font=dict(color="#111111", size=14),
        ),
    )
    fig.update_traces(textposition="outside")
    return fig, attack_df.drop(columns=["Frame"])


def build_vpn_chart(vpn_counts: dict):
    vpn_df = pd.DataFrame(
        list(vpn_counts.items()),
        columns=["VPN Type", "Count"]
    ).sort_values(by="Count", ascending=False)

    fig = px.pie(
        vpn_df,
        values="Count",
        names="VPN Type",
        hole=0.58,
        title="VPN vs Non-VPN Traffic"
    )

    # UPDATED: chart styling — white background, black text on labels/legend
    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        font=dict(color="#111111"),
        title_font=dict(size=20, color="#111111"),
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", y=-0.1, font=dict(color="#111111", size=12))
    )
    fig.update_traces(textinfo="percent+label", pull=[0.04] * len(vpn_df))
    return fig, vpn_df


def build_risk_gauge(risk_score: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        # UPDATED: dark text for gauge title and number
        title={"text": "Risk Score", "font": {"size": 22, "color": "#1e293b"}},
        number={"suffix": "/100", "font": {"color": "#1e293b"}},
        gauge={
            # UPDATED: light axis, meaningful step colors preserved
            "axis": {"range": [0, 100], "tickcolor": "#64748b"},
            "bar": {"color": "#3b82f6"},
            "bgcolor": "#f1f5f9",
            "borderwidth": 1,
            "bordercolor": "#e2e8f0",
            "steps": [
                {"range": [0, 35], "color": "#dcfce7"},
                {"range": [35, 70], "color": "#fef3c7"},
                {"range": [70, 100], "color": "#fee2e2"},
            ],
        }
    ))

    # UPDATED: white background, dark text
    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        font={"color": "#1e293b"},
        margin=dict(l=20, r=20, t=40, b=20),
        height=320
    )
    return fig


def render_threat_banner(level: str):
    if level == "Low":
        st.markdown('<div class="threat-low">✅ Network Status: Secure</div>', unsafe_allow_html=True)
    elif level == "Medium":
        st.markdown('<div class="threat-medium">⚠️ Network Status: Suspicious Activity Detected</div>', unsafe_allow_html=True)
    elif level == "High":
        st.markdown('<div class="threat-high">🚨 Network Status: High Risk — Immediate Investigation Required</div>', unsafe_allow_html=True)
    else:
        st.info("ℹ️ Network Status: Not Available")


# ---------------------------------------------------
# Header
# ---------------------------------------------------
st.markdown("""
<div class="hero-card">
    <div class="hero-title">🛡️ VPN-Aware Intrusion Detection System</div>
    <div class="hero-subtitle">
        AI-powered network security monitoring platform for intrusion detection,
        VPN traffic classification, and concept drift detection.
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Upload Section
# ---------------------------------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Upload Network Traffic Dataset")
st.markdown('<div class="small-note">Supported formats: CSV and Excel (.xlsx)</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"], label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# Main Workflow
# ---------------------------------------------------
if uploaded_file is not None:
    st.success(f"Uploaded file: {uploaded_file.name}")

    if st.button("Run Security Analysis"):
        suffix = ".xlsx" if uploaded_file.name.endswith(".xlsx") else ".csv"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_file_path = tmp_file.name

        try:
            with st.spinner("Analyzing traffic patterns, predicting attacks, and computing security risk..."):
                results = run_combined_pipeline(temp_file_path)

            soc = calculate_soc_metrics(results)
            st.success("Analysis completed successfully.")

            # Top Metrics
            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric("Total Records", results["total_records"])
                st.markdown('</div>', unsafe_allow_html=True)

            with c2:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric("Detected Input Type", results.get("input_type", "N/A").upper())
                st.markdown('</div>', unsafe_allow_html=True)

            with c3:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric("Predicted Attacks", soc["total_attacks"])
                st.markdown('</div>', unsafe_allow_html=True)

            with c4:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric("Threat Level", soc["threat_level"])
                st.markdown('</div>', unsafe_allow_html=True)

            st.write("")
            render_threat_banner(soc["threat_level"])
            st.write("")

            # SOC Metrics
            a1, a2, a3, a4 = st.columns(4)

            with a1:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric("Attack %", f"{soc['attack_percentage']}%")
                st.markdown('</div>', unsafe_allow_html=True)

            with a2:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric("VPN Traffic %", f"{soc['vpn_percentage']}%")
                st.markdown('</div>', unsafe_allow_html=True)

            with a3:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric("Top Attack Type", soc["top_attack"])
                st.markdown('</div>', unsafe_allow_html=True)

            with a4:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric("Risk Score", f"{soc['risk_score']}/100")
                st.markdown('</div>', unsafe_allow_html=True)

            st.write("")

            # Charts
            left, right = st.columns(2)

            with left:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.subheader("Attack Analytics")
                if results["attack_counts"]:
                    attack_fig, attack_df = build_attack_chart(results["attack_counts"])
                    st.dataframe(attack_df, use_container_width=True, hide_index=True)
                    st.plotly_chart(attack_fig, use_container_width=True)
                else:
                    st.info("No IDS results available for this file.")
                st.markdown('</div>', unsafe_allow_html=True)

            with right:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.subheader("VPN Traffic Analytics")
                if results["vpn_counts"]:
                    vpn_fig, vpn_df = build_vpn_chart(results["vpn_counts"])
                    st.dataframe(vpn_df, use_container_width=True, hide_index=True)
                    st.plotly_chart(vpn_fig, use_container_width=True)
                else:
                    st.info("No VPN results available for this file.")
                st.markdown('</div>', unsafe_allow_html=True)

            # Gauge + Drift
            g1, g2 = st.columns([1, 1])

            with g1:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.subheader("Threat Gauge")
                st.plotly_chart(build_risk_gauge(soc["risk_score"]), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with g2:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.subheader("Concept Drift Monitoring")
                if results["concept_drift"]:
                    drift = results["concept_drift"]
                    d1, d2 = st.columns(2)
                    with d1:
                        st.metric("JS Distance", f"{drift['js_distance']:.4f}")
                    with d2:
                        st.metric("Threshold", f"{drift['threshold']:.2f}")

                    if drift["drift_detected"]:
                        st.error(drift["drift_status"])
                    else:
                        st.success(drift["drift_status"])
                else:
                    st.info("Concept drift result not available for this file.")
                st.markdown('</div>', unsafe_allow_html=True)

            # Security Insights
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Security Insights")
            i1, i2, i3 = st.columns(3)

            with i1:
                st.info(f"Top Attack Type: {soc['top_attack']}")

            with i2:
                st.info(f"Risk Score: {soc['risk_score']} / 100")

            with i3:
                st.info(f"Attack Percentage: {soc['attack_percentage']}%")
            st.markdown('</div>', unsafe_allow_html=True)

            # Attack Intelligence Section
            if results["attack_counts"]:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.subheader("AI-Powered Attack Intelligence")

                for attack_name, count in results["attack_counts"].items():
                    intel = get_attack_intelligence(attack_name)

                    with st.expander(f"{attack_name} ({count} detected)"):
                        st.write(f"**Severity:** {intel['severity']}")
                        st.write(f"**Danger Level:** {intel['danger_level']}")
                        st.write(f"**Description:** {intel['description']}")
                        st.write(f"**System Effect:** {intel['system_effect']}")
                        st.write(f"**Performance Impact:** {intel['performance_impact']}")
                        st.write(f"**Recommended Action:** {intel['recommendation']}")

                st.markdown('</div>', unsafe_allow_html=True)

            # Warnings
            if results["warnings"]:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.subheader("System Warnings")
                for warning in results["warnings"]:
                    st.warning(warning)
                st.markdown('</div>', unsafe_allow_html=True)

            # Download
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Download Prediction Report")
            output_file_path = results.get("output_file")

            if output_file_path and os.path.exists(output_file_path):
                with open(output_file_path, "rb") as f:
                    st.download_button(
                        label="Download Final Prediction CSV",
                        data=f,
                        file_name="final_combined_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            else:
                st.info("Output file not available.")
            st.markdown('</div>', unsafe_allow_html=True)

            # Footer
            st.markdown("""
            <div class="footer-card">
                Developed by <b>G5 - CSE</b><br>
                Final Year Project • Computer Science & Engineering<br>
                <span class="small-note">VPN-Aware Intrusion Detection using Machine Learning</span>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

else:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.info("Upload a dataset to start the security analysis.")
    st.markdown('</div>', unsafe_allow_html=True)
