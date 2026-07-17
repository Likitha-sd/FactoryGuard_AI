import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dashboard.history import get_history_dataframe
from src.database.database import SessionLocal
from src.dashboard.analytics import load_dashboard_data

import pandas as pd
import streamlit as st
import requests
from datetime import datetime
# ==========================================================
# Configuration
# ==========================================================

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="FactoryGuard AI",
    page_icon="🏭",
    layout="wide",
)

# ==========================================================
# Sidebar
# ==========================================================

st.sidebar.title("🏭 FactoryGuard AI")

st.sidebar.markdown("### Industrial Predictive Maintenance")

st.sidebar.info(
    """
This application predicts machine failures using a trained
Random Forest model with automatic feature engineering.

🏭 FactoryGuard AI

Industrial Monitoring Platform

────────────────────────────

FactoryGuard AI v2.0 """
)

st.sidebar.markdown("---")
st.sidebar.success("🟢 System Status: Online")

# ==========================================================
# Header
# ==========================================================

st.title("🏭 FactoryGuard AI")

st.markdown(
    """ FactoryGuard AI

Industrial Equipment Monitoring

Monitor machine health, assess failure risk,
and support preventive maintenance decisions. """
)

st.divider()

# ==========================================================
# Input Section
# ==========================================================

st.subheader("📋 Machine Information")

col1, col2 = st.columns(2)

with col1:

    machine_id = st.selectbox(
        "Machine ID",
        [
            "MACHINE_01",
            "MACHINE_02",
            "MACHINE_03",
            "MACHINE_04",
            "MACHINE_05",
        ],
    )

    temperature = st.number_input(
        "🌡 Temperature (°C)",
        value=72.0,
        step=0.1,
    )

    vibration = st.number_input(
        "📳 Vibration (mm/s)",
        value=3.0,
        step=0.1,
    )

    pressure = st.number_input(
        "⚙ Pressure (kPa)",
        value=120.0,
        step=0.1,
    )

with col2:

    rpm = st.number_input(
        "🔄 Rotational Speed (RPM)",
        value=1450,
    )

    power = st.number_input(
        "⚡ Power Consumption (kW)",
        value=12.0,
        step=0.1,
    )

    operating_hours = st.number_input(
        "⏱ Operating Hours",
        value=2500,
    )

# ==========================================================
# Sensor Summary
# ==========================================================

st.divider()

st.subheader("📊 Live Sensor Summary")

c1, c2, c3 = st.columns(3)

c1.metric("🌡 Temperature", f"{temperature:.1f} °C")
c2.metric("📳 Vibration", f"{vibration:.2f} mm/s")
c3.metric("⚙ Pressure", f"{pressure:.1f} kPa")

c4, c5, c6 = st.columns(3)

c4.metric("🔄 RPM", f"{rpm}")
c5.metric("⚡ Power", f"{power:.1f} kW")
c6.metric("⏱ Hours", f"{operating_hours}")

st.divider()

# ==========================================================
# Prediction Button
# ==========================================================

if st.button("🔍 Predict Machine Health", use_container_width=True):

    payload = {
        "machine_id": machine_id,
        "temperature_c": temperature,
        "vibration_mm_s": vibration,
        "pressure_kpa": pressure,
        "rotational_speed_rpm": rpm,
        "power_consumption_kw": power,
        "operating_hours": operating_hours,
    }

    try:

        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:

            result = response.json()

            probability = result["failure_probability"] * 100

            st.divider()

            st.subheader("📈 Prediction Result")

            left, right = st.columns([2, 1])

            with left:

                if result["prediction"] == "Machine Failure":

                    st.error("🚨 MACHINE FAILURE PREDICTED")

                else:

                    st.success("✅ MACHINE OPERATING NORMALLY")

            with right:

                st.metric(
                    "Failure Probability",
                    f"{probability:.2f}%"
                )

            st.progress(min(probability / 100, 1.0))

            if probability < 30:

                st.success("🟢 Risk Level: LOW")

                st.info(
                    """
**Recommendation**

Continue normal operation.

No immediate maintenance action is required.
"""
                )

            elif probability < 70:

                st.warning("🟡 Risk Level: MEDIUM")

                st.warning(
                    """
**Recommendation**

Schedule preventive maintenance soon.
"""
                )

            else:

                st.error("🔴 Risk Level: HIGH")

                st.error(
                    """
**Recommendation**

Immediate inspection is recommended to prevent failure.
"""
                )

            st.caption(
                f"Prediction generated on {datetime.now().strftime('%d %B %Y • %I:%M:%S %p')}"
            )

        else:

            st.error(response.text)

    except Exception as e:

        st.error(
            f"""
Unable to connect to the FastAPI server.

Error:
{e}
"""
        )

# ==========================================================
# Analytics Dashboard
# ==========================================================

st.divider()

st.header("📊 Analytics Dashboard")

db = SessionLocal()

analytics = load_dashboard_data(db)

db.close()

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Predictions",
    analytics["total"]
)

col2.metric(
    "Normal",
    analytics["normal"]
)

col3.metric(
    "Failures",
    analytics["failure"]
)

st.divider()

st.subheader("📜 Prediction History")

history_df = get_history_dataframe()

if history_df.empty:

    st.info("No predictions available.")

else:

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True,
    )
# ==========================================================
# Footer
# ==========================================================

st.divider()

st.caption(
    "🏭 FactoryGuard AI | Built using Streamlit • FastAPI • Scikit-learn"
)