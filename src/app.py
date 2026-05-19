import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="ED Capacity Forecast", page_icon="🏥", layout="centered")

# --- CUSTOM MINIMALIST STYLING ---
st.markdown("""
    <style>
    .main .block-container { padding-top: 2rem; max-width: 800px; }
    h1 { font-weight: 700; color: #1A1A1A; letter-spacing: -0.5px; }
    .metric-card { background-color: #F8F9FA; border: 1px solid #EAEAEA; padding: 1.5rem; border-radius: 8px; text-align: center; margin-top: 1rem; }
    .metric-value { font-size: 3rem; font-weight: 700; color: #0076FF; }
    .metric-label { font-size: 1rem; color: #666666; text-transform: uppercase; }
    </style>
""", unsafe_allow_html=True)

# --- LOAD BOTH MODELS ---
@st.cache_resource
def load_models():
    # 1. Get the directory where app.py lives (the 'src' folder)
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Go UP one level, and then DOWN into the 'models' folder
    MODELS_DIR = os.path.join(SRC_DIR, '..', 'models')
    
    # 3. Stitch the full path together with the exact file names
    tactical_path = os.path.join(MODELS_DIR, 'best_ed_forecaster.pkl')
    strategic_path = os.path.join(MODELS_DIR, 'long_term_forecaster.pkl')
    
    # 4. Load the models using these bulletproof paths
    tactical = joblib.load(tactical_path)
    strategic = joblib.load(strategic_path)
    
    return tactical, strategic

tactical_model, strategic_model = load_models()

# --- HEADER ---
st.title("🏥 Emergency Department Forecaster")
st.markdown("Anticipate hourly patient arrival trends for resource and staffing allocation.")
st.markdown("---")

# --- HYBRID UI ROUTING (TABS) ---
tab1, tab2 = st.tabs(["⏱️ Short-Term (Immediate Shift)", "📅 Long-Term (Strategic Planning)"])

# ==========================================
# TAB 1: TACTICAL (REQUIRES LAGS)
# ==========================================
with tab1:
    st.write("### 🚨 Immediate Predictive Window")
    st.markdown("<small style='color:#666;'>Use this for the next 0-24 hours. Highest accuracy.</small>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1: target_date_1 = st.date_input("Select Date", datetime.date.today(), key="d1")
    with c2: target_time_1 = st.time_input("Select Hour", datetime.time(12, 0), key="t1")
    
    st.write("### ⏱️ Recent Arrivals (Context)")
    c3, c4 = st.columns(2)
    with c3:
        lag_1h = st.number_input("Past Hour", min_value=0, value=4, step=1)
        lag_2h = st.number_input("2 Hours Ago", min_value=0, value=3, step=1)
    with c4:
        lag_24h = st.number_input("Yesterday (Same Hour)", min_value=0, value=4, step=1)
        lag_168h = st.number_input("Last Week (Same Hour)", min_value=0, value=5, step=1)
        
    if st.button("Generate Tactical Forecast", type="primary", key="btn1"):
        # Calculate broader rolling features based on inputs
        rolling_mean_6h = np.mean([lag_1h, lag_2h, 3, 4, 2, 3])
        rolling_mean_12h = np.mean([lag_1h, lag_2h, 3, 4, 2, 3, 4, 3, 2, 3, 4, 2])
        rolling_mean_24h = np.mean([lag_1h, lag_24h, 3, 4, 2, 4, 3, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3])

        input_data = pd.DataFrame([{
            'Hour': target_time_1.hour, 'DayOfWeek': target_date_1.weekday(),
            'Month': target_date_1.month, 'Is_Weekend': 1 if target_date_1.weekday() >= 5 else 0,
            'Lag_1h': lag_1h, 'Lag_2h': lag_2h, 'Lag_24h': lag_24h, 'Lag_168h': lag_168h,
            'Rolling_Mean_6h': rolling_mean_6h, 'Rolling_Mean_12h': rolling_mean_12h, 'Rolling_Mean_24h': rolling_mean_24h
        }])
        
        pred = max(0, round(tactical_model.predict(input_data)[0]))
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Tactical Patient Projection</div>
                <div class="metric-value">{pred} <span style='font-size:1.2rem; color:#333;'>patients / hr</span></div>
                <p style='color:#666; font-size:0.85rem;'>Margin of Error: ~1.5 patients</p>
            </div>
        """, unsafe_allow_html=True)

# ==========================================
# TAB 2: STRATEGIC (CALENDAR ONLY)
# ==========================================
with tab2:
    st.write("### 🔮 Future Predictive Window")
    st.markdown("<small style='color:#666;'>Use this for dates weeks or months in advance. Based purely on historical seasonality.</small>", unsafe_allow_html=True)
    
    c5, c6 = st.columns(2)
    with c5: target_date_2 = st.date_input("Select Future Date", datetime.date(2026, 7, 12), key="d2")
    with c6: target_time_2 = st.time_input("Select Hour", datetime.time(14, 0), key="t2")
    
    if st.button("Generate Strategic Forecast", type="primary", key="btn2"):
        input_data = pd.DataFrame([{
            'Hour': target_time_2.hour,
            'DayOfWeek': target_date_2.weekday(),
            'Month': target_date_2.month,
            'Is_Weekend': 1 if target_date_2.weekday() >= 5 else 0
        }])
        
        pred = max(0, round(strategic_model.predict(input_data)[0]))
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Strategic Patient Projection</div>
                <div class="metric-value">{pred} <span style='font-size:1.2rem; color:#333;'>patients / hr</span></div>
                <p style='color:#666; font-size:0.85rem;'>Based on baseline historical trends.</p>
            </div>
        """, unsafe_allow_html=True)
