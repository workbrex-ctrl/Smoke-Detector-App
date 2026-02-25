import streamlit as st
import pandas as pd
import plotly.express as px
import time
from sklearn.linear_model import LinearRegression
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Smoke Detector", layout="wide", initial_sidebar_state="collapsed")

# --- CUSTOM CSS (Forced Dark Mode) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        font-family: 'Inter', sans-serif;
        background-color: #0d0d0d !important;
        color: #ffffff !important;
    }
    .block-container { padding-top: 2rem !important; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .bento-card {
        background-color: #1a1a1c !important;
        border-radius: 24px;
        padding: 24px;
        box-shadow: 0 4px 30px rgba(0, 255, 128, 0.05);
        border: 1px solid rgba(0, 255, 128, 0.15);
        margin-bottom: 20px;
        color: #ffffff !important;
    }
    .main-title { font-size: 36px; font-weight: 700; color: #ffffff !important; margin-top: -20px; }
    .sub-title { font-size: 16px; color: #888888 !important; margin-bottom: 30px; }
    .metric-label { font-size: 12px; color: #a0a0a0 !important; text-transform: uppercase; letter-spacing: 1.5px; }
    .metric-value { font-size: 40px; font-weight: 700; color: #00ff80 !important; text-shadow: 0 0 15px rgba(0, 255, 128, 0.4); }
    .metric-value-medium { font-size: 28px; font-weight: 700; color: #00ff80 !important; }
    </style>
""", unsafe_allow_html=True)

# --- STATE MANAGEMENT ---
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# --- SCREEN 1: UPLOAD PAGE ---
if not st.session_state.data_loaded:
    st.write("") 
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div class='bento-card' style='text-align: center;'><h2 style='color: #00ff80;'>⚡ System Initialization</h2><p style='color: #888888;'>Upload your SD Card <code>thesis_log.csv</code> to train the AI.</p></div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=['csv'])
        if uploaded_file is not None:
            with st.spinner("Training Linear Regression Model..."):
                time.sleep(1.5) 
                df = pd.read_csv(uploaded_file, names=['Datetime', 'Gas', 'PM', 'Temp', 'Status'])
                df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
                st.session_state.full_df = df
                st.session_state.data_loaded = True
                st.rerun()

# --- SCREEN 2: MAIN DASHBOARD ---
else:
    full_df = st.session_state.full_df
    alert_df = full_df[full_df['Status'].str.contains("DETECTED|EMERGENCY", na=False)].copy()
    
    # Header
    st.markdown("<div class='main-title'>⚡ AI Predictive Alerts Control Center</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Machine Learning Analysis & Edge Calibration</div>", unsafe_allow_html=True)

    # --- AI CALCULATIONS (The "Magic Numbers") ---
    # Convert Status to a numeric "Risk Score" for training (0 for Clear, 100 for Fire, etc)
    def map_risk(status):
        if "FIRE" in status: return 100
        if "DETECTED" in status: return 65
        return 10
    
    train_df = full_df.dropna().copy()
    train_df['RiskScore'] = train_df['Status'].apply(map_risk)
    
    # Train Linear Regression: Risk = B0 + (B1*Gas) + (B2*PM) + (B3*Temp)
    X = train_df[['Gas', 'PM', 'Temp']]
    y = train_df['RiskScore']
    model = LinearRegression().fit(X, y)
    
    b0 = round(model.intercept_, 4)
    b1, b2, b3 = [round(c, 4) for c in model.coef_]

    # --- METRICS ---
    total_incidents = len(alert_df)
    if total_incidents > 0:
        alert_df['Hour'] = alert_df['Datetime'].dt.hour
        alert_df['DayName'] = alert_df['Datetime'].dt.day_name()
        peak_hour = alert_df['Hour'].mode()[0]
        peak_time_str = f"{peak_hour % 12 or 12}:00 {'AM' if peak_hour < 12 else 'PM'}"
        peak_day = alert_df['DayName'].mode()[0]
    else:
        peak_hour, peak_time_str, peak_day = 12, "N/A", "N/A"

    # --- UI ROW 1 ---
    c1, c2 = st.columns([1, 2.5])
    with c1:
        st.markdown(f"<div class='bento-card'><div class='metric-label'>🚨 Total Incidents</div><div class='metric-value'>{total_incidents}</div><hr style='border-color: #2a2a2c; margin: 20px 0;'><div class='metric-label'>🕒 Peak Violation Time</div><div class='metric-value-medium'>{peak_time_str}</div><hr style='border-color: #2a2a2c; margin: 20px 0;'><div class='metric-label'>📅 Highest Risk Day</div><div class='metric-value-medium'>{peak_day}</div></div>", unsafe_allow_html=True)

    with c2:
        if total_incidents > 0:
            daily_counts = alert_df.groupby(alert_df['Datetime'].dt.date).size().reset_index(name='Incidents')
            fig = px.line(daily_counts, x='Datetime', y='Incidents', markers=True, title="📊 Incident Trends")
            fig.update_traces(line_color='#00ff80', line_width=3)
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#ffffff'))
            st.plotly_chart(fig, use_container_width=True)

    # --- AI CODE GENERATION ---
    with st.expander("🤖 GENERATE STAGE 2 AI CODE"):
        st.info("The coefficients below were calculated mathematically from your uploaded CSV data.")
        
        final_code = f"""// --- UPDATE THESE AT THE TOP OF YOUR SKETCH ---
float INTERCEPT = {b0};
float COEF_GAS  = {b1};
float COEF_PM   = {b2};
float COEF_TEMP = {b3};

// --- PASTE THIS IN THE DELETION ZONE ---
    int peakHour = {peak_hour}; 
    if (now.hour() == peakHour || now.hour() == peakHour - 1) {{
        samplingRate = 500;
    }} else {{
        samplingRate = 2000;
    }}

    float riskScore = INTERCEPT + (COEF_GAS * gas) + (COEF_PM * currentPM25) + (COEF_TEMP * temp);

    if (riskScore >= 80.0) {{
      status = "FIRE DETECTED";
      digitalWrite(LED_PIN, HIGH);
      sendSMS("CRITICAL ALERT: Fire signature detected!");
    }} 
    else if (riskScore >= 50.0) {{
      digitalWrite(LED_PIN, HIGH); 
      if (currentPM25 > gas) {{
        status = "VAPE DETECTED";
        sendSMS("SECURITY ALERT: Vaping detected (AI Profile).");
      }} else {{
        status = "CIGARETTE DETECTED";
        sendSMS("SECURITY ALERT: Cigarette smoking detected (AI Profile).");
      }}
    }} 
    else {{
      digitalWrite(LED_PIN, LOW);
    }}"""
        st.code(final_code, language='cpp')
