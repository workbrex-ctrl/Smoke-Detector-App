import streamlit as st
import pandas as pd
import plotly.express as px
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Smoke Detector", layout="wide", initial_sidebar_state="collapsed")

# --- CUSTOM CSS (Dark Mode, Bento Box, Green Glow) ---
st.markdown("""
    <style>
    /* Global Dark Theme & Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0d0d0d !important;
        color: #ffffff;
    }
    
    /* Pull the whole page up to remove top spacing */
    .block-container {
        padding-top: 2rem !important;
    }

    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Bento Box Card Style */
    .bento-card {
        background-color: #1a1a1c;
        border-radius: 24px;
        padding: 24px;
        box-shadow: 0 4px 30px rgba(0, 255, 128, 0.05);
        border: 1px solid rgba(0, 255, 128, 0.15);
        transition: all 0.3s ease;
        margin-bottom: 20px;
    }
    .bento-card:hover {
        box-shadow: 0 0 20px rgba(0, 255, 128, 0.2);
        border: 1px solid rgba(0, 255, 128, 0.4);
    }

    /* Titles and Accents */
    .main-title {
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 5px;
        color: #ffffff;
        margin-top: -20px;
    }
    .sub-title {
        font-size: 16px;
        color: #888888;
        margin-bottom: 30px;
    }
    .metric-label {
        font-size: 12px;
        color: #a0a0a0;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 40px;
        font-weight: 700;
        color: #00ff80; /* Green Glow Text */
        text-shadow: 0 0 15px rgba(0, 255, 128, 0.4);
        line-height: 1.1;
    }
    .metric-value-medium {
        font-size: 28px;
        font-weight: 700;
        color: #00ff80; 
        text-shadow: 0 0 15px rgba(0, 255, 128, 0.4);
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 14px;
        color: #888888;
        margin-top: 5px;
    }

    /* Streamlit Expander styling to match theme */
    .streamlit-expanderHeader {
        background-color: #1a1a1c !important;
        color: #00ff80 !important;
        border-radius: 10px !important;
        border: 1px solid rgba(0, 255, 128, 0.15) !important;
        font-weight: 600 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- STATE MANAGEMENT ---
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None

# --- SCREEN 1: UPLOAD PAGE ---
if not st.session_state.data_loaded:
    st.write("") 
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
<div class='bento-card' style='text-align: center;'>
    <h2 style='color: #00ff80; margin-bottom: 10px;'>⚡ System Initialization</h2>
    <p style='color: #888888;'>Upload your SD Card <code>thesis_log.csv</code> to launch the dashboard.</p>
</div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("", type=['csv'])
        
        if uploaded_file is not None:
            with st.spinner("Analyzing data and calibrating AI..."):
                time.sleep(1.5) 
                
                df = pd.read_csv(uploaded_file, names=['Datetime', 'Gas', 'PM', 'Temp', 'Status'])
                df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
                
                alert_df = df[df['Status'].str.contains("DETECTED|EMERGENCY", na=False)].copy()
                
                if not alert_df.empty:
                    alert_df['Date'] = alert_df['Datetime'].dt.date
                    alert_df['Time'] = alert_df['Datetime'].dt.strftime('%I:%M %p')
                    alert_df['Hour'] = alert_df['Datetime'].dt.hour
                    alert_df['DayName'] = alert_df['Datetime'].dt.day_name()
                
                st.session_state.df = alert_df
                st.session_state.data_loaded = True
                st.rerun()

# --- SCREEN 2: MAIN DASHBOARD ---
else:
    df = st.session_state.df

    # Clean Header
    st.markdown("<div class='main-title'>⚡ AI Predictive Alerts Control Center</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Solar-Powered Smoke Detection System Analysis</div>", unsafe_allow_html=True)

    # --- CALCULATE METRICS ---
    total_incidents = len(df) if df is not None else 0
    if total_incidents > 0:
        peak_hour = df['Hour'].mode()[0]
        peak_time_str = f"{peak_hour % 12 or 12}:00 {'AM' if peak_hour < 12 else 'PM'}"
        peak_day = df['DayName'].mode()[0]
        peak_date = df['Date'].mode()[0].strftime("%B %d, %Y")
    else:
        peak_time_str, peak_day, peak_date = "N/A", "N/A", "N/A"

    # --- LAYOUT: ROW 1 (Metrics & Graph) ---
    col_metrics, col_graph = st.columns([1, 2.5])

    with col_metrics:
        # SINGLE-LINE HTML: This physically cannot break into raw text anymore.
        html_metrics = f"<div class='bento-card'><div class='metric-label'>🚨 Total Incidents</div><div class='metric-value'>{total_incidents}</div><hr style='border-color: #2a2a2c; margin: 20px 0;'><div class='metric-label'>🕒 Peak Violation Time</div><div class='metric-value-medium'>{peak_time_str}</div><hr style='border-color: #2a2a2c; margin: 20px 0;'><div class='metric-label'>📅 Highest Risk Day</div><div class='metric-value-medium'>{peak_day}</div><div class='metric-sub'>{peak_date}</div></div>"
        st.markdown(html_metrics, unsafe_allow_html=True)

    with col_graph:
        if total_incidents > 0:
            daily_counts = df.groupby('Date').size().reset_index(name='Incidents')
            daily_counts['Date'] = daily_counts['Date'].astype(str) 
            
            # Reverted back to Line Graph, with markers added so it doesn't break on single days
            fig = px.line(daily_counts, x='Date', y='Incidents', markers=True, title="📊 Incidents Over Time")
            fig.update_traces(line_color='#00ff80', line_width=4, marker=dict(size=10, color='#00ff80'))
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#ffffff'),
                title_font=dict(size=18, color='#00ff80'),
                xaxis=dict(showgrid=False, title=""),
                yaxis=dict(showgrid=True, gridcolor='#2a2a2c', title="Number of Incidents"),
                margin=dict(l=20, r=20, t=50, b=20),
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No incident data found in this log.")

    # --- LAYOUT: ROW 2 (Dropdown AI Calibration) ---
    st.markdown("<br>", unsafe_allow_html=True) 
    
    with st.expander("🤖 Show AI Calibration Routine Code"):
        st.markdown("""
            <div style='color: #888888; margin-bottom: 15px;'>
                <b>Instructions:</b> Copy the generated code block below and replace the existing logic in your <code>void loop()</code>. 
                This will optimize battery consumption and increase detection speed during calculated peak hours.
            </div>
        """, unsafe_allow_html=True)
        
        ai_code = f"""// --- PREDICTIVE ALERT OVERRIDE ---
// Based on current data, update the ESP32 schedule.
// Peak violations occur at {peak_time_str} on {peak_day}s.

void configurePredictiveAI() {{
    int peakHour = {df['Hour'].mode()[0] if total_incidents > 0 else 10}; 
    
    // Increase sensor polling rate during high-risk hours
    if (now.hour() == peakHour || now.hour() == peakHour - 1) {{
        samplingRate = 500; // Fast scan (0.5 seconds)
        Serial.println("AI STATUS: High-Risk Period Active. Scanning accelerated.");
    }} else {{
        samplingRate = 2000; // Normal scan (2 seconds) to save solar battery
    }}
}}"""
        st.code(ai_code, language='cpp')