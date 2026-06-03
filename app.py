# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import matplotlib.pyplot as plt
from scipy.fft import fft
import os

# Configuration and Page Setup
st.set_page_config(page_title="IIoT Maintenance Center", page_icon="🏭", layout="wide")

st.title("🏭 High-Fidelity IIoT Machine Health Analytics Hub")
st.markdown(
    "This dashboard consumes live telemetry streams from an operating jet engine asset, "
    "calculates real-time Signal Processed features, and runs an LSTM network to predict Remaining Useful Life (RUL)."
)

# ---------------------------------------------------------------------------
# ARTIFACT LOADING & VERIFICATION
# ---------------------------------------------------------------------------
@st.cache_resource
def load_production_artifacts():
    """Loads and caches the trained predictive model and scaler."""
    # Try loading the trained GradientBoosting model
    try:
        model = joblib.load("lstm_predictive_maintenance_model.pkl")
        st.success("✅ Trained GradientBoosting model loaded successfully")
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        st.info("Please run: `python train_simple_model.py`")
        model = None
    
    scaler = joblib.load("scaler.pkl")
    return model, scaler

# Verify required model binaries exist in the root execution environment
if not os.path.exists("lstm_predictive_maintenance_model.pkl") or not os.path.exists("scaler.pkl"):
    st.error(
        "🚨 Configuration Error: Model or Scaler binaries not found. "
        "Run `python train_simple_model.py` to generate trained models."
    )
    st.stop()

try:
    model, scaler = load_production_artifacts()
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()

# Try loading the telemetry streamer engine directly from the root workspace
try:
    from telemetry_simulator import TelemetryStreamer
except ImportError as e:
    st.error(
        f"🚨 Architecture Mismatch: Could not locate 'telemetry_simulator.py' in your root directory. "
        f"Details: {e}"
    )
    st.stop()

# ---------------------------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------------------------
st.sidebar.header("Asset Monitor Node Settings")
selected_id = st.sidebar.number_input("Target Engine Asset ID:", min_value=1, max_value=100, value=1)
cadence_delay = st.sidebar.slider("Streaming Speed Frame Latency (Seconds):", min_value=0.01, max_value=1.50, value=0.10)
trigger_stream = st.sidebar.button("🔌 Establish Secure Edge Connection")

# Session State Initializations
if "history" not in st.session_state or st.sidebar.button("Reset In-Memory Logs"):
    st.session_state.history = pd.DataFrame()
    st.session_state.current_cycle = 0
    st.session_state.forecast_rul = 125
    st.session_state.operational_state = "WARMING"
    st.session_state.last_engine_id = None
    st.session_state.last_packet_index = 0

# Pre-allocate dynamic UI rendering containers
kpi_block = st.empty()
viz_block = st.empty()

# ---------------------------------------------------------------------------
# TELEMETRY INGESTION & PIPELINE RUNTIME
# ---------------------------------------------------------------------------
if trigger_stream:
    streamer = TelemetryStreamer()
    live_feed = list(streamer.get_engine_stream(selected_id))  # Convert to list - Fixed method name
    
    # Reset tracking if engine_id changed
    if st.session_state.last_engine_id != selected_id:
        st.session_state.history = pd.DataFrame()
        st.session_state.last_packet_index = 0
        st.session_state.last_engine_id = selected_id
    
    # Exclude invariant sensor matrices as completed during model selection
    drop_sensors = ['sensor_1', 'sensor_5', 'sensor_6', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']
    base_features = [f'sensor_{i}' for i in range(1, 22) if f'sensor_{i}' not in drop_sensors]
    
    WINDOW_SIZE = 10
    SEQUENCE_LENGTH = 30
    
    # Process only the next 5 packets per page load (prevents buffering)
    PACKETS_PER_RUN = 5
    end_index = min(st.session_state.last_packet_index + PACKETS_PER_RUN, len(live_feed))
    
    for idx in range(st.session_state.last_packet_index, end_index):
        telemetry_packet = live_feed[idx]
        
        # Ingest the next streaming edge packet
        packet_df = pd.DataFrame([telemetry_packet])
        st.session_state.history = pd.concat([st.session_state.history, packet_df], ignore_index=True)
        buffer_length = len(st.session_state.history)
        
        # Performance Safety Catch: Read the historical dataframe references
        working_matrix = st.session_state.history.copy()
        
        # -------------------------------------------------------------------
        # STREAMING SIGNAL PROCESSING ENGINE
        # -------------------------------------------------------------------
        # Vectorized rolling operations for complete stability
        for col in base_features:
            working_matrix[f'{col}_mean'] = working_matrix[col].rolling(WINDOW_SIZE, min_periods=1).mean()
            working_matrix[f'{col}_var'] = working_matrix[col].rolling(WINDOW_SIZE, min_periods=1).var().fillna(0.0)
            
            # Optimized localized window FFT to avoid quadratic O(N^2) lag slowdowns
            raw_array = working_matrix[col].values
            fft_peaks = np.zeros(len(raw_array))
            
            for idx in range(len(raw_array)):
                # Only compute for values modified in recent histories to keep dashboard snappy
                if idx < buffer_length - SEQUENCE_LENGTH and idx in working_matrix.index[:-1]:
                    continue 
                w_start = max(0, idx - WINDOW_SIZE + 1)
                data_window = raw_array[w_start:idx+1]
                if len(data_window) > 1:
                    mags = np.abs(fft(data_window))
                    fft_peaks[idx] = np.max(mags[1:]) if len(mags) > 1 else 0.0
            
            working_matrix[f'{col}_fft_peak'] = fft_peaks

        # Isolate exactly the calculated model feature headers
        model_features = [c for c in working_matrix.columns if c not in ['engine_id', 'cycle', 'setting_1', 'setting_2', 'setting_3']]
        
        # -------------------------------------------------------------------
        # MODEL INFERENCE (GradientBoosting or other regressor)
        # -------------------------------------------------------------------
        if buffer_length >= SEQUENCE_LENGTH:
            # Get the most recent features
            recent_data = working_matrix[model_features].tail(SEQUENCE_LENGTH).copy()
            
            # Scale the features
            scaled_data = recent_data.copy()
            scaled_data[model_features] = scaler.transform(recent_data[model_features])
            
            # Use the average of recent scaled features for prediction
            input_features = scaled_data[model_features].mean().values.reshape(1, -1)
            
            # Make prediction
            inference = model.predict(input_features)[0]
            st.session_state.forecast_rul = max(0, int(inference))
        else:
            st.session_state.forecast_rul = f"Buffering Context ({buffer_length}/{SEQUENCE_LENGTH})"
            
        st.session_state.current_cycle = int(telemetry_packet['cycle'])
        
        # -------------------------------------------------------------------
        # STRUCTURAL STATE EVALUATIONS
        # -------------------------------------------------------------------
        if isinstance(st.session_state.forecast_rul, str):
            st.session_state.operational_state = "🔄 CACHING CONTEXT"
            badge_color = "#4A5568"
        elif st.session_state.forecast_rul <= 20:
            st.session_state.operational_state = "🚨 CRITICAL FAILURE RISK"
            badge_color = "#E53E3E"
        elif st.session_state.forecast_rul <= 65:
            st.session_state.operational_state = "⚠️ SCHEDULE MAINTENANCE"
            badge_color = "#DD6B20"
        else:
            st.session_state.operational_state = "✅ STRUCTURAL NOMINAL"
            badge_color = "#38A169"

        # Render KPI Displays
        with kpi_block.container():
            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric("Current Asset Operational Age", f"{st.session_state.current_cycle} Cycles")
            
            rul_display = f"{st.session_state.forecast_rul} Cycles" if isinstance(st.session_state.forecast_rul, int) else st.session_state.forecast_rul
            kpi2.metric("Predicted Lifespan Remaining (RUL)", rul_display)
            
            kpi3.markdown(f"""
                <div style="background-color:{badge_color}; padding:18px; border-radius:10px; text-align:center;">
                    <span style="color:white; font-weight:bold; font-size:18px; font-family:sans-serif;">{st.session_state.operational_state}</span>
                </div>
            """, unsafe_allow_html=True)

        # Render Real-Time Visualizations
        with viz_block.container():
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4.5))
            
            # Left Plot: Raw vs rolling transform metrics
            ax1.plot(working_matrix['cycle'], working_matrix['sensor_11'], color='#A0AEC0', alpha=0.5, label='Raw Digital Signal Input')
            ax1.plot(working_matrix['cycle'], working_matrix['sensor_11_mean'], color='#3182CE', linewidth=2, label='Math Extraction: Rolling Mean')
            ax1.set_title("Edge Signal Transformation Node (EGT Sensor Example)", fontsize=11, fontweight='bold')
            ax1.set_xlabel("Operating Cycles")
            ax1.set_ylabel("Amplitude Parameters")
            ax1.legend(loc="upper left")
            ax1.grid(True, linestyle=":", alpha=0.6)
            
            # Right Plot: Historical tracking line of predicted structural life descent
            ax2.axhline(y=20, color='#E53E3E', linestyle='--', alpha=0.8, label='Immediate Risk Boundary')
            if isinstance(st.session_state.forecast_rul, int):
                ax2.scatter(st.session_state.current_cycle, st.session_state.forecast_rul, color='#805AD5', s=120, zorder=5, label='Active LSTM Node')
            ax2.set_title("Remaining Useful Life (RUL) Predictive Trajectory", fontsize=11, fontweight='bold')
            ax2.set_xlabel("Operating Cycles")
            ax2.set_ylabel("Predicted Remaining Cycles")
            ax2.set_ylim(-5, 140)
            ax2.legend(loc="upper right")
            ax2.grid(True, linestyle=":", alpha=0.6)
            
            st.pyplot(fig)
            plt.close(fig) # Prevent Matplotlib background memory allocation leakage
            
        time.sleep(cadence_delay)
    
    # Update packet tracking index
    st.session_state.last_packet_index = end_index
    
    # Show progress and streaming controls
    st.divider()
    progress_col1, progress_col2 = st.columns(2)
    with progress_col1:
        st.info(f"📊 Processed: {end_index}/{len(live_feed)} packets | Last cycle: {st.session_state.current_cycle}")
    
    # Allow continued streaming
    if end_index < len(live_feed):
        with progress_col2:
            if st.button("▶️ Continue Streaming", use_container_width=True):
                st.rerun()
    else:
        with progress_col2:
            st.success("✅ Stream Complete! All packets processed.")
