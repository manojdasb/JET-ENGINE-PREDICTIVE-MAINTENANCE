# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import matplotlib.pyplot as plt
from scipy.fft import fft
import tensorflow as tf
from telemetry_generator import TelemetryStreamer

st.set_page_config(page_title="IoT Maintenance Dashboard", page_icon="⚙️", layout="wide")

st.title("⚙️ Real-Time Jet Engine Predictive Maintenance Control Center")
st.markdown("This control room consumes a live data stream from an operating jet engine asset and uses an LSTM model to predict structural failure metrics on the fly.")

# Load machine learning binaries
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("lstm_predictive_maintenance_model.keras")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_assets()
    assets_ready = True
except:
    st.error("🚨 Configuration Error: Model and Scaler binaries not found. Run 'python engine_lstm_model.py' to generate artifacts first.")
    assets_ready = False

if assets_ready:
    # Sidebar control panel
    st.sidebar.header("Asset Selection")
    engine_id = st.sidebar.number_input("Select Engine Unit ID to Monitor:", min_value=1, max_value=100, value=1)
    stream_speed = st.sidebar.slider("Stream Interval Delay (Seconds):", min_value=0.1, max_value=2.0, value=0.4)
    
    start_stream = st.sidebar.button("🔌 Initialize Live Telemetry Stream")

    # Setup core session state variables to store the incoming live data streams
    if "telemetry_history" not in st.session_state or st.sidebar.button("Reset Dashboard Data"):
        st.session_state.telemetry_history = pd.DataFrame()
        st.session_state.current_cycle = 0
        st.session_state.predicted_rul = 125
        st.session_state.status = "INITIALIZING"

    # Layout placeholder structures
    kpi_space = st.empty()
    chart_space = st.empty()

    if start_stream:
        streamer = TelemetryStreamer()
        live_generator = streamer.get_engine_stream(engine_id)
        
        # Operational parameters used for feature engineering
        drop_sensors = ['sensor_1', 'sensor_5', 'sensor_6', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']
        base_features = [f'sensor_{i}' for i in range(1, 22) if f'sensor_{i}' not in drop_sensors]
        
        # Real-time processing loop
        for data_packet in live_generator:
            # 1. Append the new real-time packet to memory history
            new_row = pd.DataFrame([data_packet])
            st.session_state.telemetry_history = pd.concat([st.session_state.telemetry_history, new_row], ignore_index=True)
            current_len = len(st.session_state.telemetry_history)
            
            # 2. Compute live math metrics on the fly (Rolling windows and FFT magnitudes)
            window_size = 10
            history_slice = st.session_state.telemetry_history.copy()
            
            for col in base_features:
                # Compute rolling parameters
                history_slice[f'{col}_mean'] = history_slice[col].rolling(window_size, min_periods=1).mean()
                history_slice[f'{col}_var'] = history_slice[col].rolling(window_size, min_periods=1).var().fillna(0)
                
                # Dynamic FFT Peak tracking
                fft_peaks = []
                for idx in range(len(history_slice)):
                    start_i = max(0, idx - window_size + 1)
                    w_data = history_slice[col].values[start_i:idx+1]
                    if len(w_data) > 1:
                        freqs = np.abs(fft(w_data))
                        peak = np.max(freqs[1:]) if len(freqs) > 1 else 0
                    else:
                        peak = 0
                    fft_peaks.append(peak)
                history_slice[f'{col}_fft_peak'] = fft_peaks

            # Isolate all feature names used by model configuration
            all_cols = [c for c in history_slice.columns if c not in ['engine_id', 'cycle', 'setting_1', 'setting_2', 'setting_3']]
            
            # 3. Apply LSTM evaluation sequence shapes
            sequence_length = 30
            if current_len >= sequence_length:
                # Scale the real-time processed stream data
                scaled_features = history_slice[all_cols].copy()
                scaled_features[all_cols] = scaler.transform(scaled_features[all_cols])
                
                # Capture the last window of data and expand dimensions for the LSTM input layer
                lstm_input_window = scaled_features.values[-sequence_length:]
                lstm_input_tensor = np.expand_dims(lstm_input_window, axis=0)
                
                # Generate inference prediction
                raw_pred = model.predict(lstm_input_tensor, verbose=0)[0][0]
                st.session_state.predicted_rul = max(0, int(raw_pred))
            else:
                st.session_state.predicted_rul = "Buffering Context Window..."
                
            st.session_state.current_cycle = int(data_packet['cycle'])
            
            # 4. Status Evaluation logic
            if isinstance(st.session_state.predicted_rul, str):
                st.session_state.status = "🔄 BUFFERING TIME-SERIES"
                status_color = "gray"
            elif st.session_state.predicted_rul <= 25:
                st.session_state.status = "🚨 CRITICAL FAILURE IMMINENT"
                status_color = "red"
            elif st.session_state.predicted_rul <= 75:
                st.session_state.status = "⚠️ MAINTENANCE REQUIRED"
                status_color = "orange"
            else:
                st.session_state.status = "✅ OPERATION NOMINAL"
                status_color = "green"

            # 5. Redraw KPI Cards metric layouts dynamically
            with kpi_space.container():
                col1, col2, col3 = st.columns(3)
                col1.metric("Current Streamed Flight Cycle", f"{st.session_state.current_cycle} Cycles")
                col2.metric("LSTM Predicted Remaining Lifespan", f"{st.session_state.predicted_rul} Cycles")
                col3.markdown(f"""
                    <div style="background-color:{status_color}; padding:20px; border-radius:8px; text-align:center;">
                        <span style="color:white; font-weight:bold; font-size:16px;">{st.session_state.status}</span>
                    </div>
                """, unsafe_html=True)

            # 6. Redraw analytical plots dynamically
            with chart_space.container():
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4.5))
                
                # Left Plot: Live Telemetry Filtering Showcase
                ax1.plot(history_slice['cycle'], history_slice['sensor_11'], color='gray', alpha=0.4, label='Raw Live Input Data')
                ax1.plot(history_slice['cycle'], history_slice['sensor_11_mean'], color='blue', linewidth=2, label='Processed Rolling Mean')
                ax1.set_title("Edge Telemetry Processing Node (Sensor 11 Example)")
                ax1.set_xlabel("Flight Cycles")
                ax1.set_ylabel("Amplitude")
                ax1.legend()
                ax1.grid(True, linestyle=":", alpha=0.5)
                
                # Right Plot: Live RUL Decay Curve Visualization
                ax2.axhline(y=25, color='red', linestyle='--', alpha=0.7, label='Safety Margin Threshold')
                if not isinstance(st.session_state.predicted_rul, str):
                    ax2.scatter(st.session_state.current_cycle, st.session_state.predicted_rul, color='purple', s=100, zorder=5, label='Current Forecast')
                ax2.set_title("Asset Remaining Useful Life (RUL) Trajectory")
                ax2.set_xlabel("Flight Cycles")
                ax2.set_ylabel("Forecasted Lifespan Cycles")
                ax2.set_ylim(-5, 140)
                ax2.legend()
                ax2.grid(True, linestyle=":", alpha=0.5)
                
                st.pyplot(fig)
                plt.close()
                
            # Introduce real-time streaming cadence delay simulation
            time.sleep(stream_speed)