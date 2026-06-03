# train.py
import os
import shutil
import glob
import numpy as np
import pandas as pd
import joblib
from scipy.fft import fft
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Check if data files exist, if not try to download
print("📥 Step 1: Checking data files...")
if not os.path.exists("data/train_FD001.txt"):
    print("Attempting to download NASA CMAPSS files via KaggleHub...")
    try:
        import kagglehub
        download_path = kagglehub.dataset_download("palbha/cmapss-jet-engine-simulated-data")
        print(f"📦 Dataset downloaded to cache: {download_path}")
        os.makedirs("data", exist_ok=True)
        for file_path in glob.glob(os.path.join(download_path, "*")):
            if os.path.isfile(file_path):
                shutil.copy(file_path, "data/")
        print("✅ Files localized successfully to your local './data/' directory.")
    except Exception as e:
        print(f"⚠️  KaggleHub download failed: {e}")
        print("Please ensure data files are in ./data/ directory and rerun this script.")
        import sys
        sys.exit(1)
else:
    print("✅ Data files found in data/ directory")

print("📊 Step 2: Structuring streaming asset data matrices...")
columns = ['engine_id', 'cycle', 'setting_1', 'setting_2', 'setting_3'] + [f'sensor_{i}' for i in range(1, 22)]
train_raw = pd.read_csv('data/train_FD001.txt', sep=r'\s+', header=None, names=columns)
test_raw = pd.read_csv('data/test_FD001.txt', sep=r'\s+', header=None, names=columns)

# Cache the test bank for your live telemetry system to read from
test_raw.to_csv("simulated_test_bank.csv", index=False)

# Calculate Remaining Useful Life (RUL) with a standard upper bound clip
max_cycle = train_raw.groupby('engine_id')['cycle'].max().reset_index(name='max_cycle')
train_df = train_raw.merge(max_cycle, on='engine_id')
train_df['RUL'] = (train_df['max_cycle'] - train_df['cycle']).clip(upper=125)

drop_sensors = ['sensor_1', 'sensor_5', 'sensor_6', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']
feature_cols = [c for c in train_raw.columns if c not in ['engine_id', 'cycle', 'setting_1', 'setting_2', 'setting_3'] + drop_sensors]

print("🧮 Step 3: Running Signal Processing Engine (Rolling Statistics & FFT)...")
processed_groups = []
window_size = 10
for engine_id, group in train_df.groupby('engine_id'):
    group = group.copy().sort_values('cycle')
    for col in feature_cols:
        group[f'{col}_mean'] = group[col].rolling(window_size, min_periods=1).mean()
        group[f'{col}_var'] = group[col].rolling(window_size, min_periods=1).var().fillna(0.0)
        
        raw_values = group[col].values
        fft_peaks = np.zeros(len(raw_values))
        for i in range(len(raw_values)):
            window = raw_values[max(0, i - window_size + 1):i+1]
            if len(window) > 1:
                mags = np.abs(fft(window))
                fft_peaks[i] = np.max(mags[1:]) if len(mags) > 1 else 0.0
        group[f'{col}_fft_peak'] = fft_peaks
    processed_groups.append(group)

train_df = pd.concat(processed_groups, ignore_index=True)
all_features = [c for c in train_df.columns if c not in ['engine_id', 'cycle', 'RUL', 'setting_1', 'setting_2', 'setting_3']]

print("⚖️ Step 4: Scaling time-series feature spaces...")
scaler = MinMaxScaler()
train_df[all_features] = scaler.fit_transform(train_df[all_features])
joblib.dump(scaler, "scaler.pkl")

print("🧠 Step 5: Building temporal sequence matrices for LSTM layers...")
sequence_length = 30
X_train, y_train = [], []
for engine_id, group in train_df.groupby('engine_id'):
    data = group[all_features].values
    labels = group['RUL'].values
    for i in range(len(data) - sequence_length):
        X_train.append(data[i:i + sequence_length])
        y_train.append(labels[i + sequence_length])

X_train = np.array(X_train).astype(np.float32)
y_train = np.array(y_train).astype(np.float32)

print("🤖 Step 6: Optimizing LSTM Network Architecture...")
model = Sequential([
    LSTM(48, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(24),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=30, batch_size=256, validation_split=0.1, verbose=1)

# Export the trained binaries
model.save("lstm_predictive_maintenance_model.keras")
print("\n🚀 SUCCESS: 'scaler.pkl' and 'lstm_predictive_maintenance_model.keras' are generated!")