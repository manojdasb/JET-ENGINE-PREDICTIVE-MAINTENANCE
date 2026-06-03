# engine_lstm_model.py
import os
import kagglehub
import numpy as np
import pandas as pd
import joblib
from scipy.fft import fft
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

np.random.seed(42)
tf.random.set_seed(42)

print("Ingesting data via KaggleHub...")
path = kagglehub.dataset_download("palbha/cmapss-jet-engine-simulated-data")

columns = ['engine_id', 'cycle', 'setting_1', 'setting_2', 'setting_3'] + [f'sensor_{i}' for i in range(1, 22)]
train_df = pd.read_csv(os.path.join(path, 'train_FD001.txt'), sep=r'\s+', header=None, names=columns)
test_df = pd.read_csv(os.path.join(path, 'test_FD001.txt'), sep=r'\s+', header=None, names=columns)
true_rul = pd.read_csv(os.path.join(path, 'RUL_FD001.txt'), header=None, names=['RUL'])

# Engineering targets
max_cycle = train_df.groupby('engine_id')['cycle'].max().reset_index()
max_cycle.columns = ['engine_id', 'max_cycle']
train_df = train_df.merge(max_cycle, on='engine_id', how='left')
train_df['RUL'] = (train_df['max_cycle'] - train_df['cycle']).clip(upper=125)
train_df.drop('max_cycle', axis=1, inplace=True)

drop_sensors = ['sensor_1', 'sensor_5', 'sensor_6', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']
features = [c for c in train_df.columns if c not in ['engine_id', 'cycle', 'RUL'] + drop_sensors]

def extract_signal_features(df, feature_cols, window_size=10):
    processed_dfs = []
    for engine_id, group in df.groupby('engine_id'):
        group = group.copy()
        for col in feature_cols:
            group[f'{col}_mean'] = group[col].rolling(window_size, min_periods=1).mean()
            group[f'{col}_var'] = group[col].rolling(window_size, min_periods=1).var().fillna(0)
            
            fft_magnitudes = []
            values = group[col].values
            for i in range(len(values)):
                start_idx = max(0, i - window_size + 1)
                window_data = values[start_idx:i+1]
                if len(window_data) > 1:
                    freq_data = np.abs(fft(window_data))
                    peak_freq = np.max(freq_data[1:]) if len(freq_data) > 1 else 0
                else:
                    peak_freq = 0
                fft_magnitudes.append(peak_freq)
            group[f'{col}_fft_peak'] = fft_magnitudes
        processed_dfs.append(group)
    return pd.concat(processed_dfs, ignore_index=True)

print("Processing feature signals (Rolling window + FFT math)...")
train_df = extract_signal_features(train_df, features)
all_features = [c for c in train_df.columns if c not in ['engine_id', 'cycle', 'RUL']]

# Scale data
scaler = MinMaxScaler()
train_df[all_features] = scaler.fit_transform(train_df[all_features])
joblib.dump(scaler, "scaler.pkl")

# Generate sequence matrices for LSTM training
sequence_length = 30
X_train, y_train = [], []
for engine_id, group in train_df.groupby('engine_id'):
    data = group[all_features].values
    labels = group['RUL'].values
    for i in range(len(data) - sequence_length):
        X_train.append(data[i:i + sequence_length])
        y_train.append(labels[i + sequence_length])

X_train, y_train = np.array(X_train).astype(np.float32), np.array(y_train).astype(np.float32)

print("Compiling network layers...")
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

print("Optimizing model parameters...")
model.fit(X_train, y_train, epochs=15, batch_size=256, validation_split=0.1, verbose=1)

# Export assets
model.save("lstm_predictive_maintenance_model.keras")
# Also save a baseline copy of test data for streaming simulation convenience
test_df.to_csv("simulated_test_bank.csv", index=False)
print("Pipeline complete. Production binaries successfully generated!")