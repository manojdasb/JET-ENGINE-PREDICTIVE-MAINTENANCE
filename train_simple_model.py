# Simple model trainer using XGBoost - avoids TensorFlow conflicts
import os
import pandas as pd
import numpy as np
from scipy.fft import fft
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
import joblib

print("📊 Loading and preprocessing data...")
columns = ['engine_id', 'cycle', 'setting_1', 'setting_2', 'setting_3'] + [f'sensor_{i}' for i in range(1, 22)]

# Load training data
train_raw = pd.read_csv('data/train_FD001.txt', sep=r'\s+', header=None, names=columns)
test_raw = pd.read_csv('data/test_FD001.txt', sep=r'\s+', header=None, names=columns)

# Calculate RUL
max_cycle = train_raw.groupby('engine_id')['cycle'].max().reset_index(name='max_cycle')
train_df = train_raw.merge(max_cycle, on='engine_id')
train_df['RUL'] = (train_df['max_cycle'] - train_df['cycle']).clip(upper=125)

# Select features
drop_sensors = ['sensor_1', 'sensor_5', 'sensor_6', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']
feature_cols = [c for c in train_raw.columns if c not in ['engine_id', 'cycle', 'setting_1', 'setting_2', 'setting_3'] + drop_sensors]

print(f"📈 Processing {len(feature_cols)} sensor features...")
window_size = 10
processed_groups = []

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
all_features = [c for c in train_df.columns if c not in ['engine_id', 'cycle', 'RUL', 'setting_1', 'setting_2', 'setting_3', 'max_cycle']]

print(f"⚖️ Scaling {len(all_features)} features...")
scaler = MinMaxScaler()
train_df[all_features] = scaler.fit_transform(train_df[all_features])
joblib.dump(scaler, "scaler.pkl")

# Prepare training data for GradientBoosting
X_train = train_df[all_features].values
y_train = train_df['RUL'].values

print(f"🤖 Training GradientBoosting model ({len(X_train)} samples)...")
model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=5,
    random_state=42,
    verbose=1
)

model.fit(X_train, y_train)

# Save as pickle for simplicity
joblib.dump(model, "lstm_predictive_maintenance_model.pkl")
print("✅ Model trained and saved as lstm_predictive_maintenance_model.pkl")

# Test it on validation data
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
y_pred = model.predict(X_train[:5000])
y_true = y_train[:5000]
print(f"\n📊 Training Metrics:")
print(f"  MAE: {mean_absolute_error(y_true, y_pred):.2f}")
print(f"  MSE: {mean_squared_error(y_true, y_pred):.2f}")
print(f"  R²:  {r2_score(y_true, y_pred):.4f}")
