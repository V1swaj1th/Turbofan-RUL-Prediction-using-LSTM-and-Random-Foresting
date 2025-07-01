import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Force UTF-8 encoding for logs and outputs
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Log RMSE results in a way that avoids UnicodeEncodeError
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Load training and test data
train_df = pd.read_csv(r"C:\Users\mayan\OneDrive\Desktop\Mini-project\train_FD004.txt", sep='\s+', header=None)
test_df = pd.read_csv(r"C:\Users\mayan\OneDrive\Desktop\Mini-project\test_FD004.txt", sep='\s+', header=None)
rul_df = pd.read_csv(r"C:\Users\mayan\OneDrive\Desktop\Mini-project\RUL_FD004.txt", header=None)

# Assign column names
column_names = ["unit_number", "time_in_cycles", "operational_setting_1", "operational_setting_2", "operational_setting_3"] + \
               ["sensor_measurement_" + str(i) for i in range(1, 22)]
train_df.columns = column_names
test_df.columns = column_names

# Add RUL to training data
train_df['rul'] = 100 - train_df.groupby('unit_number')['time_in_cycles'].rank(ascending=False).astype(int)

# --- Feature Engineering ---
feature_columns = ["operational_setting_1", "operational_setting_2", "operational_setting_3"] + \
                  ["sensor_measurement_" + str(i) for i in range(1, 22)]
X_train = train_df[feature_columns]
y_train = train_df["rul"]

# Scale features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

# --- Random Forest Implementation ---
X_train_rf, X_val_rf, y_train_rf, y_val_rf = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train_rf, y_train_rf)

y_val_rf_pred = rf_model.predict(X_val_rf)
rmse_rf = np.sqrt(mean_squared_error(y_val_rf, y_val_rf_pred))
print(f"Random Forest Validation RMSE: {rmse_rf}", flush=True)

# --- LSTM Implementation ---
time_steps = 30
units = train_df["unit_number"].unique()
X_lstm, y_lstm = [], []

for unit in units:
    unit_data = train_df[train_df["unit_number"] == unit]
    features_unit = scaler.transform(unit_data[feature_columns])
    target_unit = unit_data["rul"].values

    for i in range(len(unit_data) - time_steps):
        X_lstm.append(features_unit[i:i + time_steps])
        y_lstm.append(target_unit[i + time_steps])

X_lstm = np.array(X_lstm)
y_lstm = np.array(y_lstm)

X_train_lstm, X_val_lstm, y_train_lstm, y_val_lstm = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)

# LSTM model with Input layer
lstm_model = Sequential([
    tf.keras.Input(shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
    LSTM(64, activation='relu', return_sequences=True),
    Dropout(0.2),
    LSTM(32, activation='relu', return_sequences=False),
    Dropout(0.2),
    Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train LSTM
history = lstm_model.fit(X_train_lstm, y_train_lstm, epochs=50, batch_size=64, validation_data=(X_val_lstm, y_val_lstm))

# Validate LSTM
y_val_lstm_pred = lstm_model.predict(X_val_lstm).flatten()
rmse_lstm = np.sqrt(mean_squared_error(y_val_lstm, y_val_lstm_pred))
print(f"LSTM Validation RMSE: {rmse_lstm}", flush=True)

# --- Visualization: True vs Predicted for Validation ---
plt.figure(figsize=(10, 6))
plt.scatter(y_val_rf, y_val_rf_pred, alpha=0.5, label="Random Forest")
plt.scatter(y_val_lstm, y_val_lstm_pred, alpha=0.5, label="LSTM")
plt.plot([0, 100], [0, 100], 'r--', label="Perfect Prediction")
plt.xlabel("True RUL")
plt.ylabel("Predicted RUL")
plt.title("True vs Predicted RUL (Validation)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Test Dataset Prediction ---
true_rul = rul_df.values.flatten()

# Prepare the test dataset for LSTM predictions
X_test_lstm = []

test_units = test_df["unit_number"].unique()

for unit in test_units:
    unit_data = test_df[test_df["unit_number"] == unit]
    features_unit = scaler.transform(unit_data[feature_columns])

    for i in range(len(unit_data) - time_steps):
        X_test_lstm.append(features_unit[i:i + time_steps])

X_test_lstm = np.array(X_test_lstm)

# Predict RUL for the test set
y_test_lstm_pred = lstm_model.predict(X_test_lstm).flatten()

# Aggregate predictions per unit (last prediction for each unit)
unit_predictions = []
start_index = 0

for unit in test_units:
    unit_data = test_df[test_df["unit_number"] == unit]
    end_index = start_index + len(unit_data) - time_steps
    unit_predictions.append(y_test_lstm_pred[end_index - 1])  # Last prediction for the unit
    start_index = end_index

# Convert to numpy array for plotting
unit_predictions = np.array(unit_predictions)

# --- Visualization: True vs Predicted for Test ---
plt.figure(figsize=(10, 6))
plt.plot(true_rul, label="True RUL", marker='o', linestyle='--')
plt.plot(unit_predictions, label="Predicted RUL (LSTM)", marker='x', linestyle='-')
plt.xlabel("Unit Number")
plt.ylabel("RUL")
plt.title("True RUL vs Predicted RUL (LSTM)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
