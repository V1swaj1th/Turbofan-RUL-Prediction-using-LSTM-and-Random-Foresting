import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
import sys

# Reconfigure standard output encoding to UTF-8 to avoid encoding issues
sys.stdout.reconfigure(encoding='utf-8')

# Load the datasets
train_data_path = r'C:\Users\mayan\OneDrive\Desktop\Mini-project\train_FD004.txt'
test_data_path = r'C:\Users\mayan\OneDrive\Desktop\Mini-project\test_FD004.txt'
rul_data_path = r'C:\Users\mayan\OneDrive\Desktop\Mini-project\RUL_FD004.txt'

# Load data into pandas DataFrames with UTF-8 encoding
train_df = pd.read_csv(train_data_path, delim_whitespace=True, header=None, encoding='utf-8')
test_df = pd.read_csv(test_data_path, delim_whitespace=True, header=None, encoding='utf-8')
rul_df = pd.read_csv(rul_data_path, header=None, encoding='utf-8')

# Renaming columns for clarity
cols = ['unit_number', 'time_in_cycles'] + [f'operational_setting_{i+1}' for i in range(3)] + [f'sensor_measurement_{i+1}' for i in range(21)]
train_df.columns = cols
test_df.columns = cols

# Generate RUL values for train data
max_cycles = train_df.groupby('unit_number')['time_in_cycles'].max()
train_df['RUL'] = train_df.apply(lambda row: max_cycles[row['unit_number']] - row['time_in_cycles'], axis=1)

# Feature Selection: Correlation-based feature selection
correlation_matrix = train_df.corr()
# Select only features that have a high correlation with RUL
important_features = correlation_matrix['RUL'].sort_values(ascending=False)
selected_features = important_features[important_features.abs() > 0.1].index.tolist()
selected_features.remove('RUL')  # Remove the target itself

# Prepare data for LSTM model
X_train = train_df[selected_features].values  # Features
y_train = train_df['RUL'].values  # Target (RUL)

# Scaling the features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Reshaping the data for LSTM
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))

# LSTM Model with Adjusted Architecture
model = Sequential([
    LSTM(128, activation='relu', input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]), return_sequences=True),
    Dropout(0.3),
    LSTM(64, activation='relu', return_sequences=False),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train_reshaped, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Prepare test data
X_test = test_df[selected_features].values  # Features for test data
X_test_scaled = scaler.transform(X_test)  # Scale test features

# Reshaping the test data for LSTM
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Predict RUL for the test dataset
predicted_rul = model.predict(X_test_reshaped)

# Calculate RMSE for the test dataset
actual_rul = rul_df.values.flatten()
test_rmse = np.sqrt(mean_squared_error(actual_rul, predicted_rul.flatten()[:len(actual_rul)]))

print(f"Test RMSE: {test_rmse}")
