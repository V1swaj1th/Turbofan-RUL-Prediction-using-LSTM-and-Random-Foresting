# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load training and test data (update paths as necessary)
train_df = pd.read_csv(r"C:\Users\mayan\OneDrive\Desktop\Mini-project\train_FD004.txt", delim_whitespace=True, header=None)
test_df = pd.read_csv(r"C:\Users\mayan\OneDrive\Desktop\Mini-project\test_FD004.txt", delim_whitespace=True, header=None)
rul_df = pd.read_csv(r"C:\Users\mayan\OneDrive\Desktop\Mini-project\RUL_FD004.txt", header=None)  # RUL file

column_names = ["unit_number", "time_in_cycles", "operational_setting_1", "operational_setting_2", "operational_setting_3"] + \
               ["sensor_measurement_" + str(i) for i in range(1, 22)]
train_df.columns = column_names
test_df.columns = column_names

test_df

# %%
import matplotlib.pyplot as plt

# Filter the test dataset to include only the first 289 entries
test_df_filtered = test_df.head(248)

# Generate separate plots for each sensor
num_sensors = 21
fig, axes = plt.subplots(nrows=num_sensors, ncols=1, figsize=(10, num_sensors * 2))
fig.tight_layout(pad=2.5)

for i in range(1, num_sensors + 1):
    sensor_column = f"sensor_measurement_{i}"
    axes[i - 1].plot(test_df_filtered["time_in_cycles"], test_df_filtered[sensor_column], color='b')
    axes[i - 1].set_title(f"Sensor {i} Measurements over Time (First 248 Entries)")
    axes[i - 1].set_xlabel("Time in Cycles")
    axes[i - 1].set_ylabel("Sensor Measurement")

plt.show()


# %%
rul_df

# %%
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Select the first 248 entries for the test dataset and the RUL dataset
X = test_df.head(248).iloc[:, 2:]  # Exclude "unit_number" and "time_in_cycles" columns
y = rul_df.iloc[:248, 0]  # First 248 entries of RUL as target

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Calculate performance metrics
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

# Print the results
print("Mean Squared Error:", mse)
print("R-squared Score:", r2)


# %%
plt.scatter(train_df['time_in_cycles'][:], train_df['sensor_measurement_1'][:])

# %%
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Select relevant features: Use multiple sensors instead of just one
features = train_df[["time_in_cycles"] + ["sensor_measurement_" + str(i) for i in range(1, 22)]]

# Scale the features so that they are on the same scale
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Apply KMeans to create 6 operating conditions (clusters)
kmeans = KMeans(n_clusters=6, random_state=0)
train_df["condition"] = kmeans.fit_predict(features_scaled)

# Set up the scatter plot for time_in_cycles vs sensor_measurement_1, color-coded by condition
plt.figure(figsize=(12, 8))
sns.set_palette("Set1", 6)  # Use a 6-color palette

# Scatter plot for all data points, color-coded by condition
scatter = plt.scatter(train_df["time_in_cycles"], train_df["sensor_measurement_1"], c=train_df["condition"], cmap="Set1", s=50, edgecolor='k')

# Adding labels and title
plt.xlabel("Time in Cycles")
plt.ylabel("Sensor Measurement 1")
plt.title("Sensor Measurement 1 vs Time for 6 Operating Conditions")
plt.colorbar(scatter, label="Operating Condition")  # Show color legend for operating conditions
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()


# %%
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Select relevant features: Use multiple sensors and time_in_cycles
features = train_df[["time_in_cycles"] + ["sensor_measurement_" + str(i) for i in range(1, 22)]]

# Scale the features so that they are on the same scale
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Apply KMeans to create 6 operating conditions (clusters)
kmeans = KMeans(n_clusters=6, random_state=0)
train_df["condition"] = kmeans.fit_predict(features_scaled)

# Set up the scatter plot for time_in_cycles vs sensor_measurement_2, color-coded by condition
plt.figure(figsize=(12, 8))
sns.set_palette("Set1", 6)  # Use a 6-color palette

# Scatter plot for all data points, color-coded by condition
scatter = plt.scatter(train_df["time_in_cycles"], train_df["sensor_measurement_2"], c=train_df["condition"], cmap="Set1", s=50, edgecolor='k')

# Adding labels and title
plt.xlabel("Time in Cycles")
plt.ylabel("Sensor Measurement 2")
plt.title("Sensor Measurement 2 vs Time for 6 Operating Conditions")
plt.colorbar(scatter, label="Operating Condition")  # Show color legend for operating conditions
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()


# %%
# Set the RUL column to 100 for all rows
train_df['rul'] = 100

# Display the updated dataframe
train_df.head()


# %%
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Create a dictionary to store the regression models for each cluster
regression_models = {}

# Loop through each cluster (condition)
for cluster in range(6):
    # Filter the data for the current cluster
    cluster_data = train_df[train_df["condition"] == cluster]
    
    # Define the features (time_in_cycles) and target (rul)
    X = cluster_data[["time_in_cycles"]]
    y = cluster_data["rul"]
    
    # Create and fit the linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Store the model in the dictionary
    regression_models[cluster] = model
    
    # Plot the regression line for the cluster
    plt.plot(X, model.predict(X), label=f"Cluster {cluster} Regression Line", linewidth=2)
    
    # Optionally, plot the data points for the cluster
    plt.scatter(X, y, label=f"Cluster {cluster} Data", alpha=0.6)

# Adding labels and title
plt.xlabel("Time in Cycles")
plt.ylabel("Remaining Useful Life (RUL)")
plt.title("Linear Regression on Each Cluster (6 Operating Conditions)")
plt.legend(loc="best")
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()


# %%
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Step 1: Scale the features in the test dataset (using the same scaler as the train dataset)
scaler = StandardScaler()

# Selecting relevant features in the test dataset
test_features = test_df[["time_in_cycles"] + ["sensor_measurement_" + str(i) for i in range(1, 22)]]
test_features_scaled = scaler.fit_transform(test_features)

# Step 2: Apply the KMeans model to the test data (same model as the one used in train_df)
kmeans = KMeans(n_clusters=6, random_state=0)
test_df["condition"] = kmeans.fit_predict(test_features_scaled)

# Step 3: Predict RUL for the first 248 entries of test data based on the trained linear regression models
test_df["predicted_rul"] = np.nan

# Limit to first 248 rows in test_df
test_df_subset = test_df.iloc[:248]

# Loop through each of the 6 clusters and predict the RUL using the corresponding model
for cluster in range(6):
    # Filter test data for the current cluster
    cluster_test_data = test_df_subset[test_df_subset["condition"] == cluster]
    
    # Extract the 'time_in_cycles' feature (X) for the current cluster
    X_test = cluster_test_data[["time_in_cycles"]]
    
    # Apply the corresponding linear regression model to predict RUL
    model = regression_models[cluster]  # Get the trained model for the cluster
    test_df_subset.loc[test_df_subset["condition"] == cluster, "predicted_rul"] = model.predict(X_test)

# Step 4: Compare the predicted RUL with the true RUL from rul_df (first 248 values)
# Make sure the length of rul_df is 248 and aligns with the subset of test_df
true_rul = rul_df.iloc[:248].values  # Assuming rul_df contains RUL values for the same units

# Step 5: Calculate performance metrics (for example, RMSE)
rmse = np.sqrt(mean_squared_error(true_rul, test_df_subset["predicted_rul"]))
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Optionally: Plot predicted RUL vs true RUL for visualization
plt.figure(figsize=(12, 6))
plt.scatter(true_rul, test_df_subset["predicted_rul"], alpha=0.6, color='b', label="Predicted vs True RUL")
plt.plot([0, max(true_rul)], [0, max(true_rul)], color='r', linestyle='--', label="Perfect Prediction")
plt.xlabel("True RUL")
plt.ylabel("Predicted RUL")
plt.title("True vs Predicted RUL for First 248 Entries of Test Data")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# %%
train_df['rul'][:20]

# %%
rul_df

# %%



