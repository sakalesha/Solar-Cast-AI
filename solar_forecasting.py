import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Ensure output directories exist
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# STEP 1: Load & Understand Data
dataset_path = 'Dataset'
all_csv_files = sorted(glob.glob(os.path.join(dataset_path, "*.csv")))

# Optimization: Filter for the last 10 years (2015-2024)
# Files are named like "204191_34.05_-118.26_2024.csv"
csv_files = [f for f in all_csv_files if any(str(year) in f for year in range(2015, 2025))]
print(f"Found {len(csv_files)} files for the period 2015-2024.")

df_list = []
for file in csv_files:
    # Read CSV, skipping the first 2 metadata lines. Header is on line 3.
    temp_df = pd.read_csv(file, skiprows=2)
    df_list.append(temp_df)

df = pd.concat(df_list, ignore_index=True)
print(f"Combined dataframe shape: {df.shape}")

# STEP 2: Select Required Columns
# Mapping conceptual 'Humidity' to 'Relative Humidity' found in dataset
required_cols = ['GHI', 'Temperature', 'Wind Speed', 'Relative Humidity', 'Pressure']
df = df[required_cols]

# STEP 3: Handle Missing Values
df = df.ffill()

# STEP 4: Basic Cleaning
df = df[df['GHI'] >= 0]

# STEP 5: Normalize Data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# STEP 6: Create Sequences (CORE LOGIC)
def create_sequences(data, n_steps=72):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps, 0])  # GHI is index 0
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, n_steps=72)
print(f"Sequences created. X shape: {X.shape}, y shape: {y.shape}")

# STEP 7: Train / Validation / Test Split
train_size = int(len(X) * 0.75)
val_size = int(len(X) * 0.125)

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

# STEP 8: Build LSTM Model
model = Sequential([
    LSTM(32, return_sequences=True, input_shape=(72, X.shape[2])),
    LSTM(16, return_sequences=True),
    LSTM(8),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# STEP 9: Train Model
print("Starting training (5 epochs)...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=32,
    verbose=1
)

# STEP 10: Evaluate
pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, pred))
print("RMSE:", rmse)

# Save model to designated folder
model_path = os.path.join('models', 'solar_model.h5')
# Note: Keras recommends using .keras extension for newer versions, 
# but .h5 is standard and specifically requested.
model.save(model_path)
print(f"Model saved to {model_path}")

# Visualization to designated folder
plt.figure(figsize=(12, 6))
plt.plot(y_test[:200], label='Actual GHI', color='blue', alpha=0.7)
plt.plot(pred[:200], label='Predicted GHI', color='orange', alpha=0.9)
plt.title('GHI Prediction vs Actual (First 200 samples of Test Set)')
plt.xlabel('Time Step (Hours)')
plt.ylabel('Normalized GHI')
plt.grid(True, alpha=0.3)
plt.legend()
plot_path = os.path.join('plots', 'prediction_plot.png')
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")
