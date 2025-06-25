import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from src.data.data_loader import load_cir_data
from src.config import DATA_CONFIG, GRU_CONFIG, TRAINING_CONFIG, ANALYSIS_CONFIG
import warnings
import random
from src.data.feature_engineering import create_engineered_features, select_features_rf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

random.seed(42)
warnings.filterwarnings('ignore')




df_list = []
    
# Load all available datasets
for keyword in DATA_CONFIG['datasets']:
    try:
        df_temp = load_cir_data(DATA_CONFIG['processed_dir'], filter_keyword=keyword)
        print(f"  Loaded {keyword}: {len(df_temp)} samples")
        df_list.append(df_temp)
    except:
        print(f"  {keyword} not found")

# Combine all data
df_all = pd.concat(df_list, ignore_index=True) if df_list else None

# Extract target column before feature engineering
y = df_all[DATA_CONFIG['target_column']]

# 2. Feature Engineering
df_engineered = create_engineered_features(df_all)

# Select features - exclude any coordinate-based features
feature_cols = [col for col in df_engineered.columns 
                if col not in ANALYSIS_CONFIG['feature_selection']['excluded_features']]

X = df_engineered[feature_cols]

# Select best features
selected_feature_names = select_features_rf(X, y, k=5) 

# Create DataFrame with only selected features
X_rf = X[selected_feature_names]

# Save selected features to CSV
X_rf.to_csv('data/processed/selected_features.csv', index=False)

# scale the data
scaler = MinMaxScaler()
X_rf_scaled = scaler.fit_transform(X_rf)
X_rf_scaled = pd.DataFrame(X_rf_scaled, columns=X_rf.columns)
y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_rf_scaled, y_scaled, test_size=0.2, random_state=42, shuffle=False)

# Reshape data for GRU: (samples, timesteps, features)
# Since we have single point estimation, we'll use 1 timestep
X_train_reshaped = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_reshaped = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

# create the model
model = keras.Sequential([
    keras.layers.GRU(units=100, return_sequences=False, input_shape=(1, X_train.shape[1])),
    keras.layers.Dense(units=64, activation='relu'),
    keras.layers.Dense(units=1)
])

# compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# train the model
model.fit(X_train_reshaped, y_train, epochs=100, batch_size=32, validation_data=(X_test_reshaped, y_test), shuffle=False)

# predict the future values
y_pred = model.predict(X_test_reshaped)

# inverse transform the data
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test)

# plot the results
plt.plot(y_test_inv, label='True')
plt.plot(y_pred_inv, label='Predicted')
plt.legend()
plt.show()



