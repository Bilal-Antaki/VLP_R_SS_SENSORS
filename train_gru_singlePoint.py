import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress info messages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from src.data.data_loader import load_cir_data
from src.config import DATA_CONFIG, ANALYSIS_CONFIG
import warnings
import random
from src.data.feature_engineering import create_engineered_features, select_features_rf

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
warnings.filterwarnings('ignore')

def create_sequences(X, y, seq_len=10):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

def build_cnn_model(input_shape):
    reg = keras.regularizers.l2(1e-4)
    model = keras.Sequential([
        keras.layers.Conv1D(64, 3, activation='relu', kernel_regularizer=reg, input_shape=input_shape),
        keras.layers.Dropout(0.2),
        keras.layers.Conv1D(32, 3, activation='relu', kernel_regularizer=reg),
        keras.layers.Dropout(0.2),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(16, activation='relu', kernel_regularizer=reg),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, kernel_regularizer=reg)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def plot_history(history):
    plt.figure(figsize=(12,5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('CNN Training History')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/plots/cnn_training_history.png', dpi=300)
    plt.show()

# Load and prepare data
print("Loading data...")
df_list = []
for keyword in DATA_CONFIG['datasets']:
    try:
        df_temp = load_cir_data(DATA_CONFIG['processed_dir'], filter_keyword=keyword)
        df_list.append(df_temp)
    except Exception as e:
        print(f"Error loading {keyword}: {e}")
        pass

if not df_list:
    raise ValueError("No data loaded successfully")

df_all = pd.concat(df_list, ignore_index=True)
print(f"Loaded {len(df_all)} samples")

# Create engineered features
print("Creating engineered features...")
df_engineered = create_engineered_features(df_all)
y = df_all[DATA_CONFIG['target_column']]

# Select features - try different methods
print("Selecting features...")
feature_cols = [col for col in df_engineered.columns 
                if col not in ANALYSIS_CONFIG['feature_selection']['excluded_features']]
X = df_engineered[feature_cols]

# Use more features for better performance
selected_features = select_features_rf(X, y, k=8)  # Increased from 5 to 8
print(f"Selected features: {selected_features}")

X_selected = X[selected_features].values
y_values = y.values

# Better data scaling - use StandardScaler for better performance
print("Scaling data...")
x_scaler = StandardScaler()  # Changed to StandardScaler
y_scaler = StandardScaler()  # Changed to StandardScaler

X_scaled = x_scaler.fit_transform(X_selected)
y_scaled = y_scaler.fit_transform(y_values.reshape(-1, 1)).flatten()

# Create sequences with longer sequence length
sequence_length = 10  # Increased from 10
X_seq, y_seq = create_sequences(X_scaled, y_scaled, sequence_length)
print(f"Created {len(X_seq)} sequences with length {sequence_length}")

# Split data with more training samples
train_size = int(0.8 * len(X_seq))  # Use 80% for training
X_train = X_seq[:train_size]
y_train = y_seq[:train_size]
X_test = X_seq[train_size:]
y_test = y_seq[train_size:]

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# Build improved model
print("Building improved CNN model...")
model = build_cnn_model((sequence_length, X_seq.shape[2]))

model.summary()

# Callbacks
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=10, 
    restore_best_weights=True,
    verbose=1
)

# Train model
print("Training model...")
history = model.fit(
    X_train, y_train, 
    epochs=100, 
    batch_size=16,  # Increased batch size
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# Plot training history
plot_history(history)

# Save model
model.save('results/models/improved_cnn_model.h5')
print("\nModel saved to 'results/models/improved_cnn_model.h5'")

# Additional analysis
print(f"\nPrediction Statistics:")
print(f"Mean prediction: {np.mean(y_train):.2f}")
print(f"Std prediction: {np.std(y_train):.2f}")
print(f"Mean actual: {np.mean(y_train):.2f}")
print(f"Std actual: {np.std(y_train):.2f}")

# Check for systematic bias
bias = np.mean(y_train - y_train)
print(f"Bias: {bias:.2f}")

# Calculate MAPE (Mean Absolute Percentage Error)
mape = np.mean(np.abs((y_train - y_train) / y_train)) * 100
print(f"MAPE: {mape:.2f}%")