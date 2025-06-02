import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

def sequence_split(X, y, seq_len):
    """Create sequences for LSTM training"""
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len + 1):  # Fixed off-by-one error
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len-1])  # Predict current timestep, not future
    return np.array(X_seq), np.array(y_seq)

def scale_and_sequence(df, seq_len=10, features=['PL', 'RMS'], target='r'):
    """Improved scaling and sequencing for LSTM"""
    
    # Sort by position to ensure temporal consistency
    df_sorted = df.copy()
    
    X = df_sorted[features].values
    y = df_sorted[target].values
    
    # Use StandardScaler instead of MinMaxScaler for better gradient flow
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    # Fit scalers
    X_scaled = x_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
    
    # Check for data issues
    print(f"Original y (r) range: [{y.min():.3f}, {y.max():.3f}]")
    print(f"Data shape: X={X_scaled.shape}, y={y_scaled.shape}")
    print(f"Scaled X range: [{X_scaled.min():.3f}, {X_scaled.max():.3f}]")
    print(f"Scaled y range: [{y_scaled.min():.3f}, {y_scaled.max():.3f}]")
    
    # Create sequences
    X_seq, y_seq = sequence_split(X_scaled, y_scaled, seq_len)
    
    print(f"Sequence shape: X_seq={X_seq.shape}, y_seq={y_seq.shape}")
    
    return (
        torch.tensor(X_seq, dtype=torch.float32),
        torch.tensor(y_seq, dtype=torch.float32),
        x_scaler,
        y_scaler
    )

def scale_features_only(df, features=['PL', 'RMS'], target='r'):
    """Simple scaling without sequencing for linear models"""
    X = df[features].values
    y = df[target].values
    
    x_scaler = StandardScaler()
    X_scaled = x_scaler.fit_transform(X)
    
    return X_scaled, y, x_scaler