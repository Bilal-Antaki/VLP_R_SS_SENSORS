import pandas as pd
import os
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from ..config import DATA_CONFIG

def load_cir_data(processed_dir: str, filter_keyword: str = None) -> pd.DataFrame:
    """Load CIR data from processed directory"""
    filepath = os.path.join(processed_dir, DATA_CONFIG['input_file'].split('/')[-1])
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} data points from {filepath}")
        return df
    else:
        raise FileNotFoundError(f"File not found: {filepath}")

def extract_features_and_target(df: pd.DataFrame):
    """Extract features and target from DataFrame"""
    features = DATA_CONFIG['feature_columns']
    target = DATA_CONFIG['target_column']
    
    # Check if all required columns exist
    missing_cols = [col for col in features + [target] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in data: {missing_cols}")
    
    X = df[features]
    y = df[target]
    return X, y

def sequence_split(X, y, seq_len):
    """Create sequences for RNN training"""
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len + 1):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len-1])
    return np.array(X_seq), np.array(y_seq)

def scale_and_sequence(df, seq_len=10):
    """Scale features and create sequences for RNN models"""
    X, y = extract_features_and_target(df)
    
    X = X.values
    y = y.values
    
    # Use StandardScaler for better gradient flow
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    X_scaled = x_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
    
    print(f"Original y (r) range: [{y.min():.3f}, {y.max():.3f}]")
    print(f"Data shape: X={X_scaled.shape}, y={y_scaled.shape}")
    
    # Create sequences
    X_seq, y_seq = sequence_split(X_scaled, y_scaled, seq_len)
    
    print(f"Sequence shape: X_seq={X_seq.shape}, y_seq={y_seq.shape}")
    
    return (
        torch.tensor(X_seq, dtype=torch.float32),
        torch.tensor(y_seq, dtype=torch.float32),
        x_scaler,
        y_scaler
    )

def scale_features_only(df):
    """Simple scaling without sequencing for linear models"""
    X, y = extract_features_and_target(df)
    
    X = X.values
    y = y.values
    
    x_scaler = StandardScaler()
    X_scaled = x_scaler.fit_transform(X)
    
    return X_scaled, y, x_scaler 