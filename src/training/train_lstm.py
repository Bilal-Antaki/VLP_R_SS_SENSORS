import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from src.models.model_registry import get_model
from src.data.loader import load_cir_data
from src.data.preprocessing import scale_and_sequence
from src.config import DATA_CONFIG
import numpy as np
import pandas as pd
import random
import time

def train_lstm_on_all(processed_dir: str, batch_size: int = 32, epochs: int = 300, lr: float = 0.01):
    # Generate random seed based on current time
    random_seed = int(time.time() * 1000) % 100000
    print(f"Using random seed: {random_seed}")
    
    # Set random seeds
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # Use longer sequences for better temporal patterns
    seq_len = 10
    
    # Load data using dataset from config
    df = load_cir_data(processed_dir, filter_keyword=DATA_CONFIG['datasets'][0])
    print(f"Loaded {len(df)} data points from {DATA_CONFIG['datasets'][0]}")
    
    # Check data distribution
    print(f"Target (r) statistics:")
    print(f"  Mean: {df['r'].mean():.2f}")
    print(f"  Std: {df['r'].std():.2f}")
    
    # Scale and create sequences
    X_seq, y_seq, x_scaler, y_scaler = scale_and_sequence(df, seq_len=seq_len)
    
    if len(X_seq) < 100:
        print(f"Warning: Very few sequences ({len(X_seq)}). Consider reducing seq_len.")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42, shuffle=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(X_train, y_train), 
        batch_size=batch_size, 
        shuffle=True,
        drop_last=True  # Ensure consistent batch sizes
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), 
        batch_size=batch_size,
        drop_last=False
    )
    
    # Create model with better architecture
    model = get_model("lstm", input_dim=2, hidden_dim=64, num_layers=2, dropout=0.2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"Using device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    train_loss_hist, val_loss_hist = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 20
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_batches = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1
        
        train_loss /= train_batches
        
        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        y_val_actual, y_val_pred = [], []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item()
                val_batches += 1
                
                y_val_actual.extend(y_batch.cpu().numpy())
                y_val_pred.extend(preds.cpu().numpy())
        
        val_loss /= val_batches
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        
        # Learning rate scheduling
        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        # Manual verbose output for learning rate changes
        if new_lr != prev_lr:
            print(f"  Learning rate reduced from {prev_lr:.6f} to {new_lr:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
            # Check prediction diversity
            pred_std = np.std(y_val_pred)
            print(f"  Prediction std: {pred_std:.6f}")
    
    # Generate predictions on full dataset
    model.eval()
    with torch.no_grad():
        full_preds_scaled = model(X_seq.to(device)).cpu().numpy()
        full_targets_scaled = y_seq.numpy()
    
    # Inverse transform
    full_preds = y_scaler.inverse_transform(full_preds_scaled.reshape(-1, 1)).flatten()
    full_targets = y_scaler.inverse_transform(full_targets_scaled.reshape(-1, 1)).flatten()
    
    rmse = np.sqrt(np.mean((full_targets - full_preds) ** 2))
    
    print(f"\nFinal Metrics:")
    print(f"RMSE: {rmse:.4f}")
    print(f"Prediction range: [{full_preds.min():.2f}, {full_preds.max():.2f}]")
    print(f"Target range: [{full_targets.min():.2f}, {full_targets.max():.2f}]")
    print(f"Prediction std: {np.std(full_preds):.4f}")
    print(f"Target std: {np.std(full_targets):.4f}")
    
    # Return additional info for size alignment
    return {
        'r_actual': full_targets.tolist(),
        'r_pred': full_preds.tolist(),
        'train_loss': train_loss_hist,
        'val_loss': val_loss_hist,
        'rmse': rmse,
        'original_df_size': len(df),
        'sequence_size': len(full_targets),
        'seq_len': seq_len
    }