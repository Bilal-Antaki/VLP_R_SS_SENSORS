import sys
import os
# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from src.models.model_registry import get_model
from src.data.data_loader import load_cir_data, scale_and_sequence
from src.config import DATA_CONFIG, GRU_CONFIG, TRAINING_CONFIG, MODEL_CONFIG, TRAINING_OPTIONS
import numpy as np
import random

def train_gru_on_all(processed_dir: str):
    """Train GRU model for position estimation"""
    
    # Set random seed for reproducibility
    random_seed = TRAINING_CONFIG['random_seed']
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    seq_len = MODEL_CONFIG['sequence_length']
    
    # Load data
    df = load_cir_data(processed_dir)
    print(f"Loaded {len(df)} data points")
    
    # Scale and create sequences
    X_seq, y_seq, x_scaler, y_scaler = scale_and_sequence(df, seq_len=seq_len)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X_seq, y_seq, test_size=TRAINING_CONFIG['validation_split'], 
        random_state=random_seed, shuffle=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(X_train, y_train), 
        batch_size=TRAINING_CONFIG['batch_size'], 
        shuffle=True,
        drop_last=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), 
        batch_size=TRAINING_CONFIG['batch_size'],
        drop_last=False
    )
    
    # Create model - 6 features now
    model = get_model("gru", input_dim=6, hidden_dim=GRU_CONFIG['hidden_dim'], 
                     num_layers=GRU_CONFIG['num_layers'], dropout=GRU_CONFIG['dropout'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"Using device: {device}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=TRAINING_CONFIG['learning_rate'], 
        weight_decay=TRAINING_CONFIG['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    train_loss_hist, val_loss_hist = [], []
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(TRAINING_CONFIG['epochs']):
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
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item()
                val_batches += 1
        
        val_loss /= val_batches
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= MODEL_CONFIG['patience']:
            print(f"Early stopping at epoch {epoch+1}")
            break

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Generate predictions on validation set
    model.eval()
    all_val_preds = []
    all_val_targets = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            all_val_preds.extend(preds)
            all_val_targets.extend(y_batch.numpy())

    # Convert to arrays and inverse transform
    val_preds_scaled = np.array(all_val_preds)
    val_targets_scaled = np.array(all_val_targets)

    val_preds = y_scaler.inverse_transform(val_preds_scaled.reshape(-1, 1)).flatten()
    val_targets = y_scaler.inverse_transform(val_targets_scaled.reshape(-1, 1)).flatten()

    rmse = np.sqrt(np.mean((val_targets - val_preds) ** 2))

    print(f"GRU RMSE: {rmse:.4f}")

    return {
        'r_actual': val_targets.tolist(),
        'r_pred': val_preds.tolist(),
        'train_loss': train_loss_hist,
        'val_loss': val_loss_hist,
        'rmse': rmse
    }
