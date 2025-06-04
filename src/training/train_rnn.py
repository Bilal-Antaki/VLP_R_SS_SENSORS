import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from src.models.rnn import SimpleRNN
from src.config import MODEL_CONFIG, TRAINING_OPTIONS
from src.data.loader import load_cir_data
import time
import os
import random

def create_sequences(X, y, seq_length):
    """Create sequences for RNN training with overlap for better data utilization"""
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length + 1):  # Changed to include more sequences
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length - 1])
    return np.array(X_seq), np.array(y_seq)

def train_rnn_on_all(processed_dir):
    """Train RNN model on all available data"""
    print("\nTraining RNN model...")
    
    # Set fixed random seed for reproducibility
    random_seed = 42
    print(f"Using fixed random seed: {random_seed}")
    
    # Set random seeds for all sources of randomness
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # Load data
    df = load_cir_data(processed_dir, filter_keyword='FCPR-D1')
    print(f"Loaded {len(df)} data points from FCPR-D1")
    
    # Prepare features and target
    X = df[['PL', 'RMS']].values
    y = df['r'].values
    
    print(f"Original y (r) range: [{y.min():.3f}, {y.max():.3f}]")
    
    # Scale the data
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    X_scaled = X_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()
    
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Scaled X range: [{X_scaled.min():.3f}, {X_scaled.max():.3f}]")
    print(f"Scaled y range: [{y_scaled.min():.3f}, {y_scaled.max():.3f}]")
    
    # Create sequences with overlap
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, MODEL_CONFIG['sequence_length'])
    print(f"Sequence shape: X_seq={X_seq.shape}, y_seq={y_seq.shape}")
    
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X_seq)
    y_tensor = torch.FloatTensor(y_seq).reshape(-1, 1)
    
    # Create train/validation split with stratification
    train_size = int(len(X_tensor) * 0.8)
    indices = np.random.permutation(len(X_tensor))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    X_train, X_val = X_tensor[train_indices], X_tensor[val_indices]
    y_train, y_val = y_tensor[train_indices], y_tensor[val_indices]
    
    # Create data loaders with better batch size
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=MODEL_CONFIG['batch_size'], 
        shuffle=True,
        num_workers=0,  # Adjust based on your system
        pin_memory=True  # Faster data transfer to GPU
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=MODEL_CONFIG['batch_size'],
        num_workers=0,
        pin_memory=True
    )
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = SimpleRNN(
        input_size=X.shape[1],
        hidden_size=MODEL_CONFIG['hidden_size'],
        num_layers=MODEL_CONFIG['num_layers'],
        dropout=MODEL_CONFIG['dropout']
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Loss function and optimizer with weight decay
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=MODEL_CONFIG['learning_rate'],
        weight_decay=1e-5
    )
    
    # Cosine annealing with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # First restart epoch
        T_mult=2,  # Multiply T_0 by this factor after each restart
        eta_min=1e-6  # Minimum learning rate
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience = MODEL_CONFIG['patience']
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    # Gradient clipping threshold
    grad_clip = 1.0
    
    for epoch in range(MODEL_CONFIG['epochs']):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                val_loss += criterion(y_pred, y_batch).item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
            }, 'results/models/best_rnn_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch + 1:03d}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}, LR = {current_lr:.6f}")
    
    # Load best model for evaluation
    checkpoint = torch.load('results/models/best_rnn_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Make predictions on validation set
    with torch.no_grad():
        y_pred = model(X_val.to(device)).cpu().numpy()
    
    # Inverse transform predictions
    y_pred = y_scaler.inverse_transform(y_pred)
    y_val_orig = y_scaler.inverse_transform(y_val.numpy())
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((y_val_orig - y_pred) ** 2))
    
    print("\nFinal Metrics:")
    print(f"RMSE: {rmse:.4f}")
    print(f"Prediction range: [{y_pred.min():.2f}, {y_pred.max():.2f}]")
    print(f"Target range: [{y_val_orig.min():.2f}, {y_val_orig.max():.2f}]")
    print(f"Prediction std: {y_pred.std():.4f}")
    print(f"Target std: {y_val_orig.std():.4f}")
    
    return {
        'rmse': rmse,
        'train_loss': train_losses,
        'val_loss': val_losses,
        'r_actual': y_val_orig.flatten(),
        'r_pred': y_pred.flatten()
    } 