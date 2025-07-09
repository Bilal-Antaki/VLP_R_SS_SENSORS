import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from src.models.rnn import SimpleRNN
from src.config import MODEL_CONFIG, TRAINING_CONFIG
from src.data.data_loader import load_cir_data, scale_and_sequence
from sklearn.model_selection import train_test_split

def train_rnn_on_all(processed_dir):
    """Train RNN model for position estimation"""
    
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
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), 
        batch_size=TRAINING_CONFIG['batch_size']
    )
    
    # Initialize model - 6 features now
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = SimpleRNN(
        input_size=6,
        hidden_size=MODEL_CONFIG['hidden_size'],
        num_layers=MODEL_CONFIG['num_layers'],
        dropout=MODEL_CONFIG['dropout']
    ).to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=TRAINING_CONFIG['learning_rate'],
        weight_decay=TRAINING_CONFIG['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    best_model_state = None
    
    for epoch in range(TRAINING_CONFIG['epochs']):
        # Training
        model.train()
        train_loss = 0
        train_batches = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1
        
        train_loss /= train_batches
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                val_loss += criterion(y_pred, y_batch).item()
                val_batches += 1
        
        val_loss /= val_batches
        val_losses.append(val_loss)
        
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
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        # Print progress
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:03d}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Evaluate on validation set
    model.eval()
    all_val_preds = []
    all_val_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            all_val_preds.extend(preds)
            all_val_targets.extend(y_batch.cpu().numpy())

    # Convert to arrays and inverse transform
    val_preds_scaled = np.array(all_val_preds)
    val_targets_scaled = np.array(all_val_targets)
    val_preds = y_scaler.inverse_transform(val_preds_scaled.reshape(-1, 1)).flatten()
    val_targets = y_scaler.inverse_transform(val_targets_scaled.reshape(-1, 1)).flatten()

    rmse = np.sqrt(np.mean((val_targets - val_preds) ** 2))

    print(f"RNN RMSE: {rmse:.4f}")

    return {
        'rmse': rmse,
        'train_loss': train_losses,
        'val_loss': val_losses,
        'r_actual': val_targets.flatten(),
        'r_pred': val_preds.flatten()
    } 