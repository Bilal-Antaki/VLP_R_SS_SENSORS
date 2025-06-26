import sys
import os
# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from src.data.data_loader import load_cir_data, scale_and_sequence
from src.config import DATA_CONFIG, GRU_CONFIG, TRAINING_CONFIG, MODEL_CONFIG, TRAINING_OPTIONS
import numpy as np
import random

class GRUModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=GRU_CONFIG['hidden_dim'], num_layers=GRU_CONFIG['num_layers'], dropout=GRU_CONFIG['dropout']):
        super(GRUModel, self).__init__()
        # Use provided parameters or fall back to config values
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GRU layer with matching LSTM architecture
        self.gru = nn.GRU(
            input_dim, 
            self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout
        )
        
        # Complex output layer matching LSTM
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        # GRU forward pass
        gru_out, _ = self.gru(x)
        
        # Use only the last output
        last_out = gru_out[:, -1, :]
        
        # Pass through complex output layer
        out = self.fc(last_out)
        return out.squeeze(-1)

def train_gru_on_all(processed_dir: str, batch_size: int = None, epochs: int = None, lr: float = None):
    """
    Train GRU model for position estimation with architecture matching successful LSTM
    """
    # Use config values if not provided
    batch_size = batch_size if batch_size is not None else TRAINING_CONFIG['batch_size']
    epochs = epochs if epochs is not None else TRAINING_CONFIG['epochs']
    lr = lr if lr is not None else TRAINING_CONFIG['learning_rate']
    
    # Generate random seed based on current time
    random_seed = TRAINING_CONFIG['random_seed']
    print(f"Using random seed: {random_seed}")
    
    # Set random seeds
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # Use sequence length from config
    seq_len = GRU_CONFIG.get('sequence_length', 10)
    
    # Load data using dataset from config
    df = load_cir_data(processed_dir, filter_keyword=DATA_CONFIG['datasets'][0])
    
    
    # Scale and create sequences
    X_seq, y_seq, x_scaler, y_scaler = scale_and_sequence(df, seq_len=seq_len)
    
    if len(X_seq) < 100:
        print(f"Warning: Very few sequences ({len(X_seq)}). Consider reducing seq_len.")
    
    # Split data using config validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_seq, y_seq, test_size=TRAINING_CONFIG['validation_split'], 
        random_state=random_seed, shuffle=True
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
    
    # Create model with config values
    model = GRUModel(
        input_dim=2,
        hidden_dim=GRU_CONFIG['hidden_dim'],
        num_layers=GRU_CONFIG['num_layers'],
        dropout=GRU_CONFIG['dropout']
    )
    
    # Initialize weights using orthogonal initialization
    def init_weights(m):
        if isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.uniform_(param, -0.1, 0.1)
        elif isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.uniform_(m.bias, -0.1, 0.1)
    
    model.apply(init_weights)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"Using device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Use same loss and optimizer as LSTM with config values
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=lr, 
        weight_decay=TRAINING_CONFIG['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    train_loss_hist, val_loss_hist = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = TRAINING_CONFIG.get('patience', 20)
    
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

        print(f"\nFinal Metrics:")
        print(f"RMSE: {rmse:.4f}")
        print(f"Prediction range: [{val_preds.min():.2f}, {val_preds.max():.2f}]")
        print(f"Target range: [{val_targets.min():.2f}, {val_targets.max():.2f}]")
        print(f"Prediction std: {np.std(val_preds):.4f}")
        print(f"Target std: {np.std(val_targets):.4f}")

        # Return validation results instead of full dataset results
        return {
        'r_actual': val_targets.tolist(),
        'r_pred': val_preds.tolist(),
        'train_loss': train_loss_hist,
        'val_loss': val_loss_hist,
        'rmse': rmse,
        'original_df_size': len(df),
        'sequence_size': len(val_targets),  # Now this is validation size
        'seq_len': seq_len
    }
