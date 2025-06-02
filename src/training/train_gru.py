import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from src.data.loader import load_cir_data
from src.data.preprocessing import scale_and_sequence
from src.config import DATA_CONFIG
import numpy as np
import random
import time

class GRUModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=256, num_layers=3, dropout=0.4):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Simple GRU layer
        self.gru = nn.GRU(
            input_dim, 
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Simple output layer
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        # GRU forward pass
        gru_out, _ = self.gru(x)
        
        # Use only the last output
        last_out = gru_out[:, -1, :]
        
        # Predict single value
        out = self.fc(last_out)
        return out.squeeze(-1)

def train_gru_on_all(processed_dir: str, model_variant: str = "gru", 
                     batch_size: int = 32, epochs: int = 300, 
                     lr: float = 0.001, seq_len: int = 15):
    """
    Train simplified GRU model for position estimation
    """
    # Generate random seed based on current time
    random_seed = int(time.time() * 1000) % 100000
    print(f"Using random seed: {random_seed}")
    
    # Set random seeds
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # Load data using dataset from config
    df = load_cir_data(processed_dir, filter_keyword=DATA_CONFIG['datasets'][0])
    print(f"Loaded {len(df)} data points from {DATA_CONFIG['datasets'][0]}")
    
    # Scale and create sequences
    X_seq, y_seq, x_scaler, y_scaler = scale_and_sequence(df, seq_len=seq_len)
    
    # Split data with random seed
    X_train, X_val, y_train, y_val = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=random_seed, shuffle=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(X_train, y_train), 
        batch_size=batch_size, 
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(random_seed + 1)
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), 
        batch_size=batch_size,
        drop_last=False
    )
    
    # Create simplified model
    model = GRUModel(
        input_dim=2,
        hidden_dim=256,
        num_layers=3,
        dropout=0.4
    )
    
    # Initialize weights
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
    
    # Use MSE loss
    criterion = nn.MSELoss()
    
    # Use AdamW optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
        betas=(0.9, 0.99)
    )
    
    # Use OneCycle learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    train_loss_hist = []
    val_loss_hist = []
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
            scheduler.step()
            
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
    
    # Generate predictions on full dataset
    model.eval()
    with torch.no_grad():
        X_seq_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)
        full_preds_scaled = model(X_seq_tensor).cpu().numpy()
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
    
    return {
        'r_actual': full_targets.tolist(),
        'r_pred': full_preds.tolist(),
        'train_loss': train_loss_hist,
        'val_loss': val_loss_hist,
        'rmse': rmse
    }