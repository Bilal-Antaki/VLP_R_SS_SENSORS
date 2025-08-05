from sklearn.model_selection import train_test_split
from src.data.data_loader import load_cir_data, scale_and_sequence
from src.config import TRAINING_CONFIG, GRU_CONFIG, LSTM_CONFIG, RNN_CONFIG, DATA_CONFIG
from src.evaluation.metrics import calculate_rmse, time_execution
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :]).squeeze(-1)

class GRURegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))
    
    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_output = self.layer_norm(gru_out[:, -1, :])
        return self.fc(last_output).squeeze(-1)

class RNNRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.rnn(x, h0)
        return self.fc(out[:, -1, :]).squeeze(-1)

@time_execution
def train_deep_model(processed_dir: str, model_name: str):
    """Train deep learning model with simplified training loop"""
    torch.manual_seed(TRAINING_CONFIG['random_seed'])
    np.random.seed(TRAINING_CONFIG['random_seed'])
    
    df = load_cir_data(processed_dir)
    X_seq, y_seq, x_scaler, y_scaler = scale_and_sequence(df, seq_len=10)
    
    # Determine input dimension from feature columns
    input_dim = len(DATA_CONFIG['feature_columns'])
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_seq, y_seq, test_size=TRAINING_CONFIG['validation_split'], 
        random_state=TRAINING_CONFIG['random_seed'], shuffle=False
    )
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=TRAINING_CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=TRAINING_CONFIG['batch_size'], shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_classes = {
        'lstm': lambda: LSTMRegressor(
            input_dim=input_dim,
            hidden_dim=LSTM_CONFIG['hidden_dim'],
            num_layers=LSTM_CONFIG['num_layers'],
            dropout=LSTM_CONFIG['dropout']
        ),
        'gru': lambda: GRURegressor(
            input_dim=input_dim,
            hidden_dim=GRU_CONFIG['hidden_dim'],
            num_layers=GRU_CONFIG['num_layers'],
            dropout=GRU_CONFIG['dropout']
        ),
        'rnn': lambda: RNNRegressor(
            input_dim=input_dim,
            hidden_dim=RNN_CONFIG['hidden_dim'],
            num_layers=RNN_CONFIG['num_layers'],
            dropout=RNN_CONFIG['dropout']
        )
    }
    
    model = model_classes[model_name]().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAINING_CONFIG['learning_rate'], weight_decay=TRAINING_CONFIG['weight_decay'])
    
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(TRAINING_CONFIG['epochs']):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                val_loss += criterion(pred, y_batch).item()
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Generate predictions on validation set
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y_batch.numpy())

    # Inverse transform predictions
    y_pred = y_scaler.inverse_transform(np.array(all_preds).reshape(-1, 1)).flatten()
    y_actual = y_scaler.inverse_transform(np.array(all_targets).reshape(-1, 1)).flatten()
    
    rmse = calculate_rmse(y_actual, y_pred)
    
    return {
        'name': model_name,
        'y_actual': y_actual.tolist(),
        'y_pred': y_pred.tolist(),
        'train_loss': train_losses,
        'val_loss': val_losses,
        'rmse': rmse
    }

def train_all_deep_models(processed_dir: str):
    """Train all deep learning models"""
    models = ['lstm', 'gru', 'rnn']
    results = []
    
    for model_name in models:
        result, runtime = train_deep_model(processed_dir, model_name)
        result['runtime'] = runtime
        results.append(result)
    
    return results
