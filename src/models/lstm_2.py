# src/models/lstm_2.py
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from ..config import MODEL_CONFIG, TRAINING_CONFIG, FEATURE_CONFIG

class SequenceToSequenceLSTM(nn.Module):
    """
    LSTM for predicting complete simulation sequences
    Input: 10-step sequence with engineered features
    Output: 10-step sequence of r values
    """
    def __init__(self, input_dim):
        super(SequenceToSequenceLSTM, self).__init__()
        
        self.hidden_dim = MODEL_CONFIG['hidden_dim']
        self.num_layers = MODEL_CONFIG['num_layers']
        self.dropout = MODEL_CONFIG['dropout']
        
        # Encoder LSTM
        self.encoder = nn.LSTM(
            input_dim, 
            self.hidden_dim, 
            self.num_layers, 
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Decoder LSTM 
        self.decoder = nn.LSTM(
            self.hidden_dim * 2,  # *2 for bidirectional
            self.hidden_dim,
            self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        # x shape: [batch, seq_len=10, input_dim]
        
        # Encode the input sequence
        encoder_out, (hidden, cell) = self.encoder(x)
        
        # Prepare decoder input (use encoder output)
        decoder_out, _ = self.decoder(encoder_out)
        
        # Generate predictions for each timestep
        predictions = []
        for t in range(decoder_out.size(1)):
            out = self.fc(decoder_out[:, t, :])
            predictions.append(out)
        
        # Stack predictions: [batch, seq_len=10, 1]
        output = torch.stack(predictions, dim=1)
        
        return output.squeeze(-1)  # [batch, seq_len=10]

def engineer_features(pl, rms):
    """Create engineered features from PL and RMS"""
    features = {
        'PL': pl,
        'RMS': rms,
        'PL_squared': pl ** 2,
        'RMS_squared': rms ** 2,
        'PL_RMS_product': pl * rms,
        'PL_RMS_ratio': pl / (rms + 1e-8),
        'PL_minus_RMS': pl - rms,
        'PL_plus_RMS': pl + rms,
        'PL_log': np.log1p(np.abs(pl)),
        'RMS_log': np.log1p(np.abs(rms)),
        'PL_sqrt': np.sqrt(np.abs(pl)),
        'RMS_sqrt': np.sqrt(np.abs(rms))
    }
    return pd.DataFrame(features)

def select_best_features(X, y, threshold=None):
    """Select features based on correlation with target"""
    if threshold is None:
        threshold = FEATURE_CONFIG['correlation_threshold']
        
    correlations = []
    for col in X.columns:
        corr = np.corrcoef(X[col].values.flatten(), y.values.flatten())[0, 1]
        correlations.append((col, abs(corr)))
    
    correlations.sort(key=lambda x: x[1], reverse=True)
    selected_features = [col for col, corr in correlations if corr > threshold]
    
    # Always include base features
    for feat in FEATURE_CONFIG['base_features']:
        if feat not in selected_features:
            selected_features.append(feat)
    
    return selected_features

def prepare_simulation_data(df, sim_length=None):
    """
    Prepare data by separating into simulations
    Returns sequences of shape (n_simulations, sim_length, n_features)
    """
    if sim_length is None:
        sim_length = FEATURE_CONFIG['simulation_length']
        
    n_simulations = len(df) // sim_length
    
    # Extract features and target
    features_df = engineer_features(df['PL'].values, df['RMS'].values)
    target = df['r'].values
    
    # Select best features
    selected_features = select_best_features(features_df, df['r'])
    print(f"Selected features: {selected_features}")
    
    # Reshape into simulations
    X = features_df[selected_features].values.reshape(n_simulations, sim_length, -1)
    y = target.reshape(n_simulations, sim_length)
    
    return X, y, selected_features

def train_sequence_lstm(df, train_simulations=None, epochs=None, batch_size=None, lr=None):
    """
    Train LSTM model using configuration parameters
    """
    # Use config values if not specified
    train_simulations = train_simulations or TRAINING_CONFIG['train_simulations']
    epochs = epochs or TRAINING_CONFIG['epochs']
    batch_size = batch_size or TRAINING_CONFIG['batch_size']
    lr = lr or TRAINING_CONFIG['learning_rate']
    
    # Prepare data
    X, y, feature_names = prepare_simulation_data(df)
    
    print(f"Total simulations: {len(X)}")
    print(f"Sequence shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Split into train and test
    X_train = X[:train_simulations]
    y_train = y[:train_simulations]
    X_test = X[train_simulations:]
    y_test = y[train_simulations:]
    
    # Further split training data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, 
        test_size=TRAINING_CONFIG['validation_split'], 
        random_state=TRAINING_CONFIG['random_seed']
    )
    
    # Scale features and target
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    n_features = X_train.shape[2]
    
    # Reshape for scaling
    X_train_flat = X_train.reshape(-1, n_features)
    X_val_flat = X_val.reshape(-1, n_features)
    X_test_flat = X_test.reshape(-1, n_features)
    
    # Scale features
    X_train_scaled = feature_scaler.fit_transform(X_train_flat).reshape(X_train.shape)
    X_val_scaled = feature_scaler.transform(X_val_flat).reshape(X_val.shape)
    X_test_scaled = feature_scaler.transform(X_test_flat).reshape(X_test.shape)
    
    # Scale targets
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
    y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).reshape(y_val.shape)
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train_scaled)
    y_train_t = torch.FloatTensor(y_train_scaled)
    X_val_t = torch.FloatTensor(X_val_scaled)
    y_val_t = torch.FloatTensor(y_val_scaled)
    X_test_t = torch.FloatTensor(X_test_scaled)
    
    # Create model
    model = SequenceToSequenceLSTM(input_dim=n_features)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=lr, 
        weight_decay=TRAINING_CONFIG['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15
    )
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        predictions = model(X_train_t.to(device))
        loss = criterion(predictions, y_train_t.to(device))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val_t.to(device))
            val_loss = criterion(val_predictions, y_val_t.to(device))
            val_losses.append(val_loss.item())
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: Train Loss = {loss.item():.6f}, Val Loss = {val_loss.item():.6f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Test on last 4 simulations
    model.eval()
    with torch.no_grad():
        scaled_predictions = model(X_test_t.to(device)).cpu().numpy()
        # Inverse transform the predictions to get back to original scale
        test_predictions = target_scaler.inverse_transform(scaled_predictions.reshape(-1, 1)).reshape(scaled_predictions.shape)
    
    # Calculate metrics
    test_rmse = np.sqrt(np.mean((y_test - test_predictions) ** 2))
    
    # Per-simulation RMSE
    sim_rmses = []
    for i in range(len(y_test)):
        sim_rmse = np.sqrt(np.mean((y_test[i] - test_predictions[i]) ** 2))
        sim_rmses.append(sim_rmse)
    
    print(f"\nTest Results (Last 4 Simulations):")
    print(f"Overall RMSE: {test_rmse:.4f}")
    print(f"Per-simulation RMSE: {sim_rmses}")
    print(f"Mean simulation RMSE: {np.mean(sim_rmses):.4f} ± {np.std(sim_rmses):.4f}")
    
    return {
        'model': model,
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'feature_names': feature_names,
        'predictions': test_predictions,
        'actuals': y_test,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_rmse': test_rmse,
        'sim_rmses': sim_rmses
    }

def build_lstm_2(**kwargs):
    """Factory function for model registry"""
    return SequenceToSequenceLSTM(**kwargs)

# Example usage
