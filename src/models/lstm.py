import torch
import torch.nn as nn
from ..config import MODEL_CONFIG

class LSTMRegressor(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=None, num_layers=None, dropout=None):
        super(LSTMRegressor, self).__init__()
        # Use provided parameters or fall back to config values
        self.hidden_dim = hidden_dim if hidden_dim is not None else MODEL_CONFIG['hidden_dim']
        self.num_layers = num_layers if num_layers is not None else MODEL_CONFIG['num_layers']
        self.dropout = dropout if dropout is not None else MODEL_CONFIG['dropout']
        
        # Add dropout for regularization
        self.lstm = nn.LSTM(
            input_dim, 
            self.hidden_dim, 
            self.num_layers, 
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )
        
        # Add a more complex output layer
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 1)
        )

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        lstm_out, _ = self.lstm(x)
        # Take the last timestep output
        last_output = lstm_out[:, -1, :]  # [batch, hidden_dim]
        output = self.fc(last_output)
        return output.squeeze(-1)  # [batch]

def build_lstm_model(**kwargs):
    return LSTMRegressor(**kwargs)
