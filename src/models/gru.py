import torch
import torch.nn as nn
import torch.nn.functional as F
from ..config import GRU_CONFIG

class GRURegressor(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=None, num_layers=None, dropout=None):
        super(GRURegressor, self).__init__()
        
        # Use provided parameters or fall back to config values
        self.hidden_dim = hidden_dim if hidden_dim is not None else GRU_CONFIG['hidden_dim']
        self.num_layers = num_layers if num_layers is not None else GRU_CONFIG['num_layers']
        self.dropout = dropout if dropout is not None else GRU_CONFIG['dropout']
        
        # GRU layer
        self.gru = nn.GRU(
            input_dim, 
            self.hidden_dim, 
            self.num_layers, 
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 1)
        )
        
        # Layer normalization for better gradient flow
        self.layer_norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        batch_size = x.size(0)
        
        # GRU forward pass
        gru_out, hidden = self.gru(x)
        # gru_out: [batch, seq_len, hidden_dim]
        
        # Use the last timestep output (standard approach)
        last_output = gru_out[:, -1, :]  # [batch, hidden_dim]
        
        # Apply layer normalization
        last_output = self.layer_norm(last_output)
        
        # Pass through output layers
        output = self.fc(last_output)
        
        return output.squeeze(-1)  # [batch]

    def init_hidden(self, batch_size, device):
        """Initialize hidden state for GRU"""
        h0 = torch.zeros(
            self.num_layers, 
            batch_size, 
            self.hidden_dim
        ).to(device)
        return h0


class GRUWithResidual(GRURegressor):
    """GRU with residual connections for deeper networks"""
    def __init__(self, input_dim=2, **kwargs):
        super(GRUWithResidual, self).__init__(input_dim, **kwargs)
        
        # Add a projection layer for residual connection if dimensions don't match
        if input_dim != self.hidden_dim:
            self.input_projection = nn.Linear(input_dim, self.hidden_dim)
        else:
            self.input_projection = None
    
    def forward(self, x):
        # Get the base GRU output
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Project input for residual if needed
        if self.input_projection is not None:
            residual = self.input_projection(x[:, -1, :])  # Use last timestep
        else:
            residual = x[:, -1, :]
        
        # Standard GRU processing
        output = super().forward(x)
        
        return output


def build_gru_model(**kwargs):
    """Factory function for basic GRU model"""
    return GRURegressor(**kwargs)

def build_gru_residual(**kwargs):
    """Factory function for GRU with residual connections"""
    return GRUWithResidual(**kwargs)