import torch
import torch.nn as nn
from src.config import MODEL_CONFIG

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super(SimpleRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # RNN layer with residual connection
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Enhanced output layers with residual connection
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc3 = nn.Linear(hidden_size // 4, 1)
        
        # Layer normalization for better training stability
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate RNN
        out, _ = self.rnn(x, h0)
        
        # Apply layer normalization
        out = self.layer_norm(out[:, -1, :])
        
        # Enhanced output processing with residual connection
        residual = out
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.elu(out)
        out = self.dropout(out)
        
        out = self.fc3(out)
        
        # Add residual connection if dimensions match
        if residual.size(-1) == out.size(-1):
            out = out + residual
        
        return out
