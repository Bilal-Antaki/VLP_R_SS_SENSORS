# src/evaluation/metrics.py
import numpy as np
from sklearn.metrics import mean_squared_error

def calculate_rmse(y_true, y_pred):
    """
    Calculate RMSE for regression evaluation
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        RMSE value
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

def print_rmse(model_name, y_true, y_pred):
    """
    Print RMSE for a model
    
    Args:
        model_name: Name of the model
        y_true: True values
        y_pred: Predicted values
    """
    rmse = calculate_rmse(y_true, y_pred)
    print(f"{model_name} RMSE: {rmse:.4f}")
    return rmse