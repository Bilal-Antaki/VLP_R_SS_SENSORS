import numpy as np
from sklearn.metrics import mean_squared_error
import time

def calculate_rmse(y_true, y_pred):
    """Calculate RMSE for regression evaluation"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def time_execution(func):
    """Decorator to measure execution time"""
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        if isinstance(result, dict):
            result['runtime'] = execution_time
        return result, execution_time
    return wrapper
