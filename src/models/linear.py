from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from ..config import TRAINING_CONFIG

def build_linear_model(model_type='linear', **kwargs):
    """
    Args:
        model_type: 'linear'
        **kwargs: model-specific parameters
    """
    if model_type == 'linear':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('linear', LinearRegression(**kwargs))
        ])
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

# Backwards compatibility
def build_linear_model_simple(**kwargs):
    """Simple linear regression for backwards compatibility"""
    return build_linear_model(model_type='linear', **kwargs)