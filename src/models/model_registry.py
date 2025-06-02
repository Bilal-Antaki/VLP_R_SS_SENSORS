# src/models/model_registry.py
from .linear import build_linear_model_simple
from .svr import build_svr_model, build_svr_optimized
from .lstm import build_lstm_model
from .gru import build_gru_model, build_gru_with_attention, build_gru_bidirectional, build_gru_residual
#from .lstm_2 import build_lstm_model_2


MODEL_REGISTRY = {
    # Linear models
    "linear": build_linear_model_simple,
    
    # SVR models
    "svr": build_svr_optimized,
    "svr_rbf": lambda **kwargs: build_svr_model(kernel='rbf', **kwargs),
    
    # RNN models
    "lstm": build_lstm_model,
    "gru": build_gru_model,
    "gru_attention": build_gru_with_attention,
    "gru_bidirectional": build_gru_bidirectional,
    "gru_residual": build_gru_residual,
    #"lstm_2": build_lstm_model_2,
}

# Update the categories in list_available_models:
categories = {
    'Linear': ['linear'],
    'SVM': ['svr'],
    'RNN': ['lstm', 'gru', 'gru_attention', 'gru_bidirectional', 'gru_residual']
}


def get_model(name: str, **kwargs):
    """
    Get a model from the registry
    
    Args:
        name: Model name from MODEL_REGISTRY
        **kwargs: Model-specific parameters
        
    Returns:
        Configured model instance
    """
    name = name.lower()
    if name not in MODEL_REGISTRY:
        available_models = ', '.join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Model '{name}' not found in registry. Available models: {available_models}")
    
    return MODEL_REGISTRY[name](**kwargs)

def list_available_models():
    """List all available models grouped by category"""
    categories = {
        'Linear': ['linear'],
        'SVM': ['svr'],
        'RNN': ['lstm']
    }
    
    print("Available Models by Category:")
    print("=" * 50)
    for category, models in categories.items():
        print(f"\n{category}:")
        for model in models:
            if model in MODEL_REGISTRY:
                print(f"  - {model}")
    
    return list(MODEL_REGISTRY.keys())