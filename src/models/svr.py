from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from ..config import TRAINING_CONFIG

def build_svr_model(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale', **kwargs):
    """
    Build SVR model with different kernels and parameters
    
    Args:
        kernel: 'linear', 'poly', 'rbf', 'sigmoid'
        C: Regularization parameter
        epsilon: Epsilon in the epsilon-SVR model
        gamma: Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
        **kwargs: Additional SVR parameters
    """
    
    # Build SVR with specified parameters
    svr_params = {
        'kernel': kernel,
        'C': C,
        'epsilon': epsilon,
        'gamma': gamma,
        **kwargs
    }
    
    # SVR benefits greatly from feature scaling, so include it in pipeline
    return Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(**svr_params))
    ])

def build_svr_optimized(**kwargs):
    """
    Build SVR with optimized default parameters for position estimation
    """
    # Use weight decay from training config as inverse of C parameter
    C = 1.0 / TRAINING_CONFIG['weight_decay'] if TRAINING_CONFIG['weight_decay'] > 0 else 100.0
    
    return build_svr_model(
        kernel='rbf',
        C=C,           # Use C derived from weight decay
        epsilon=0.01,  # Lower epsilon for better fit
        gamma='auto',  # Auto-select gamma
        **kwargs
    )