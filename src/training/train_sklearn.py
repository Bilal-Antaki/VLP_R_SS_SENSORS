# src/training/train_enhanced.py
from sklearn.model_selection import train_test_split
from src.data.data_loader import load_cir_data, extract_features_and_target
from src.evaluation.metrics import calculate_rmse
from sklearn.preprocessing import StandardScaler
from src.models.model_registry import get_model
from src.config import DATA_CONFIG
import numpy as np

def train_all_models_enhanced(processed_dir: str, test_size: float = 0.2):
    """
    Train and compare Linear and SVR models
    """
    print("Training sklearn models...")
    
    # Load data
    df = load_cir_data(processed_dir)
    X, y = extract_features_and_target(df)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {list(X.columns)}")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Define models to test
    models = [
        ('Linear', 'linear'),
        ('SVR', 'svr')
    ]
    
    results = []
    
    for model_name, model_type in models:
        print(f"Training {model_name}...")
        
        try:
            # Get and train model
            model = get_model(model_type)
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Calculate RMSE
            rmse = calculate_rmse(y_test, y_pred)
            
            print(f"{model_name} RMSE: {rmse:.4f}")
            
            results.append({
                'success': True,
                'name': model_name,
                'y_pred': y_pred,
                'y_test': y_test,
                'metrics': {'rmse': rmse}
            })
            
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            results.append({
                'success': False,
                'name': model_name,
                'error': str(e)
            })
    
    return results