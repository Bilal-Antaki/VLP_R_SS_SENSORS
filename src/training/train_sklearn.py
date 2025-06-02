# src/training/train_enhanced.py
from src.models.model_registry import get_model
from src.data.loader import load_cir_data, extract_features_and_target
from src.evaluation.metrics import calculate_all_metrics
from src.config import DATA_CONFIG
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

def train_model_with_metrics(model_name, X_train, X_test, y_train, y_test, **model_kwargs):
    """Train a model and calculate comprehensive metrics"""
    start_time = time.time()
    
    try:
        # Get and train model
        model = get_model(model_name, **model_kwargs)
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Training time
        train_time = time.time() - start_time
        
        # Calculate metrics
        metrics = calculate_all_metrics(y_test, y_pred, model_name)
        metrics['train_time'] = train_time
        
        # Cross-validation if model supports it
        try:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                      scoring='neg_mean_squared_error', n_jobs=-1)
            metrics['cv_rmse'] = np.sqrt(-cv_scores.mean())
            metrics['cv_std'] = np.sqrt(cv_scores.std())
        except:
            metrics['cv_rmse'] = np.nan
            metrics['cv_std'] = np.nan
        
        return {
            'success': True,
            'model': model,
            'y_pred': y_pred,
            'y_test': y_test,
            'metrics': metrics,
            'name': model_name
        }
        
    except Exception as e:
        print(f"Error training {model_name}: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'name': model_name
        }

def train_all_models_enhanced(processed_dir: str, test_size: float = 0.2, 
                            include_slow_models: bool = False,
                            include_deep_learning: bool = False):
    """
    Train and compare Linear and SVR models with comprehensive metrics
    """
    print("Enhanced Model Training and Evaluation")
    print("=" * 60)
    
    # Load data
    print("Loading data...")
    df = load_cir_data(processed_dir, filter_keyword=DATA_CONFIG['datasets'][0])
    print(f"Using dataset: {DATA_CONFIG['datasets'][0]}")
    X, y = extract_features_and_target(df)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {list(X.columns)}")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"Target mean: {y.mean():.2f}, std: {y.std():.2f}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models to test - ONLY Linear and SVR variants
    model_configs = [
        # Linear models
        ('Linear Regression', 'linear', {}, False),
        
        # SVM models
        ('SVR RBF', 'svr', {}, False)
    ]
    
    # Train all models
    results = []
    successful_results = []
    
    print(f"\nTraining model configurations...")
    print("-" * 60)
    
    for display_name, model_name, kwargs, needs_scaling in model_configs:
        print(f"\nTraining {display_name}...", end=' ', flush=True)
        
        # Use scaled or unscaled data
        X_tr = X_train_scaled if needs_scaling else X_train
        X_te = X_test_scaled if needs_scaling else X_test
        
        result = train_model_with_metrics(
            model_name, X_tr, X_te, y_train, y_test, **kwargs
        )
        
        if result['success']:
            # Format time properly
            train_time = result['metrics']['train_time']
            if train_time < 1:
                time_str = f"{train_time:.3f}s"
            else:
                time_str = f"{train_time:.2f}s"
            
            print(f"✓ RMSE: {result['metrics']['rmse']:.4f}, Time: {time_str}")
            results.append(result)
            successful_results.append({
                'name': display_name,
                'y_true': y_test,
                'y_pred': result['y_pred']
            })
        else:
            print(f"✗ Failed: {result['error']}")
    
    if not results:
        print("\nNo models trained successfully!")
        return None, None, None
    
    return results