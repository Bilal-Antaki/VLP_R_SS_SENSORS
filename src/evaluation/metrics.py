# src/evaluation/metrics.py
import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, 
    explained_variance_score, mean_absolute_percentage_error
)
from scipy import stats
import pandas as pd

def calculate_all_metrics(y_true, y_pred, model_name="Model"):
    """
    Calculate comprehensive metrics for regression evaluation
    
    Args:
        y_true: True values
        y_pred: Predicted values
        model_name: Name of the model for display
        
    Returns:
        Dictionary of metrics
    """
    # Ensure arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Basic metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    ev = explained_variance_score(y_true, y_pred)
    
    # Additional metrics
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100  # Convert to percentage
    
    # Custom metrics
    errors = y_true - y_pred
    abs_errors = np.abs(errors)
    
    # Percentile errors
    p50_error = np.percentile(abs_errors, 50)  # Median absolute error
    p90_error = np.percentile(abs_errors, 90)
    p95_error = np.percentile(abs_errors, 95)
    
    # Maximum error
    max_error = np.max(abs_errors)
    
    # Relative errors
    relative_errors = abs_errors / (y_true + 1e-10)  # Add small value to avoid division by zero
    mean_relative_error = np.mean(relative_errors) * 100
    
    # Statistical tests
    # Test if residuals are normally distributed (Shapiro-Wilk test)
    if len(errors) > 3:
        _, normality_p_value = stats.shapiro(errors)
    else:
        normality_p_value = np.nan
    
    # Create metrics dictionary
    metrics = {
        'model_name': model_name,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'explained_variance': ev,
        'mape': mape,
        'median_abs_error': p50_error,
        'p90_error': p90_error,
        'p95_error': p95_error,
        'max_error': max_error,
        'mean_relative_error': mean_relative_error,
        'residual_std': np.std(errors),
        'residual_normality_p': normality_p_value
    }
    
    return metrics

def compare_models(results_list):
    """
    Compare multiple models and create a comparison DataFrame
    
    Args:
        results_list: List of dictionaries with 'name', 'y_true', and 'y_pred'
        
    Returns:
        DataFrame with model comparison
    """
    all_metrics = []
    
    for result in results_list:
        metrics = calculate_all_metrics(
            result['y_true'], 
            result['y_pred'], 
            result['name']
        )
        all_metrics.append(metrics)
    
    # Create DataFrame
    df = pd.DataFrame(all_metrics)
    
    # Sort by RMSE
    df = df.sort_values('rmse')
    
    # Add rank columns
    for metric in ['rmse', 'mae', 'mape']:
        df[f'{metric}_rank'] = df[metric].rank()
    
    return df

def print_metrics_report(metrics_dict):
    """
    Print a formatted metrics report
    
    Args:
        metrics_dict: Dictionary of metrics from calculate_all_metrics
    """
    print(f"\n{'='*50}")
    print(f"METRICS REPORT: {metrics_dict['model_name']}")
    print(f"{'='*50}")
    
    print("\nPrimary Metrics:")
    print(f"  RMSE: {metrics_dict['rmse']:.4f}")
    print(f"  MAE:  {metrics_dict['mae']:.4f}")
    print(f"  R²:   {metrics_dict['r2']:.4f}")
    print(f"  MAPE: {metrics_dict['mape']:.2f}%")
    
    print("\nError Distribution:")
    print(f"  Median Abs Error: {metrics_dict['median_abs_error']:.4f}")
    print(f"  90th Percentile:  {metrics_dict['p90_error']:.4f}")
    print(f"  95th Percentile:  {metrics_dict['p95_error']:.4f}")
    print(f"  Maximum Error:    {metrics_dict['max_error']:.4f}")
    
    print("\nStatistical Properties:")
    print(f"  Residual Std Dev: {metrics_dict['residual_std']:.4f}")
    print(f"  Mean Relative Error: {metrics_dict['mean_relative_error']:.2f}%")
    
    if not np.isnan(metrics_dict['residual_normality_p']):
        normality = "Yes" if metrics_dict['residual_normality_p'] > 0.05 else "No"
        print(f"  Residuals Normal: {normality} (p={metrics_dict['residual_normality_p']:.4f})")
    
    print(f"{'='*50}\n")

def calculate_confidence_intervals(y_true, y_pred, confidence=0.95):
    """
    Calculate confidence intervals for predictions
    
    Args:
        y_true: True values
        y_pred: Predicted values
        confidence: Confidence level (default: 0.95)
        
    Returns:
        Dictionary with confidence interval information
    """
    errors = y_true - y_pred
    n = len(errors)
    
    # Calculate standard error
    se = np.std(errors) / np.sqrt(n)
    
    # Calculate confidence interval
    alpha = 1 - confidence
    t_stat = stats.t.ppf(1 - alpha/2, df=n-1)
    margin = t_stat * se
    
    return {
        'mean_error': np.mean(errors),
        'std_error': se,
        'confidence_level': confidence,
        'margin_of_error': margin,
        'ci_lower': np.mean(errors) - margin,
        'ci_upper': np.mean(errors) + margin
    }

def cross_validation_metrics(cv_scores):
    """
    Calculate statistics from cross-validation scores
    
    Args:
        cv_scores: Array of cross-validation scores
        
    Returns:
        Dictionary with CV statistics
    """
    return {
        'cv_mean': np.mean(cv_scores),
        'cv_std': np.std(cv_scores),
        'cv_min': np.min(cv_scores),
        'cv_max': np.max(cv_scores),
        'cv_range': np.max(cv_scores) - np.min(cv_scores)
    }