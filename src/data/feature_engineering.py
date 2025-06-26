# src/data/feature_engineering.py
import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

def create_engineered_features(df):
    """
    Create engineered features from PL and RMS
    
    Args:
        df: Input DataFrame with PL and RMS columns
        
    Returns:
        DataFrame with engineered features
    """
    if 'PL' not in df.columns or 'RMS' not in df.columns:
        print("Warning: PL and/or RMS columns not found. Returning original DataFrame.")
        return df
    
    feature_df = df.copy()
    
    # Clean base features
    feature_df['PL'] = pd.to_numeric(feature_df['PL'], errors='coerce').round(3)
    feature_df['RMS'] = pd.to_numeric(feature_df['RMS'], errors='coerce').round(3)
    
    # Create features
    _create_features(feature_df)
    
    # Remove coordinate columns to prevent data leakage
    coordinate_cols = ['X', 'Y', 'r']
    for col in coordinate_cols:
        if col in feature_df.columns:
            feature_df = feature_df.drop(col, axis=1)
    
    print(f"Created {len(feature_df.columns)} total features")
    return feature_df

def _create_features(feature_df):
    """Create all engineered features"""
    pl = feature_df['PL']
    rms = feature_df['RMS']
    
    # Basic arithmetic
    feature_df['PL_RMS_sum'] = (pl + rms).round(3)
    feature_df['PL_RMS_diff'] = (pl - rms).round(3)
    feature_df['PL_RMS_product'] = (pl * rms).round(3)
    
    # Safe ratios
    feature_df['PL_RMS_ratio'] = np.where(np.abs(rms) > 1e-6, pl / rms, 0).round(3)
    feature_df['RMS_PL_ratio'] = np.where(np.abs(pl) > 1e-6, rms / pl, 0).round(3)
    
    # Min/Max
    feature_df['PL_RMS_min'] = np.minimum(pl, rms).round(3)
    feature_df['PL_RMS_max'] = np.maximum(pl, rms).round(3)
    
    # Powers
    feature_df['PL_squared'] = (pl ** 2).round(3)
    feature_df['RMS_squared'] = (rms ** 2).round(3)
    feature_df['PL_cubed'] = (pl ** 3).round(3)
    feature_df['RMS_cubed'] = (rms ** 3).round(3)
    
    # Square roots (handle negatives)
    feature_df['PL_sqrt'] = np.where(pl >= 0, np.sqrt(pl), -np.sqrt(-pl)).round(3)
    feature_df['RMS_sqrt'] = np.where(rms >= 0, np.sqrt(rms), -np.sqrt(-rms)).round(3)
    
    # Safe logs
    feature_df['PL_log'] = np.where(pl > 0, np.log(pl), 
                                   np.where(pl < 0, -np.log(-pl), 0)).round(3)
    feature_df['RMS_log'] = np.where(rms > 0, np.log(rms), 
                                    np.where(rms < 0, -np.log(-rms), 0)).round(3)
    
    # Safe inverses
    feature_df['PL_inv'] = np.where(np.abs(pl) > 1e-6, 1.0 / pl, 0).round(6)
    feature_df['RMS_inv'] = np.where(np.abs(rms) > 1e-6, 1.0 / rms, 0).round(6)
    
    # Combined features
    feature_df['PL2_minus_RMS2'] = (pl**2 - rms**2).round(3)
    feature_df['log_PL_RMS_ratio'] = np.where((pl > 0) & (rms > 0), 
                                             np.log(pl / rms), 0).round(3)

def select_features_rf(X, y, k=5):
    """
    Select k best features using Random Forest
    
    Args:
        X: Feature DataFrame
        y: Target values
        k: Number of features to select
        
    Returns:
        List of selected feature names
    """
    # Get numeric features only
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['X', 'Y', 'r', 'source_file']
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    if len(numeric_cols) == 0:
        raise ValueError("No valid features found")
    
    X_clean = X[numeric_cols].fillna(0)
    
    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_clean, y)
    
    # Get feature importance
    importance = pd.Series(rf.feature_importances_, index=X_clean.columns)
    selected_features = importance.nlargest(k).index.tolist()
    
    print(f"Selected top {k} features by Random Forest:")
    
    return selected_features

def select_features_pca(X, y, variance_threshold=0.95):
    """
    Select PCA components that explain variance_threshold of total variance
    
    Args:
        X: Feature DataFrame
        y: Target values (not used but kept for consistency)
        variance_threshold: Minimum cumulative variance to explain (e.g., 0.95 = 95%)
        
    Returns:
        Tuple of (component_names, transformed_data)
    """
    # Get numeric features only
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['X', 'Y', 'r', 'source_file']
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    if len(numeric_cols) == 0:
        raise ValueError("No valid features found")
    
    X_clean = X[numeric_cols].fillna(0).replace([np.inf, -np.inf], [1e6, -1e6])
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    
    # Apply PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # Find number of components for desired variance
    cumulative_variance = pca.explained_variance_ratio_.cumsum()
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    
    # Component names
    component_names = [f'PC{i+1}' for i in range(n_components)]
    
    print(f"Selected {n_components} PCA components explaining {cumulative_variance[n_components-1]:.3f} of variance:")
    for i in range(n_components):
        print(f"  PC{i+1}: {pca.explained_variance_ratio_[i]:.3f} variance")
    
    return component_names, X_pca[:, :n_components]

# Convenience functions
def select_features(X, y, method='rf', **kwargs):
    """
    Select features using specified method
    
    Args:
        X: Feature DataFrame
        y: Target values
        method: 'rf' for Random Forest, 'pca' for PCA, 'correlation' for correlation-based
        **kwargs: Additional arguments for the specific method
        
    Returns:
        List of selected feature names or component names
    """
    if method.lower() == 'rf':
        k = kwargs.get('k', 5)
        return select_features_rf(X, y, k)
    elif method.lower() == 'pca':
        variance_threshold = kwargs.get('variance_threshold', 0.95)
        return select_features_pca(X, y, variance_threshold)
    elif method.lower() == 'correlation':
        threshold = kwargs.get('threshold', 0.1)
        return select_features_correlation(X, y, threshold)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'rf', 'pca', or 'correlation'")

def select_features_correlation(X, y, threshold=0.1):
    """
    Select features based on correlation with target
    
    Args:
        X: Feature DataFrame
        y: Target values
        threshold: Minimum absolute correlation to include feature
        
    Returns:
        List of selected feature names
    """
    # Get numeric features only
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['X', 'Y', 'r', 'source_file']
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    if len(numeric_cols) == 0:
        raise ValueError("No valid features found")
    
    X_clean = X[numeric_cols].fillna(0)
    
    # Calculate correlations
    correlations = X_clean.corrwith(y).abs()
    selected_features = correlations[correlations >= threshold].index.tolist()
    
    print(f"Selected {len(selected_features)} features with correlation >= {threshold}:")
    for feature in selected_features:
        print(f"  {feature}: {correlations[feature]:.3f}")
    
    return selected_features

def get_best_features_rf(df, target_col='r', k=5):
    """Get k best features using Random Forest"""
    X = df.drop([target_col], axis=1, errors='ignore')
    y = df[target_col]
    return select_features_rf(X, y, k)

def get_best_features_pca(df, target_col='r', variance_threshold=0.95):
    """Get PCA components explaining desired variance"""
    X = df.drop([target_col], axis=1, errors='ignore')
    y = df[target_col]
    return select_features_pca(X, y, variance_threshold)
