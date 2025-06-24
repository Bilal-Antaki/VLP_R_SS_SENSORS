# src/data/feature_engineering.py
import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

def create_engineered_features(df, features=['PL', 'RMS'], include_categorical=True):
    """
    Create engineered features for better model performance
    
    Args:
        df: Input DataFrame with at least PL and RMS columns
        features: Base features to use
        include_categorical: Whether to include categorical interaction features
        complexity_level: 'simple', 'moderate', or 'complex' - controls feature complexity
        
    Returns:
        DataFrame with engineered features
    """
    feature_df = df.copy()
    
    # Clean and validate base features
    feature_df['PL'] = pd.to_numeric(feature_df['PL'], errors='coerce')
    feature_df['RMS'] = pd.to_numeric(feature_df['RMS'], errors='coerce')
    
    # Round base features
    feature_df['PL'] = feature_df['PL'].round(3)
    feature_df['RMS'] = feature_df['RMS'].round(3)
    
    _create_features(feature_df)
    
    # Statistical features if source_file information is available
    if 'source_file' in df.columns:
        _create_statistical_features(feature_df, features)
    
    # Categorical interaction features
    if include_categorical:
        _create_categorical_features(feature_df)
    
    # Remove coordinate-based features to prevent data leakage
    coordinate_cols = ['X', 'Y']
    for col in coordinate_cols:
        if col in feature_df.columns:
            feature_df = feature_df.drop(col, axis=1)
    
    # Final data quality check
    feature_df = _clean_features(feature_df)
    
    print(f"Created {len(feature_df.columns)} total features")
    
    return feature_df

def _create_features(feature_df):
    """Create basic ratio and interaction features"""
    pl = feature_df['PL']
    rms = feature_df['RMS']
    
    # Safe division with proper zero handling
    feature_df['PL_RMS_ratio'] = np.where(
        np.abs(rms) > 1e-6, 
        pl / rms, 
        np.sign(pl) * 1000  # Large value with same sign as PL
    ).round(3)
    
    feature_df['RMS_PL_ratio'] = np.where(
        np.abs(pl) > 1e-6, 
        rms / pl, 
        np.sign(rms) * 1000  # Large value with same sign as RMS
    ).round(3)
    
    # Basic arithmetic operations
    feature_df['PL_RMS_sum'] = (pl + rms).round(3)
    feature_df['PL_RMS_diff'] = (pl - rms).round(3)
    feature_df['PL_RMS_abs_diff'] = np.abs(pl - rms).round(3)
    feature_df['PL_RMS_product'] = (pl * rms).round(3)
    
    # Min and max
    feature_df['PL_RMS_min'] = np.minimum(pl, rms).round(3)
    feature_df['PL_RMS_max'] = np.maximum(pl, rms).round(3)
    
    # Absolute values
    feature_df['PL_abs'] = np.abs(pl).round(3)
    feature_df['RMS_abs'] = np.abs(rms).round(3)
    
    # Squared features (most important polynomial)
    feature_df['PL_squared'] = (pl ** 2).round(3)
    feature_df['RMS_squared'] = (rms ** 2).round(3)
    
    # Square root features (handle negatives)
    feature_df['PL_sqrt'] = np.where(
        pl >= 0, 
        np.sqrt(pl), 
        -np.sqrt(np.abs(pl))
    ).round(3)
    
    feature_df['RMS_sqrt'] = np.where(
        rms >= 0, 
        np.sqrt(rms), 
        -np.sqrt(np.abs(rms))
    ).round(3)
    
    # Logarithmic features (safe log)
    feature_df['PL_log'] = np.where(
        pl > 0,
        np.log(pl),
        np.where(pl < 0, -np.log(-pl), 0)
    ).round(3)
    
    feature_df['RMS_log'] = np.where(
        rms > 0,
        np.log(rms),
        np.where(rms < 0, -np.log(-rms), 0)
    ).round(3)
    
    # Combined squared difference
    feature_df['PL2_minus_RMS2'] = (pl**2 - rms**2).round(3)
    
    # Harmonic mean (safe)
    feature_df['PL_RMS_harmonic'] = np.where(
        (pl != 0) & (rms != 0),
        2 * pl * rms / (pl + rms),
        0
    ).round(3)
    
    # Higher order polynomials (limited to avoid overfitting)
    feature_df['PL_cubed'] = (pl ** 3).round(3)
    feature_df['RMS_cubed'] = (rms ** 3).round(3)
    
    # Exponential features (scaled to prevent overflow)
    feature_df['PL_exp_scaled'] = np.exp(np.clip(pl / 100, -10, 10)).round(3)
    feature_df['RMS_exp_scaled'] = np.exp(np.clip(rms / 100, -10, 10)).round(3)
    
    # Inverse features (safe)
    feature_df['PL_inv'] = np.where(
        np.abs(pl) > 1e-6,
        1.0 / pl,
        np.sign(pl) * 1000  # Large value with appropriate sign
    ).round(6)
    
    feature_df['RMS_inv'] = np.where(
        np.abs(rms) > 1e-6,
        1.0 / rms,
        np.sign(rms) * 1000  # Large value with appropriate sign
    ).round(6)
    
    # Log ratio (safe)
    feature_df['log_PL_RMS_ratio'] = np.where(
        (pl > 0) & (rms > 0),
        np.log(pl / rms),
        0
    ).round(3)
    
    # Standardized features (within this dataset)
    if len(feature_df) > 1:
        pl_std = pl.std()
        rms_std = rms.std()
        if pl_std > 1e-6:
            feature_df['PL_standardized'] = ((pl - pl.mean()) / pl_std).round(3)
        if rms_std > 1e-6:
            feature_df['RMS_standardized'] = ((rms - rms.mean()) / rms_std).round(3)

def _create_statistical_features(feature_df, features):
    """Create statistical features based on grouping"""
    for feature in features:
        if feature in feature_df.columns:
            try:
                # Group statistics by source file
                group_stats = feature_df.groupby('source_file')[feature].agg(['mean', 'std', 'min', 'max'])
                
                # Add group mean and std
                feature_df[f'{feature}_group_mean'] = feature_df['source_file'].map(group_stats['mean']).round(3)
                group_std = feature_df['source_file'].map(group_stats['std']).fillna(0)
                feature_df[f'{feature}_group_std'] = group_std.round(3)
                
                # Normalized features (deviation from group mean)
                feature_df[f'{feature}_normalized'] = np.where(
                    group_std > 1e-6,
                    (feature_df[feature] - feature_df[f'{feature}_group_mean']) / group_std,
                    0
                ).round(3)
                
            except Exception as e:
                print(f"Warning: Could not create group statistics for {feature}: {e}")

def _create_categorical_features(feature_df):
    """Create categorical interaction features"""
    try:
        # Create quantile-based bins for interactions
        pl_bins = pd.qcut(feature_df['PL'], q=3, labels=['Low', 'Med', 'High'], duplicates='drop')
        rms_bins = pd.qcut(feature_df['RMS'], q=3, labels=['Low', 'Med', 'High'], duplicates='drop')
        
        # Create interaction categories
        feature_df['PL_RMS_interaction'] = pl_bins.astype(str) + '_' + rms_bins.astype(str)
        
        # Convert to dummy variables (one-hot encoding)
        interaction_dummies = pd.get_dummies(feature_df['PL_RMS_interaction'], prefix='interaction')
        feature_df = pd.concat([feature_df, interaction_dummies], axis=1)
        
        # Drop the original categorical column
        feature_df = feature_df.drop('PL_RMS_interaction', axis=1)
        
    except Exception as e:
        print(f"Warning: Could not create categorical features: {e}")
    
    return feature_df

def _clean_features(feature_df):
    """Clean and validate all features"""
    # Replace infinite values with large but finite numbers
    feature_df = feature_df.replace([np.inf, -np.inf], [1e6, -1e6])
    
    # Fill any remaining NaN values with 0
    feature_df = feature_df.fillna(0)
    
    # Clip extreme values to prevent numerical issues
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'r':  # Don't clip the target variable
            q99 = feature_df[col].quantile(0.99)
            q1 = feature_df[col].quantile(0.01)
            feature_df[col] = feature_df[col].clip(lower=q1, upper=q99)
    
    return feature_df

def select_features(X, y, method='correlation', threshold=0.1, max_features=None, 
                   remove_intercorrelated=True, intercor_threshold=0.9):
    """
    Select relevant features using various methods
    
    Args:
        X: Feature DataFrame
        y: Target values
        method: 'correlation', 'mutual_info', 'rf', 'lasso', or 'pca'
        threshold: Threshold for feature selection
        max_features: Maximum number of features to select
        remove_intercorrelated: Whether to remove highly intercorrelated features
        intercor_threshold: Correlation threshold for removing features
        
    Returns:
        List of selected feature names (or transformed data for PCA)
    """
    print(f"Starting feature selection with method: {method}")
    
    # Identify numeric columns only
    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove coordinate-based features
    coordinate_cols = ['X', 'Y', 'r']
    numeric_columns = [col for col in numeric_columns if col not in coordinate_cols]
    
    if len(numeric_columns) == 0:
        raise ValueError("No numeric features found in X")
    
    X_numeric = X[numeric_columns].copy()
    
    print(f"Working with {len(numeric_columns)} numeric features")
    
    # Clean data
    X_numeric = X_numeric.fillna(0)
    X_numeric = X_numeric.replace([np.inf, -np.inf], [1e6, -1e6])
    
    # Remove highly intercorrelated features first
    if remove_intercorrelated and len(numeric_columns) > 1:
        X_numeric, numeric_columns = _remove_intercorrelated_features(
            X_numeric, numeric_columns, intercor_threshold
        )
    
    # Apply feature selection method
    if method == 'correlation':
        selected_features = _select_by_correlation(X_numeric, y, threshold)
    elif method == 'rf':
        selected_features = _select_by_random_forest(X_numeric, y, threshold)
    elif method == 'mutual_info':
        selected_features = _select_by_mutual_info(X_numeric, y, threshold)
    elif method == 'lasso':
        selected_features = _select_by_lasso(X_numeric, y, threshold)
    elif method == 'pca':
        return _select_by_pca(X_numeric, threshold)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Limit number of features if specified
    if max_features and len(selected_features) > max_features:
        # Re-rank by correlation and take top max_features
        correlations = X_numeric[selected_features].corrwith(pd.Series(y)).abs()
        selected_features = correlations.nlargest(max_features).index.tolist()
    
    # Always ensure PL and RMS are included if they exist and weren't filtered out
    essential_features = ['PL', 'RMS']
    for feat in essential_features:
        if feat in X.columns and feat not in selected_features:
            selected_features.append(feat)
    
    print(f"Selected {len(selected_features)} features: {selected_features[:10]}...")
    
    return selected_features

def _remove_intercorrelated_features(X_numeric, numeric_columns, threshold):
    """Remove highly intercorrelated features"""
    corr_matrix = X_numeric.corr().abs()
    
    # Find pairs of highly correlated features
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Find features to drop (keep the first occurrence)
    to_drop = [column for column in upper_triangle.columns 
               if any(upper_triangle[column] > threshold)]
    
    # Keep essential features
    essential_features = ['PL', 'RMS']
    to_drop = [col for col in to_drop if col not in essential_features]
    
    if to_drop:
        print(f"Removing {len(to_drop)} highly correlated features")
        X_numeric = X_numeric.drop(columns=to_drop)
        numeric_columns = [col for col in numeric_columns if col not in to_drop]
    
    return X_numeric, numeric_columns

def _select_by_correlation(X_numeric, y, threshold):
    """Select features based on correlation with target"""
    correlations = X_numeric.corrwith(pd.Series(y)).abs()
    selected_features = correlations[correlations > threshold].index.tolist()
    
    # If no features meet threshold, take top 5
    if len(selected_features) == 0:
        print(f"No features meet correlation threshold {threshold}, selecting top 5")
        selected_features = correlations.nlargest(5).index.tolist()
    
    return selected_features

def _select_by_random_forest(X_numeric, y, threshold):
    """Select features based on Random Forest importance"""
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_numeric, y)
    importances = pd.Series(rf.feature_importances_, index=X_numeric.columns)
    selected_features = importances[importances > threshold].index.tolist()
    
    # If no features meet threshold, take top 10
    if len(selected_features) == 0:
        print(f"No features meet RF importance threshold {threshold}, selecting top 10")
        selected_features = importances.nlargest(10).index.tolist()
    
    return selected_features

def _select_by_mutual_info(X_numeric, y, threshold):
    """Select features based on mutual information"""
    mi_scores = mutual_info_regression(X_numeric, y, random_state=42)
    mi_series = pd.Series(mi_scores, index=X_numeric.columns)
    selected_features = mi_series[mi_series > threshold].index.tolist()
    
    # If no features meet threshold, take top 10
    if len(selected_features) == 0:
        print(f"No features meet MI threshold {threshold}, selecting top 10")
        selected_features = mi_series.nlargest(10).index.tolist()
    
    return selected_features

def _select_by_lasso(X_numeric, y, threshold):
    """Select features based on Lasso coefficients"""
    # Scale features for Lasso
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)
    
    # Use lighter regularization
    lasso = Lasso(alpha=0.01, random_state=42)
    lasso.fit(X_scaled, y)
    
    # Get non-zero coefficients
    coefficients = pd.Series(np.abs(lasso.coef_), index=X_numeric.columns)
    selected_features = coefficients[coefficients > threshold].index.tolist()
    
    # If no features selected, use lighter regularization
    if len(selected_features) == 0:
        lasso = Lasso(alpha=0.001, random_state=42)
        lasso.fit(X_scaled, y)
        coefficients = pd.Series(np.abs(lasso.coef_), index=X_numeric.columns)
        selected_features = coefficients[coefficients > 0].index.tolist()
    
    return selected_features

def _select_by_pca(X_numeric, threshold):
    """Select principal components"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)
    
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # Select components that explain at least threshold variance
    cumvar = pca.explained_variance_ratio_.cumsum()
    n_components = np.argmax(cumvar >= threshold) + 1
    
    print(f"Selected {n_components} PCs explaining {cumvar[n_components-1]:.3f} variance")
    
    # Return both component names and transformed data
    component_names = [f'PC{i+1}' for i in range(n_components)]
    
    return component_names, X_pca[:, :n_components]

def select_top_k_features(X, y, method='correlation', k=10):
    """
    Select exactly k top features by a specified method
    
    Args:
        X: Feature DataFrame
        y: Target values
        method: Selection method
        k: Number of features to select
        
    Returns:
        List of selected feature names
    """
    # Get all numeric features
    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    coordinate_cols = ['X', 'Y', 'r']
    numeric_columns = [col for col in numeric_columns if col not in coordinate_cols]
    
    X_numeric = X[numeric_columns].fillna(0)
    X_numeric = X_numeric.replace([np.inf, -np.inf], [1e6, -1e6])
    
    if method == 'correlation':
        scores = X_numeric.corrwith(pd.Series(y)).abs()
    elif method == 'rf':
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_numeric, y)
        scores = pd.Series(rf.feature_importances_, index=X_numeric.columns)
    elif method == 'mutual_info':
        mi_scores = mutual_info_regression(X_numeric, y, random_state=42)
        scores = pd.Series(mi_scores, index=X_numeric.columns)
    elif method == 'lasso':
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_numeric)
        lasso = Lasso(alpha=0.01, random_state=42)
        lasso.fit(X_scaled, y)
        scores = pd.Series(np.abs(lasso.coef_), index=X_numeric.columns)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Sort and select top k
    selected_features = scores.nlargest(k).index.tolist()
    
    print(f"Selected top {k} features by {method}: {selected_features}")
    
    return selected_features
