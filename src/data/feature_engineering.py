# src/data/feature_engineering.py
import numpy as np
import pandas as pd

def create_engineered_features(df, features=['PL', 'RMS'], include_categorical=True):
    """
    Create engineered features for better model performance
    
    Args:
        df: Input DataFrame with at least PL and RMS columns
        features: Base features to use
        include_categorical: Whether to include categorical interaction features
        
    Returns:
        DataFrame with engineered features
    """
    feature_df = df.copy()
    
    # use PL and RMS features
    if 'PL' in df.columns and 'RMS' in df.columns:
        # Ratio features
        feature_df['PL_RMS_ratio'] = (df['PL'] / (df['RMS'] + 1e-10)).round(3)
        feature_df['RMS_PL_ratio'] = (df['RMS'] / (df['PL'] + 1e-10)).round(3)
        
        # Product features
        feature_df['PL_RMS_product'] = (df['PL'] * df['RMS']).round(3)
        
        # Difference features
        feature_df['PL_RMS_diff'] = (df['PL'] - df['RMS']).round(3)
        feature_df['PL_RMS_abs_diff'] = np.abs(df['PL'] - df['RMS']).round(3)
        
        # Power features
        feature_df['PL_squared'] = (df['PL'] ** 2).round(3)
        feature_df['RMS_squared'] = (df['RMS'] ** 2).round(3)
        feature_df['PL_sqrt'] = np.sqrt(np.abs(df['PL'])).round(3)
        feature_df['RMS_sqrt'] = np.sqrt(np.abs(df['RMS'])).round(3)
        
        # Log features (handle negative values)
        feature_df['PL_log'] = np.log1p(np.abs(df['PL'])).round(3)
        feature_df['RMS_log'] = np.log1p(np.abs(df['RMS'])).round(3)
        
        # Exponential features (scaled to prevent overflow)
        feature_df['PL_exp'] = np.exp(df['PL'] / 100).round(3)
        feature_df['RMS_exp'] = np.exp(df['RMS'] / 10).round(3)

        # Cubic and higher-order polynomial features
        feature_df['PL_cubed'] = (df['PL'] ** 3).round(3)
        feature_df['RMS_cubed'] = (df['RMS'] ** 3).round(3)
        feature_df['PL_fourth'] = (df['PL'] ** 4).round(3)
        feature_df['RMS_fourth'] = (df['RMS'] ** 4).round(3)
        # Inverse features (avoid division by zero)
        feature_df['PL_inv'] = (1.0 / (df['PL'] + 1e-10)).round(6)
        feature_df['RMS_inv'] = (1.0 / (df['RMS'] + 1e-10)).round(6)
        # Combined log-ratio
        feature_df['log_PL_RMS_ratio'] = (np.log1p(np.abs(df['PL'] / (df['RMS'] + 1e-10)))).round(3)
        # Min/max between PL and RMS
        feature_df['PL_RMS_min'] = np.minimum(df['PL'], df['RMS']).round(3)
        feature_df['PL_RMS_max'] = np.maximum(df['PL'], df['RMS']).round(3)
        # Standardized features
        feature_df['PL_standardized'] = ((df['PL'] - df['PL'].mean()) / (df['PL'].std() + 1e-10)).round(3)
        feature_df['RMS_standardized'] = ((df['RMS'] - df['RMS'].mean()) / (df['RMS'].std() + 1e-10)).round(3)
        # Interaction: sum and absolute sum
        feature_df['PL_RMS_sum'] = (df['PL'] + df['RMS']).round(3)
        feature_df['PL_RMS_abs_sum'] = (np.abs(df['PL']) + np.abs(df['RMS'])).round(3)
        # Interaction: difference of squares
        feature_df['PL2_minus_RMS2'] = (df['PL']**2 - df['RMS']**2).round(3)
        feature_df['RMS2_minus_PL2'] = (df['RMS']**2 - df['PL']**2).round(3)
    
    # Statistical features
    if 'source_file' in df.columns:
        # Add group statistics
        for feature in features:
            if feature in df.columns:
                group_stats = df.groupby('source_file')[feature].agg(['mean', 'std', 'min', 'max'])
                feature_df[f'{feature}_group_mean'] = df['source_file'].map(group_stats['mean']).round(3)
                feature_df[f'{feature}_group_std'] = df['source_file'].map(group_stats['std']).round(3)
                feature_df[f'{feature}_normalized'] = (
                    (df[feature] - feature_df[f'{feature}_group_mean']) / 
                    (feature_df[f'{feature}_group_std'] + 1e-10)
                ).round(3)
    
    # Interaction features
    if include_categorical and 'PL' in df.columns and 'RMS' in df.columns:
        # Binned interactions
        try:
            pl_bins = pd.qcut(df['PL'], q=5, labels=['VL', 'L', 'M', 'H', 'VH'])
            rms_bins = pd.qcut(df['RMS'], q=5, labels=['VL', 'L', 'M', 'H', 'VH'])
            feature_df['PL_RMS_interaction'] = pl_bins.astype(str) + '_' + rms_bins.astype(str)
            
            # Convert to dummy variables
            interaction_dummies = pd.get_dummies(feature_df['PL_RMS_interaction'], prefix='interaction')
            feature_df = pd.concat([feature_df, interaction_dummies], axis=1)
            
            # Drop the original categorical column
            feature_df = feature_df.drop('PL_RMS_interaction', axis=1)
        except:
            # Skip if binning fails
            pass
    
    # Remove any coordinate-based features
    for col in ['X', 'Y']:
        if col in feature_df.columns:
            feature_df = feature_df.drop(col, axis=1)

    # Round the original features
    feature_df['r'] = feature_df['r'].round(3)
    feature_df['PL'] = feature_df['PL'].round(3)
    feature_df['RMS'] = feature_df['RMS'].round(3)

    # Save with float_format to ensure all numbers are rounded
    feature_df.to_csv(r'data\processed\feature_df.csv', header=True, index=True, float_format='%.3f')
    
    return feature_df

def select_features(X, y, method='lasso', threshold=0.1, remove_intercorrelated=True, intercor_threshold=0.9):
    """
    Select relevant features based on correlation or importance
    Optionally remove highly intercorrelated features before selection.
    Args:
        X: Feature DataFrame
        y: Target values
        method: 'correlation', 'mutual_info', 'rf', 'lasso', or 'pca'
        threshold: Threshold for feature selection
        remove_intercorrelated: Whether to remove highly intercorrelated features
        intercor_threshold: Correlation threshold for removing features
    Returns:
        List of selected feature names (or principal component names for PCA)
    """
    # First, identify numeric columns only
    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove any coordinate-based features
    coordinate_cols = ['X', 'Y', 'r']
    numeric_columns = [col for col in numeric_columns if col not in coordinate_cols]
    
    X_numeric = X[numeric_columns]
    
    if len(numeric_columns) == 0:
        raise ValueError("No numeric features found in X")
    
    # Remove highly intercorrelated features before selection
    if remove_intercorrelated and len(numeric_columns) > 1:
        corr_matrix = X_numeric.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > intercor_threshold)]
        numeric_columns = [f for f in numeric_columns if f not in to_drop]
        X_numeric = X[numeric_columns]
    
    if method == 'correlation':
        # Calculate correlation with target (numeric features only)
        correlations = X_numeric.corrwith(pd.Series(y)).abs()
        selected_features = correlations[correlations > threshold].index.tolist()

    elif method == 'rf':
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_numeric, y)
        importances = rf.feature_importances_
        selected_features = X_numeric.columns[importances > threshold].tolist()

    elif method == 'mutual_info':
        from sklearn.feature_selection import mutual_info_regression
        mi_scores = mutual_info_regression(X_numeric, y)
        mi_df = pd.DataFrame({'feature': X_numeric.columns, 'mi_score': mi_scores})
        mi_df = mi_df.sort_values('mi_score', ascending=False)
        selected_features = mi_df[mi_df['mi_score'] > threshold]['feature'].tolist()

    elif method == 'lasso':
        from sklearn.linear_model import Lasso
        lasso = Lasso(alpha=0.1)
        lasso.fit(X_numeric, y)
        importances = lasso.coef_
        selected_features = X_numeric.columns[importances > threshold].tolist()

    elif method == 'pca':
        from sklearn.decomposition import PCA
        pca = PCA(n_components=0.98)
        X_pca = pca.fit_transform(X_numeric)
        n_components = X_pca.shape[1]
        # Return names as PC1, PC2, ...
        selected_features = [f'PC{i+1}' for i in range(n_components)]
        # Explained variance ratio for each PC
        print("Explained variance ratio for each PC:", pca.explained_variance_ratio_)

        # Cumulative explained variance
        print("Cumulative explained variance:", pca.explained_variance_ratio_.cumsum())

        # Return both names and transformed data
        return selected_features, X_pca

    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Always include original features PL and RMS if present in X
    for feat in ['PL', 'RMS']:
        if feat in X.columns and feat not in selected_features:
            selected_features.append(feat)
    
    return selected_features

def select_top_k_uncorrelated_features(X, y, method='correlation', k=5, threshold=0.1, corr_threshold=0.9):
    """
    Select the top k features by a method, ensuring low inter-correlation between selected features.
    Args:
        X: Feature DataFrame
        y: Target values
        method: 'correlation', 'mutual_info', 'rf', or 'lasso'
        k: number of features to select
        threshold: threshold for feature selection (importance/correlation)
        corr_threshold: maximum allowed absolute correlation between selected features
    Returns:
        List of selected feature names
    """
    import numpy as np
    import pandas as pd
    # Identify numeric columns only
    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    coordinate_cols = ['X', 'Y', 'r']
    numeric_columns = [col for col in numeric_columns if col not in coordinate_cols]
    X_numeric = X[numeric_columns]
    if len(numeric_columns) == 0:
        raise ValueError("No numeric features found in X")

    # Get feature scores
    if method == 'correlation':
        scores = X_numeric.corrwith(pd.Series(y)).abs()
    elif method == 'rf':
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_numeric, y)
        scores = pd.Series(rf.feature_importances_, index=X_numeric.columns)
    elif method == 'mutual_info':
        from sklearn.feature_selection import mutual_info_regression
        mi_scores = mutual_info_regression(X_numeric, y)
        scores = pd.Series(mi_scores, index=X_numeric.columns)
    elif method == 'lasso':
        from sklearn.linear_model import Lasso
        lasso = Lasso(alpha=0.1)
        lasso.fit(X_numeric, y)
        scores = pd.Series(lasso.coef_, index=X_numeric.columns).abs()
    else:
        raise ValueError(f"Unknown method: {method}")

    # Sort features by score, descending
    sorted_features = scores.sort_values(ascending=False).index.tolist()
    selected = []
    for feat in sorted_features:
        if len(selected) >= k:
            break
        # Check correlation with already selected features
        if selected:
            corrs = X_numeric[selected].corrwith(X_numeric[feat]).abs()
            if (corrs > corr_threshold).any():
                continue  # skip if too correlated
        selected.append(feat)
    return selected