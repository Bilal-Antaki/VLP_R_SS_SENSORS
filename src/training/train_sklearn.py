from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.data.data_loader import load_cir_data, extract_features_and_target
from src.evaluation.metrics import calculate_rmse, time_execution

def build_linear_model(**kwargs):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('linear', LinearRegression(**kwargs))
    ])

def build_svr_model(kernel='rbf', C=100.0, epsilon=0.01, gamma='auto', **kwargs):
    svr_params = {
        'kernel': kernel,
        'C': C,
        'epsilon': epsilon,
        'gamma': gamma,
        **kwargs
    }
    
    return Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(**svr_params))
    ])

def tune_svr_hyperparameters(X_train, y_train):
    """Tune hyperparameters for SVR"""
    param_grid = {
        'svr__C': [0.1, 1.0, 10.0, 100.0, 1000.0],
        'svr__epsilon': [0.001, 0.01, 0.1, 1.0],
        'svr__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
        'svr__kernel': ['rbf', 'linear', 'poly']
    }
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR())
    ])
    
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_

@time_execution
def train_sklearn_model(processed_dir: str, model_name: str, test_size: float = 0.2, tune_hyperparams: bool = False):
    """Train sklearn model with optional hyperparameter tuning"""
    df = load_cir_data(processed_dir)
    X, y = extract_features_and_target(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=False
    )
    
    if model_name == 'linear':
        model = build_linear_model()
    elif model_name == 'svr':
        if tune_hyperparams:
            model = tune_svr_hyperparameters(X_train, y_train)
        else:
            model = build_svr_model()
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = calculate_rmse(y_test, y_pred)
    
    # Get best parameters if tuning was used
    best_params = None
    if tune_hyperparams and model_name == 'svr' and hasattr(model, 'get_params'):
        best_params = {k: v for k, v in model.get_params().items() if not k.startswith('scaler')}
    
    return {
        'name': model_name,
        'y_pred': y_pred,
        'y_test': y_test,
        'rmse': rmse,
        'best_params': best_params
    }

def train_all_sklearn_models(processed_dir: str, test_size: float = 0.2, tune_hyperparams: bool = False):
    """Train and compare Linear and SVR models"""
    models = ['linear', 'svr']
    results = []
    
    for model_name in models:
        result, runtime = train_sklearn_model(processed_dir, model_name, test_size, tune_hyperparams)
        result['runtime'] = runtime
        results.append(result)
    
    return results
