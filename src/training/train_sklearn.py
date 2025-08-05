from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.data.data_loader import load_cir_data, extract_features_and_target
from src.evaluation.metrics import calculate_rmse, time_execution
from src.config import SVR_CONFIG

def build_linear_model(**kwargs):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('linear', LinearRegression(**kwargs))
    ])

def build_svr_model(**kwargs):
    svr_params = {
        'kernel': SVR_CONFIG['kernel'],
        'C': SVR_CONFIG['C'],
        'epsilon': SVR_CONFIG['epsilon'],
        'gamma': SVR_CONFIG['gamma'],
        **kwargs
    }
    
    return Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(**svr_params))
    ])



@time_execution
def train_sklearn_model(processed_dir: str, model_name: str, test_size: float = 0.2):
    """Train sklearn model"""
    df = load_cir_data(processed_dir)
    X, y = extract_features_and_target(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=False
    )
    
    if model_name == 'linear':
        model = build_linear_model()
    elif model_name == 'svr':
        model = build_svr_model()
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = calculate_rmse(y_test, y_pred)
    

    
    return {
        'name': model_name,
        'y_pred': y_pred,
        'y_test': y_test,
        'rmse': rmse
    }

def train_all_sklearn_models(processed_dir: str, test_size: float = 0.2):
    """Train and compare Linear and SVR models"""
    models = ['linear', 'svr']
    results = []
    
    for model_name in models:
        result, runtime = train_sklearn_model(processed_dir, model_name, test_size)
        result['runtime'] = runtime
        results.append(result)
    
    return results
