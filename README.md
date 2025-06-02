# Position Estimation Using Channel Impulse Response (CIR) Data

An enhanced machine learning framework for position estimation using Path Loss (PL) and RMS delay spread features from wireless channel measurements.

## 🚀 Key Enhancements

### 1. **Expanded Model Library**
- **Linear Models**: Linear, Ridge, Lasso, ElasticNet, Polynomial Regression
- **Support Vector Machines**: SVR with RBF, Linear, and Polynomial kernels
- **Tree-Based Ensembles**: Random Forest, Gradient Boosting, Extra Trees, AdaBoost
- **Neural Networks**: MLP (sklearn & PyTorch), LSTM
- **Advanced Ensembles**: Voting, Bagging, XGBoost, LightGBM
- **Neighbor-Based**: KNN with various distance metrics

### 2. **Feature Engineering**
- Automatic feature creation (ratios, products, powers, logs)
- Domain-specific wireless propagation features
- Coordinate-based features (polar, Manhattan distance)
- Statistical features across datasets
- Feature selection methods (correlation, mutual information)

### 3. **Comprehensive Evaluation**
- Multiple metrics: RMSE, MAE, R², MAPE, percentile errors
- Cross-validation with confidence intervals
- Statistical significance testing
- Residual analysis and normality tests
- Model comparison frameworks

### 4. **Enhanced Visualization**
- Feature correlation heatmaps
- Model performance comparisons
- Residual plots
- Training history visualization
- 2D position heatmaps
- Error distribution analysis

## 📂 Project Structure

```
Position_Estimation_Code/
├── data/
│   ├── raw/           # Original PL and RMS CSV files
│   └── processed/     # Combined CIR CSV files
├── src/
│   ├── data/
│   │   ├── loader.py              # Data loading utilities
│   │   ├── preprocessing.py       # Data preprocessing
│   │   ├── data_processing.py     # Raw to CIR conversion
│   │   └── feature_engineering.py # Feature creation
│   ├── models/
│   │   ├── linear.py             # Linear models
│   │   ├── svr.py                # Support Vector Regressors
│   │   ├── knn.py                # K-Nearest Neighbors
│   │   ├── mlp.py                # Neural Networks
│   │   ├── ensemble.py           # Ensemble methods
│   │   ├── lstm.py               # LSTM implementation
│   │   └── model_registry.py     # Central model registry
│   ├── training/
│   │   ├── train_sklearn.py      # Basic sklearn training
│   │   ├── train_dl.py           # Deep learning training
│   │   └── train_enhanced.py     # Enhanced training with all models
│   ├── evaluation/
│   │   ├── metrics.py            # Comprehensive metrics
│   │   └── visualization.py      # Plotting utilities
│   └── utils/
├── main.py                # Original main script
├── main_enhanced.py       # Enhanced analysis script
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🔧 Installation

```bash
# Clone the repository
git clone <repository-url>
cd Position_Estimation_Code

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install advanced ML libraries
pip install xgboost lightgbm
```

## 🏃 Quick Start

### 1. Basic Usage
```python
# Run original comparison (Linear, SVR, LSTM)
python main.py

# Run enhanced analysis with all models
python main_enhanced.py
```

### 2. Data Processing
```python
from src.data.data_processing import process_all_pairs

# Convert raw PL/RMS files to CIR format
process_all_pairs("data/raw", "data/processed", decimals=2)
```

### 3. Train Specific Models
```python
from src.models.model_registry import get_model
from src.data.loader import load_cir_data, extract_features_and_target
from sklearn.model_selection import train_test_split

# Load data
df = load_cir_data("data/processed", filter_keyword="FCPR-D1")
X, y = extract_features_and_target(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a model
model = get_model("random_forest", n_estimators=200)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### 4. Feature Engineering
```python
from src.data.feature_engineering import create_engineered_features

# Create enhanced features
df_enhanced = create_engineered_features(df, include_coordinates=True)

# Select best features
selected_features = select_features(df_enhanced, y, method='correlation')
```

## 📊 Model Performance

Typical performance on FCPR-D1 dataset (RMSE in meters):

| Model | RMSE | Training Time |
|-------|------|---------------|
| Random Forest | ~50-60 | Fast |
| XGBoost | ~45-55 | Moderate |
| LSTM | ~60-70 | Slow |
| Ridge Regression | ~70-80 | Very Fast |
| SVR RBF | ~55-65 | Moderate |

## 🔬 Advanced Features

### Hyperparameter Tuning
```python
from src.training.train_enhanced import hyperparameter_search

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

best_params = hyperparameter_search('random_forest', X_train, y_train, param_grid)
```

### Custom Model Pipeline
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.models.model_registry import get_model

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', get_model('mlp', hidden_layers=(100, 50, 25)))
])
```

### Cross-Environment Training
```python
# Train on multiple environments
df_all = pd.concat([
    load_cir_data(processed_dir, filter_keyword="FCPR"),
    load_cir_data(processed_dir, filter_keyword="ICU")
])
```

## 📈 Results Visualization

The enhanced framework provides comprehensive visualization:
- Feature importance plots
- Learning curves
- Prediction vs actual scatter plots
- Residual analysis
- Cross-validation scores
- Model comparison dashboards

## 🔮 Future Enhancements

1. **Deep Learning Models**
   - GRU and Transformer architectures
   - CNN for spatial pattern recognition
   - Attention mechanisms for feature importance

2. **Advanced Features**
   - Frequency domain features
   - Wavelet transforms
   - Auto-encoder features

3. **Online Learning**
   - Incremental model updates
   - Adaptive algorithms

4. **Deployment**
   - Model serialization
   - REST API for predictions
   - Real-time inference

## 🤝 Contributing

Feel free to add more algorithms or enhance existing ones:

1. Add new model implementations in `src/models/`
2. Register them in `model_registry.py`
3. Update training scripts to include new models
4. Add tests and documentation

## 📝 Citation

If you use this code in your research, please cite:
```
@software{position_estimation_cir,
  title = {Position Estimation Using Channel Impulse Response},
  year = {2024},
  url = {https://github.com/your-repo}
}
```

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.