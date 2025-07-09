"""
Configuration settings for the Position Estimation project
"""

# Model Configuration
MODEL_CONFIG = {
    'hidden_size': 64,
    'num_layers': 2,
    'dropout': 0.2,
    'sequence_length': 10,
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 100,
    'patience': 20
}

# GRU Model Configuration
GRU_CONFIG = {
    'hidden_dim': 64,
    'num_layers': 2,
    'dropout': 0.2
}

# Training Configuration
TRAINING_CONFIG = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'weight_decay': 1e-5,
    'validation_split': 0.2,
    'random_seed': 42
}

# Feature Engineering Configuration
FEATURE_CONFIG = {
    'correlation_threshold': 0.1,
    'simulation_length': 10,
    'base_features': ['PL', 'RMS']
}

# Data Processing Configuration
DATA_CONFIG = {
    'input_file': 'data/processed/FCPR_CIR.csv',
    'target_column': 'r',
    'feature_columns': ['PL_1', 'RMS_1', 'PL_2', 'RMS_2', 'PL_3', 'RMS_3'],
    'processed_dir': 'data/processed',
    'datasets': ['FCPR']
}

# Analysis Configuration
ANALYSIS_CONFIG = {
    'feature_selection': {
        'correlation_threshold': 0.3,
        'excluded_features': ['r', 'X', 'Y']
    },
    'visualization': {
        'figure_sizes': {
            'data_exploration': (12, 5),
            'model_comparison': (17, 6)
        },
        'height_ratios': [1, 1],
        'scatter_alpha': 0.6,
        'scatter_size': 20,
        'grid_alpha': 0.3
    },
    'output': {
        'results_dir': 'results',
        'report_file': 'analysis_report.txt'
    }
}

# Training Options
TRAINING_OPTIONS = {
    'save_predictions': True
}
