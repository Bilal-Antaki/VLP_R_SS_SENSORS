"""Configuration settings for the Position Estimation project"""

MODEL_CONFIG = {
    'hidden_size': 64,
    'num_layers': 2,
    'dropout': 0.4,
    'sequence_length': 10,
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 300
}

GRU_CONFIG = {
    'hidden_dim': 64,
    'num_layers': 2,
    'dropout': 0.4
}

LSTM_CONFIG = {
    'hidden_dim': 100,
    'num_layers': 2,
    'dropout': 0.3
}

RNN_CONFIG = {
    'hidden_dim': 64,
    'num_layers': 2,
    'dropout': 0.2
}

SVR_CONFIG = {
    'kernel': 'rbf',
    'C': 100.0,
    'epsilon': 0.01,
    'gamma': 'auto'
}

TRAINING_CONFIG = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 300,
    'weight_decay': 1e-5,
    'validation_split': 0.2,
    'random_seed': 42
}

DATA_CONFIG = {
    'input_file': 'data/processed/FCPR_CIR.csv',
    'target_column': 'Y',
    'feature_columns': ['PL_1', 'RMS_1', 'PL_2', 'RMS_2', 'PL_3', 'RMS_3'],
    'processed_dir': 'data/processed',
    'datasets': ['FCPR']
}