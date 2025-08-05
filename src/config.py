"""Configuration settings for the Position Estimation project"""

MODEL_CONFIG = {
    'hidden_size': 100,
    'num_layers': 2,
    'dropout': 0.3,
    'sequence_length': 10,
    'batch_size': 32,
    'learning_rate': 0.001,
<<<<<<< HEAD
    'epochs': 300
=======
    'epochs': 300,
    'patience': 30
>>>>>>> 4a5587a19d7e800ecc74d1435a3f8cc8c0d24b85
}

GRU_CONFIG = {
    'hidden_dim': 64,
    'num_layers': 2,
    'dropout': 0.3
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
    'target_column': 'X',  # Changed from 'r' to 'X'
    'feature_columns': ['PL_1', 'RMS_1', 'PL_2', 'RMS_2', 'PL_3', 'RMS_3'],
    'processed_dir': 'data/processed',
    'datasets': ['FCPR']
}