import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from src.data.loader import load_cir_data
from src.config import DATA_CONFIG, GRU_CONFIG, TRAINING_CONFIG, ANALYSIS_CONFIG
import warnings
import random
from src.data.feature_engineering import create_engineered_features, select_features

random.seed(42)
warnings.filterwarnings('ignore')




df_list = []
    
# Load all available datasets
for keyword in DATA_CONFIG['datasets']:
    try:
        df_temp = load_cir_data(DATA_CONFIG['processed_dir'], filter_keyword=keyword)
        print(f"  Loaded {keyword}: {len(df_temp)} samples")
        df_list.append(df_temp)
    except:
        print(f"  {keyword} not found")

# Combine all data
df_all = pd.concat(df_list, ignore_index=True) if df_list else None

if df_all is None:
    print("No data found!")
    exit()

# 2. Feature Engineering
df_engineered = create_engineered_features(df_all, include_categorical=True)

# Select features - exclude any coordinate-based features
feature_cols = [col for col in df_engineered.columns 
                if col not in ANALYSIS_CONFIG['feature_selection']['excluded_features']]

X = df_engineered[feature_cols]
y = df_engineered[DATA_CONFIG['target_column']]

# Select best features
selected_features = select_features(X, y)
print(f"  Selected {len(selected_features)} features from {len(feature_cols)} total")
print(selected_features)

for feat in selected_features:
    if len(selected_features) >= 5:
        break
    if selected_features:
        corrs = X[selected_features].corrwith(X[feat]).abs()
        print(f"Trying {feat}, correlations with selected: {corrs.values}")
        if (corrs > 0.9).any():
            print(f"Skipping {feat} due to high correlation.")
            continue
    selected_features.append(feat)
    print(f"Selected: {feat}")


