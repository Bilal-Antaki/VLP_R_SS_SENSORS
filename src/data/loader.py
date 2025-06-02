import pandas as pd
import os

def load_cir_data(processed_dir: str, filter_keyword: str = None) -> pd.DataFrame:
    all_data = []
    for file in os.listdir(processed_dir):
        if file.endswith('_CIR.csv') and (filter_keyword is None or filter_keyword in file):
            filepath = os.path.join(processed_dir, file)
            df = pd.read_csv(filepath)
            df['source_file'] = file
            all_data.append(df)
    if not all_data:
        raise FileNotFoundError(f"No matching CIR files in {processed_dir}")
    return pd.concat(all_data, ignore_index=True)

def extract_features_and_target(df: pd.DataFrame, features=['PL', 'RMS'], target='r'):
    X = df[features]
    y = df[target]
    return X, y
