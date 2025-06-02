import os
import pandas as pd
import numpy as np

def compute_radius(x: float, y: float, decimals: int) -> float:
    """Compute polar radius with configurable decimal precision."""
    return round(np.sqrt(x**2 + y**2), decimals)

def process_pair(pl_path: str, rms_path: str, output_path: str, decimals: int) -> None:
    """
    Merge a PL and RMS CSV file (with matching X, Y coordinates) into a CIR file.
    The output format is: X, Y, r, PL, RMS
    """
    df_pl = pd.read_csv(pl_path, header=None)
    df_rms = pd.read_csv(rms_path, header=None)

    # Assign column names
    df_pl.columns = ['X', 'Y', 'PL']
    df_rms.columns = ['X', 'Y', 'RMS']

    # Ensure coordinates match
    if not df_pl[['X', 'Y']].equals(df_rms[['X', 'Y']]):
        raise ValueError(f"Coordinate mismatch between:\n{pl_path}\nand\n{rms_path}")

    # Create final DataFrame
    df = df_pl[['X', 'Y']].copy()
    df['r'] = df.apply(lambda row: compute_radius(row['X'], row['Y'], decimals), axis=1)
    df['PL'] = df_pl['PL']
    df['RMS'] = df_rms['RMS']

    df.to_csv(output_path, index=False)

def process_all_pairs(raw_dir: str, processed_dir: str, decimals: int) -> None:
    """
    Process all _PL.csv and _RMS.csv file pairs in the raw directory.
    Output is one _CIR.csv per pair in the processed directory.
    """
    # Clean processed directory
    if os.path.exists(processed_dir):
        for file in os.listdir(processed_dir):
            file_path = os.path.join(processed_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(processed_dir)

    files = os.listdir(raw_dir)
    pl_files = [f for f in files if f.endswith('_PL.csv')]

    for pl_file in pl_files:
        base_name = pl_file.replace('_PL.csv', '')
        rms_file = base_name + '_RMS.csv'
        output_file = base_name + '_CIR.csv'

        pl_path = os.path.join(raw_dir, pl_file)
        rms_path = os.path.join(raw_dir, rms_file)
        output_path = os.path.join(processed_dir, output_file)

        try:
            if not os.path.exists(rms_path):
                print(f"Skipping {pl_file} — RMS file not found.")
                continue

            process_pair(pl_path, rms_path, output_path, decimals)
            print(f"Processed {output_file}")
        except Exception as e:
            print(f"Error processing {pl_file} and {rms_file}: {e}")
