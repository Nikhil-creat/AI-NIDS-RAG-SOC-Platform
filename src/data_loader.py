"""
data_loader.py
Loads and cleans the CICIDS2017 network traffic dataset.
Place the downloaded CSV file(s) inside the /data folder before running.
"""

import pandas as pd
import numpy as np
import glob
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_data(pattern="*.csv"):
    """Load and concatenate all CSV files found in the data directory."""
    files = glob.glob(os.path.join(DATA_DIR, pattern))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {DATA_DIR}. "
            "Download CICIDS2017 from Kaggle and place the CSV(s) here."
        )
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df.columns = df.columns.str.strip()
    return df


def clean_data(df):
    """Handle infinities, NaNs, duplicates."""
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    df = df.drop_duplicates()
    return df


def add_binary_label(df, label_col="Label"):
    """Add a binary column: 0 = benign, 1 = attack."""
    df["Label_binary"] = df[label_col].apply(lambda x: 0 if x == "BENIGN" else 1)
    return df


if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    df = add_binary_label(df)
    print("Final shape:", df.shape)
    print(df["Label"].value_counts())
