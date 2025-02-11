# data_processing_utils.py

import pandas as pd
import numpy as np

def process_csv(file_path):
    """Load and clean CSV for X, Y centroids, and SPEED."""
    data = pd.read_csv(file_path)

    # Limit to the first 5400 rows
    data = data.iloc[:5400]

    # Convert columns to numeric, dropping invalid entries
    X_centroid = pd.to_numeric(data['X#wcentroid (cm)'], errors='coerce')
    Y_centroid = pd.to_numeric(data['Y#wcentroid (cm)'], errors='coerce')

    # Drop NaN and Inf values explicitly
    X_centroid = X_centroid[~X_centroid.isin([np.inf, -np.inf])]
    Y_centroid = Y_centroid[~Y_centroid.isin([np.inf, -np.inf])]

    # Return cleaned centroids
    return (
        X_centroid.dropna().values,
        Y_centroid.dropna().values
    )