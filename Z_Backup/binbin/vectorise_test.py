import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

def compute_pairwise_distances_vectorized(df):
    """
    Computes pairwise neighbor distances within each ('group_id', 'frame')
    using a vectorized approach with tqdm progress tracking.
    """
    # Ensure index contains 'group_id' and 'frame', reset index to work with columns
    if not {'group_id', 'frame'}.issubset(df.index.names):
        raise ValueError("DataFrame index must include 'group_id' and 'frame'.")

    df = df.reset_index()  # Convert index to columns for easier grouping

    # Group by 'group_id' and 'frame'
    grouped = df.groupby(['group_id', 'frame'], sort=False)
    total_groups = len(grouped)

    # Prepare storage
    mean_distances = np.full(df.shape[0], np.nan)

    # Iterate over groups with tqdm progress bar
    for (group_id, frame), group in tqdm(grouped, total=total_groups, desc="Processing groups"):
        if len(group) < 2:
            continue  # Skip groups with only one individual

        coords = group[['x', 'y']].to_numpy()

        # Compute pairwise distances
        dist_matrix = cdist(coords, coords)

        # Mean distance per individual
        mean_distances[group.index] = dist_matrix.mean(axis=1)

    # Assign results and restore index
    df['pairwise_distance'] = mean_distances
    df = df.set_index(['group_id', 'frame'])  # Restore original index
    return df

if __name__ == '__main__':
    # Load your data
    df_final = pd.read_pickle(
        '/Users/aljoscha/Downloads/2012_nompC_Crimson_WT_4min_50p_TRex_beta_2101/results/data/data_frame_groups.pkl')

    df_final3 = compute_pairwise_distances_vectorized(df_final)

    print(df_final3)