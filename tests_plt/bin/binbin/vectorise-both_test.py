import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

# Parameters
DISTANCE_THRESHOLD = 0.5  # Distance threshold for encounters
MAX_DURATION_THRESHOLD = 1200  # Max encounter duration filter

def compute_pairwise_distances_and_encounters(df):
    """
    Computes pairwise neighbor distances within each ('group_id', 'frame'),
    and tracks encounter events (based on DISTANCE_THRESHOLD).
    """
    # Ensure required index levels exist
    required_index = {'group_id', 'frame', 'individual_id'}
    if not required_index.issubset(df.index.names):
        raise ValueError("DataFrame index must include 'group_id', 'frame', and 'individual_id'.")

    # Reset index for easier calculations
    df = df.reset_index()

    # Group by 'group_id' and 'frame'
    grouped = df.groupby(['group_id', 'frame'], sort=False)
    total_groups = len(grouped)

    # Prepare storage
    mean_distances = np.full(df.shape[0], np.nan)
    encounter_counts = np.full(df.shape[0], 0)  # How often an individual is in an encounter

    # Iterate over groups with tqdm progress bar
    for (group_id, frame), group in tqdm(grouped, total=total_groups, desc="Processing groups"):
        if len(group) < 2:
            continue  # Skip groups with only one individual

        coords = group[['x', 'y']].to_numpy()
        indices = group.index

        # Compute pairwise distances
        dist_matrix = cdist(coords, coords)

        # Compute mean distance per individual
        mean_distances[indices] = dist_matrix.mean(axis=1)

        # Identify encounter events (count of times an individual is within threshold of others)
        encounter_matrix = dist_matrix < DISTANCE_THRESHOLD
        encounter_counts[indices] = encounter_matrix.sum(axis=1) - 1  # Subtract self-comparison

    # Assign results
    df['pairwise_distance'] = mean_distances
    df['encounter_count'] = encounter_counts  # Number of encounters per individual

    # Restore original multi-index
    df = df.set_index(['group_id', 'frame', 'individual_id'])

    return df

if __name__ == '__main__':
    # Load your data
    df_final = pd.read_pickle(
        '/Users/aljoscha/Downloads/2012_nompC_Crimson_WT_4min_50p_TRex_beta_2101/results/data/data_frame_groups.pkl')

    df_final3 = compute_pairwise_distances_and_encounters(df_final)

    print(df_final3)
    print("Index names:", df_final3.index.names)
    print("Columns:", df_final3.columns.tolist())