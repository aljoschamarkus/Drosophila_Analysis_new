import pandas as pd
import numpy as np
import os

from tqdm import tqdm
from package.util_data_preperation import handle_main_dir
from package import config_settings

# Parameters
DISTANCE_THRESHOLD = 0.5  # Distance threshold for encounters
MAX_DURATION_THRESHOLD = 1200  # Max encounter duration filter

def compute_pairwise_distances_and_encounters(df):
    """
    Computes:
    - Mean pairwise distances
    - Encounter counts (number of frames within DISTANCE_THRESHOLD)
    - Nearest Neighbor Distance (NND)
    """
    from scipy.spatial.distance import cdist

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
    nearest_neighbor_distances = np.full(df.shape[0], np.nan)
    encounter_counts = np.full(df.shape[0], 0)  # Number of encounters per individual

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

        # Compute Nearest Neighbor Distance (NND) - minimum nonzero distance
        np.fill_diagonal(dist_matrix, np.inf)  # Ignore self-distance by setting diagonal to infinity
        nearest_neighbor_distances[indices] = np.min(dist_matrix, axis=1)

        # Identify encounter events (count of times an individual is within threshold of others)
        encounter_matrix = dist_matrix < DISTANCE_THRESHOLD
        encounter_counts[indices] = encounter_matrix.sum(axis=1) - 1  # Subtract self-comparison

    # Assign results
    df['pairwise_distance'] = mean_distances
    df['nearest_neighbor_distance'] = nearest_neighbor_distances
    df['encounter_count'] = encounter_counts  # Number of encounters per individual

    # Restore original multi-index
    df = df.set_index(['group_type', 'genotype', 'group_id', 'individual_id', 'frame'])

    return df

if __name__ == '__main__':
    condition_dir = handle_main_dir('/Users/aljoscha/Downloads/2012_nompC_Crimson_WT_4min_50p_TRex_beta_2101', config_settings.condition)
    df_final = pd.read_pickle(os.path.join(condition_dir[2][0], 'data_frame_groups.pkl'))

    df_final3 = compute_pairwise_distances_and_encounters(df_final)
    df_final3.to_pickle(os.path.join(condition_dir[2][0], "data_frame_group_parameters.pkl"))

    print(df_final3)
    print("Index names:", df_final3.index.names)
    print("Columns:", df_final3.columns.tolist())