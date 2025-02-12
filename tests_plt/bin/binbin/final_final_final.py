import pandas as pd
import numpy as np
import os
from scipy.spatial.distance import cdist
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from package.util_df_prep import handle_main_dir
from package import config_settings

# Parameters
DISTANCE_THRESHOLD = 0.5  # Distance threshold for encounters
MAX_DURATION_THRESHOLD = 1200  # Max encounter duration filter


def process_group(group):
    """
    Computes:
    - Mean pairwise distances
    - Encounter counts (number of frames within DISTANCE_THRESHOLD)
    - Nearest Neighbor Distance (NND)
    for a single (group_id, frame) group.
    """
    if len(group) < 2:
        group['pairwise_distance'] = np.nan
        group['nearest_neighbor_distance'] = np.nan
        group['encounter_count'] = 0
        return group

    coords = group[['x', 'y']].to_numpy()

    # Compute pairwise distances
    dist_matrix = cdist(coords, coords)

    # Compute mean distance per individual
    group['pairwise_distance'] = dist_matrix.mean(axis=1)

    # Compute Nearest Neighbor Distance (NND) - minimum nonzero distance
    np.fill_diagonal(dist_matrix, np.inf)  # Ignore self-distance by setting diagonal to infinity
    group['nearest_neighbor_distance'] = np.min(dist_matrix, axis=1)

    # Identify encounter events (count of times an individual is within threshold of others)
    encounter_matrix = dist_matrix < DISTANCE_THRESHOLD
    group['encounter_count'] = encounter_matrix.sum(axis=1) - 1  # Subtract self-comparison

    return group


def compute_pairwise_distances_and_encounters(df):
    """
    Parallelized computation of pairwise distances and encounters.
    """
    # Ensure required index levels exist
    required_index = {'group_id', 'frame', 'individual_id'}
    if not required_index.issubset(df.index.names):
        raise ValueError("DataFrame index must include 'group_id', 'frame', and 'individual_id'.")

    # Reset index for easier calculations
    df = df.reset_index()

    # Group by 'group_id' and 'frame'
    grouped = list(df.groupby(['group_id', 'frame'], sort=False))  # Convert to list for tqdm
    total_groups = len(grouped)

    results = []
    with ProcessPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
        for result in tqdm(executor.map(lambda g: process_group(g[1]), grouped),
                           total=total_groups, desc="Processing groups"):
            results.append(result)

    # Concatenate results
    df = pd.concat(results)

    # Restore original multi-index
    df = df.set_index(['group_type', 'genotype', 'group_id', 'individual_id', 'frame'])

    return df


if __name__ == '__main__':
    condition_dir = handle_main_dir('/Users/aljoscha/Downloads/2012_nompC_Crimson_WT_4min_50p_TRex_beta_2101',
                                    config_settings.condition)
    df_final = pd.read_pickle(os.path.join(condition_dir[2][0], 'data_frame_groups.pkl'))

    df_final3 = compute_pairwise_distances_and_encounters(df_final)
    df_final3.to_pickle(os.path.join(condition_dir[2][0], "data_frame_group_parameters.pkl"))

    print(df_final3)
    print("Index names:", df_final3.index.names)
    print("Columns:", df_final3.columns.tolist())