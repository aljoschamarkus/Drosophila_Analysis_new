import pandas as pd
import os
import numpy as np
from scipy.spatial.distance import pdist, squareform
from concurrent.futures import ProcessPoolExecutor
from package.util_data_preperation import handle_main_dir
from package import config_settings
from tqdm import tqdm

def process_group(group):
    """
    Computes pairwise neighbor distances for a single group.
    """
    coords = group[['x', 'y']].values
    if len(coords) > 1:
        dist_matrix = squareform(pdist(coords))  # Compute pairwise distances
        mean_distances = dist_matrix.mean(axis=1)  # Mean distance per individual
    else:
        mean_distances = np.array([0])  # No neighbors if only one individual

    group['pairwise_distance'] = mean_distances
    return group

def compute_pairwise_distances(df):
    """
    Computes pairwise neighbor distances within each 'group_id' and 'frame'.
    Stores the mean pairwise distance for each individual in the group.
    """
    grouped = list(df.groupby(['group_id', 'frame']))  # Convert generator to list for tqdm
    # grouped = list(df.groupby(['frame']))  # Convert generator to list for tqdm
    total_groups = len(grouped)

    results = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for result in tqdm(executor.map(process_group, (group for _, group in grouped)),
                           total=total_groups, desc="Processing groups"):
            results.append(result)

    return pd.concat(results)

if __name__ == '__main__':
    # Load your data here
    df_final = pd.read_pickle(
        '/Users/aljoscha/Downloads/2012_nompC_Crimson_WT_4min_50p_TRex_beta_2101/results/data/data_frame_groups.pkl')

    # df_final2 = df_final.xs('G_ID_RGN_WTxCrimson_0', level='group_id')
    # df_final3 = compute_pairwise_distances(df_final2)
    df_final3 = compute_pairwise_distances(df_final)

    # Save results
    # df_final.to_pickle(os.path.join(condition_dir[2][0], "data_frame_with_distances.pkl"))

    print(df_final3)
