def handle_main_dir(main_directory, condition):
    """
    Creates a list containing the directories for the data of the conditions,
    as well as result directories.

    Args:
        main_directory (str): The main directory containing the conditions subdirectories.
        condition (list): A list containing the conditions.

    Returns:
        data structure (list): [condition0_dir, condition1_dir, results_data_dir, results_plt_dir]
    """

    import os
    group_dir, single_dir = None, None

    # Ensure condition has at least two elements
    if not isinstance(condition, (list, tuple)) or len(condition) < 2:
        raise ValueError("condition must be a list or tuple with at least two elements.")

    for folder in os.listdir(main_directory):
        folder_lower = folder.lower()  # Avoid recalculating `folder.lower()`
        if any(part in folder_lower for part in condition):
            if condition[0] in folder_lower:
                group_dir = os.path.join(main_directory, folder)
            elif condition[1] in folder_lower:
                single_dir = os.path.join(main_directory, folder)

    results_main_dir = '/Users/aljoscha/Downloads/results'
    os.makedirs(results_main_dir, exist_ok=True)
    results_data_dir = os.path.join(results_main_dir, 'data')
    os.makedirs(results_data_dir, exist_ok=True)
    results_plt_dir = os.path.join(results_main_dir, 'plt')
    os.makedirs(results_plt_dir, exist_ok=True)

    return group_dir, single_dir, results_data_dir, results_plt_dir


import pandas as pd
import random


def create_mapping_actual_groups(df, condition):
    """
    Creates actual groups by assigning each 'individual_id' a unique 'group_id'
    while still grouping within 'sub_dir'.

    Args:
        df (pd.DataFrame): Multi-indexed DataFrame with levels ['sub_dir', 'individual_id', 'condition', 'genotype', 'frame'].
        condition (str): The condition to filter data for actual groups.

    Returns:
        pd.DataFrame: A mapping DataFrame with ['individual_id', 'group_id'].
    """
    # Filter for the specified condition
    df_filtered = df[df.index.get_level_values('condition') == condition]

    # Store mappings
    mapping_data = []
    counter = 0
    prev_geno_name = None

    # Iterate over each 'sub_dir'
    for sub_dir, sub_dir_df in df_filtered.groupby(level='sub_dir'):
        geno_name = sub_dir_df.index.get_level_values('genotype').unique()[0]

        if geno_name != prev_geno_name:
            counter = 0

        group_id = f"G_ID_RGN_{geno_name}_{counter}"

        # Assign the same group_id to all 'individual_id' within this 'sub_dir'
        for individual_id in sub_dir_df.index.get_level_values('individual_id').unique():
            mapping_data.append({'individual_id': individual_id, 'group_id': group_id})

        counter += 1
        prev_geno_name = geno_name

    return pd.DataFrame(mapping_data)


def create_mapping_artificial_groups_bootstrapped(df, condition, group_size, bootstrap_reps=2):
    """
    Creates artificial groups by bootstrapping within genotypes, ensuring each 'individual_id'
    is used instead of 'sub_dir', while maintaining grouping logic.

    Args:
        df (pd.DataFrame): Multi-indexed DataFrame with levels ['sub_dir', 'individual_id', 'condition', 'genotype', 'frame'].
        condition (str): A string containing the condition, used to filter data.
        group_size (int): Size of each group.
        bootstrap_reps (int): Number of times to apply bootstrapping.

    Returns:
        pd.DataFrame: A mapping table linking individuals ('individual_id') to artificial groups.
    """
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError(
            "The input DataFrame must have a MultiIndex with levels ['sub_dir', 'individual_id', 'condition', 'genotype', 'frame'].")

    # Filter data for the given condition
    df_filtered = df[df.index.get_level_values('condition') == condition]
    mapping_list = []

    # Iterate over each genotype
    for geno, geno_df in df_filtered.groupby(level='genotype'):
        individual_ids = geno_df.index.get_level_values('individual_id').unique()
        group_id_counter = 0

        if len(individual_ids) < group_size:
            raise ValueError(f"Not enough individuals for genotype {geno} to form groups of size {group_size}.")

        for _ in range(bootstrap_reps):
            for base_id in individual_ids:
                # Randomly select group_size - 1 other individuals
                other_ids = random.sample(
                    [ind for ind in individual_ids if ind != base_id], group_size - 1
                )

                group_members = [base_id] + other_ids
                group_id = f"G_ID_AIB_{geno}_{group_id_counter}"

                # Store mapping (each individual -> multiple group IDs)
                for individual_id in group_members:
                    mapping_list.append({'individual_id': individual_id, 'group_id': group_id})

                group_id_counter += 1

    return pd.DataFrame(mapping_list)


def create_mapping_semi_artificial_groups_bootstrapped(df, condition, group_size, bootstrap_reps=2):
    """
    Creates artificial groups by bootstrapping within genotypes, ensuring that no group
    contains data from the same sub_dir more than once, while using 'individual_id' as the key.

    Args:
        df (pd.DataFrame): Multi-indexed DataFrame with levels ['sub_dir', 'individual_id', 'condition', 'genotype', 'frame'].
        condition (str): The condition used to filter data.
        group_size (int): Size of each group.
        bootstrap_reps (int): Number of times to apply bootstrapping.

    Returns:
        pd.DataFrame: A mapping table linking individuals ('individual_id') to artificial groups.
    """
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError(
            "The input DataFrame must have a MultiIndex with levels ['sub_dir', 'individual_id', 'condition', 'genotype', 'frame'].")

    df_filtered = df[df.index.get_level_values('condition') == condition]
    mapping_list = []

    for geno, geno_df in df_filtered.groupby(level='genotype'):
        sub_dir_groups = geno_df.groupby(level='sub_dir')
        sub_dir_to_individuals = {sub_dir: list(sub_dir_df.index.get_level_values('individual_id').unique()) for
                                  sub_dir, sub_dir_df in sub_dir_groups}
        sub_dirs = list(sub_dir_to_individuals.keys())
        group_id_counter = 0

        if len(sub_dirs) < group_size:
            raise ValueError(f"Not enough trials for genotype {geno} to form groups of size {group_size}.")

        for _ in range(bootstrap_reps):
            random.shuffle(sub_dirs)
            used_sub_dirs = set()

            while len(used_sub_dirs) + group_size <= len(sub_dirs):
                group_sub_dirs = random.sample([sub for sub in sub_dirs if sub not in used_sub_dirs], group_size)
                group_id = f"G_ID_AGB_{geno}_{group_id_counter}"

                for sub_dir in group_sub_dirs:
                    individual_id = random.choice(sub_dir_to_individuals[sub_dir])
                    mapping_list.append({'individual_id': individual_id, 'group_id': group_id})

                used_sub_dirs.update(group_sub_dirs)
                group_id_counter += 1

    return pd.DataFrame(mapping_list)

# def create_mapping_semi_artificial_groups_bootstrapped(df, condition, group_size, bootstrap_reps=2):
#     """
#     Creates artificial groups by bootstrapping within genotypes but ensures that no group
#     contains data from the same sub_dir more than once.
#
#     Instead of duplicating data, this function creates a separate mapping table linking individuals
#     (sub_dir) to artificial group IDs.
#
#     Args:
#         df (pd.DataFrame): Multi-indexed DataFrame with levels ['sub_dir', 'condition', 'genotype', 'frame'].
#         condition (str): The condition used to filter data.
#         group_size (int): Size of each group.
#         bootstrap_reps (int): Number of times to apply bootstrapping.
#
#     Returns:
#         pd.DataFrame: A mapping table linking individuals (sub_dir) to artificial groups.
#     """
#     import random
#     import pandas as pd
#
#     if not isinstance(df.index, pd.MultiIndex):
#         raise ValueError(
#             "The input DataFrame must have a MultiIndex with levels ['sub_dir', 'condition', 'genotype', 'frame'].")
#
#     if 'genotype' not in df.index.names:
#         raise ValueError("The input DataFrame must have a 'genotype' level in its MultiIndex.")
#
#     # Filter data for the given condition
#     df_filtered = df[df.index.get_level_values('condition') == condition]
#
#     mapping_list = []  # Stores (sub_dir, group_id) pairs
#
#     # Iterate over each genotype in the filtered dataframe
#     for geno, geno_df in df_filtered.groupby(level='genotype'):
#         sub_dirs = list(geno_df.index.get_level_values('sub_dir').unique())
#         group_id_counter = 0
#
#         if len(sub_dirs) < group_size:
#             raise ValueError(f"Not enough trials for genotype {geno} to form groups of size {group_size}.")
#
#         for _ in range(bootstrap_reps):
#             # Shuffle to ensure randomness
#             random.shuffle(sub_dirs)
#             used_sub_dirs = set()
#
#             while len(used_sub_dirs) + group_size <= len(sub_dirs):
#                 # Select `group_size` sub_dirs that haven't been used together
#                 group_sub_dirs = random.sample([sub for sub in sub_dirs if sub not in used_sub_dirs], group_size)
#
#                 # Assign a unique group_id
#                 group_id = f"ID_AGB_{geno}_{group_id_counter}"
#
#                 # Store mapping (each individual -> multiple group IDs)
#                 for sub_dir in group_sub_dirs:
#                     mapping_list.append((sub_dir, group_id))
#
#                 # Mark these sub_dirs as used in this round
#                 used_sub_dirs.update(group_sub_dirs)
#                 group_id_counter += 1
#
#     # Convert mapping list into a DataFrame
#     mapping_artificial_groups_exclusive = pd.DataFrame(mapping_list, columns=['sub_dir', 'group_id'])
#
#     return mapping_artificial_groups_exclusive

def compute_pairwise_distances_and_encounters(df, distance_threshold_encounter):
    """
    Computes:
    - Mean pairwise distances
    - Encounter counts (number of frames within DISTANCE_THRESHOLD)
    - Nearest Neighbor Distance (NND)
    """
    import numpy as np
    from scipy.spatial.distance import cdist
    from tqdm import tqdm

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
        encounter_matrix = dist_matrix < distance_threshold_encounter
        encounter_counts[indices] = (encounter_matrix.sum(axis=1) > 1).astype(int)

    # Assign results
    df['PND'] = mean_distances
    df['NND'] = nearest_neighbor_distances
    df['encounter_count'] = encounter_counts  # Number of encounters per individual

    # Restore original multi-index
    df = df.set_index(['sub_dir', 'condition', 'genotype', 'group_type', 'group_id', 'individual_id', 'frame'])

    return df