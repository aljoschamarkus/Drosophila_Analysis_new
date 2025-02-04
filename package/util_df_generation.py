def handle_main_dir(main_directory, condition):
    import os
    group_dir, single_dir = None, None  # Initialize variables with default values

    results_dir = os.path.join(main_directory, 'results')
    os.makedirs(results_dir, exist_ok=True)
    # Ensure condition2 has at least two elements
    if not isinstance(condition, (list, tuple)) or len(condition) < 2:
        raise ValueError("condition2 must be a list or tuple with at least two elements.")

    for folder in os.listdir(main_directory):
        folder_lower = folder.lower()  # Avoid recalculating `folder.lower()`
        if any(part in folder_lower for part in condition):
            if condition[0] in folder_lower:
                group_dir = [os.path.join(main_directory, folder), condition[0]]
            elif condition[1] in folder_lower:
                single_dir = [os.path.join(main_directory, folder), condition[1]]

    # Return the directories (or None if not found)
    return group_dir, single_dir, results_dir

def create_actual_groups(df_initial, condition):
    import pandas as pd
    # actual_condition = condition_dir[0][1]  # Get the condition name for actual groups
    df_filtered = df_initial[df_initial.index.get_level_values('condition') == condition]  # Filter data for actual groups

    grouped_data_actual = []

    # Iterate over each sub_dir in the filtered DataFrame
    counter = 0
    for sub_dir, sub_dir_df in df_filtered.groupby(level='sub_dir'):
        sub_dir_df = sub_dir_df.copy()
        geno_name = sub_dir_df.index.get_level_values('genotype').unique()[0]
        sub_dir_df['group_id'] = f"5_{geno_name}_{counter}"  # Assign unique group_id based on sub_dir
        grouped_data_actual.append(sub_dir_df)
        counter += 1

    # Concatenate all data for actual groups
    df_actual_groups = pd.concat(grouped_data_actual, axis=0)

    return df_actual_groups


def create_artificial_groups_bootstrapped(df, condition, group_size, bootstrap_reps=2):
    """
    Creates artificial groups by bootstrapping within genotypes, but only for data where
    condition == condition_dir[1][0]. Each trial is grouped with group_size - 1
    additional trials, ensuring no duplicate trials within a group. The bootstrapping is applied bootstrap_reps times.

    Instead of duplicating data, this function creates a separate mapping table linking individuals
    to multiple artificial group IDs.

    Args:
        df (pd.DataFrame): Multi-indexed DataFrame with levels ['sub_dir', 'condition', 'genotype', 'frame'].
        condition (string): A string containing the condition, used to filter data.
        group_size (int): Size of each group.
        bootstrap_reps (int): Number of times to apply bootstrapping.

    Returns:
        pd.DataFrame: A mapping table linking individuals (sub_dir) to artificial groups.
    """
    import random
    import pandas as pd
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError(
            "The input DataFrame must have a MultiIndex with levels ['sub_dir', 'condition', 'genotype', 'frame'].")

    if 'genotype' not in df.index.names:
        raise ValueError("The input DataFrame must have a 'genotype' level in its MultiIndex.")

    # Filter data for condition == condition_dir[1][1]
    df_filtered = df[df.index.get_level_values('condition') == condition]

    mapping_list = []  # Stores (sub_dir, group_id) pairs

    # Iterate over each genotype in the filtered dataframe
    for geno, geno_df in df_filtered.groupby(level='genotype'):
        sub_dirs = geno_df.index.get_level_values('sub_dir').unique()
        group_id_counter = 0

        if len(sub_dirs) < group_size:
            raise ValueError(f"Not enough trials for genotype {geno} to form groups of size {group_size}.")

        for _ in range(bootstrap_reps):
            for base_sub_dir in sub_dirs:
                # Randomly select group_size - 1 other trials
                other_sub_dirs = random.sample(
                    [sub for sub in sub_dirs if sub != base_sub_dir], group_size - 1
                )

                group_sub_dirs = [base_sub_dir] + other_sub_dirs
                group_id = f"1_ID_{geno}_{group_id_counter}"

                # Store mapping (each individual -> multiple group IDs)
                for sub_dir in group_sub_dirs:
                    mapping_list.append((sub_dir, group_id))

                group_id_counter += 1

    # Convert mapping list into a DataFrame
    group_mapping = pd.DataFrame(mapping_list, columns=['sub_dir', 'group_id'])

    return group_mapping