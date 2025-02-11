from tqdm import tqdm

def handle_main_dir(main_directory, condition):
    """
    Creates a list of lists containing the directories and the condition for the conditions data,
    as well as a result directory in the main directory.

    Args:
        main_directory (str): The main directory containing the conditions subdirectories.
        condition (list): A list containing the conditions.

    Returns:
        data structure (list of lists): [[condition0_dir, condition0], [condition1_dir, condition1], results_dir]
    """

    import os
    group_dir, single_dir = None, None  # Initialize variables with default values

    results_main_dir = os.path.join(main_directory, 'results')
    os.makedirs(results_main_dir, exist_ok=True)
    results_data_dir = os.path.join(results_main_dir, 'data')
    os.makedirs(results_data_dir, exist_ok=True)
    results_plt_dir = os.path.join(results_main_dir, 'plt')
    os.makedirs(results_plt_dir, exist_ok=True)
    results_dir = [results_data_dir, results_plt_dir]
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


def create_mapping_actual_groups(df_initial, condition):
    """
    Creates actual groups by assigning each 'sub_dir' a unique 'group_id'
    based on its genotype without bootstrapping.

    Args:
        df_initial (pd.DataFrame): Multi-indexed DataFrame with levels ['sub_dir', 'condition', 'genotype', 'frame'].
        condition (str): The condition to filter data for actual groups.

    Returns:
        pd.DataFrame: A mapping DataFrame with ['sub_dir', 'group_id'].
    """
    import pandas as pd
    # Filter for the specified condition
    df_filtered = df_initial[df_initial.index.get_level_values('condition') == condition]

    # Store mappings
    mapping_data = []

    # Iterate over each 'sub_dir' and assign a group_id
    counter = 0
    prev_geno_name = None  # Initialize a variable to track the previous genotype

    for sub_dir, sub_dir_df in df_filtered.groupby(level='sub_dir'):
        geno_name = sub_dir_df.index.get_level_values('genotype').unique()[0]

        # If the genotype has changed, reset the counter
        if geno_name != prev_geno_name:
            counter = 0

        group_id = f"ID_RGN_{geno_name}_{counter}"  # Unique group_id

        # Store the mapping
        mapping_data.append({'sub_dir': sub_dir, 'group_id': group_id})
        counter += 1

        # Update the previous genotype to the current one
        prev_geno_name = geno_name

    # Convert to a DataFrame
    mapping_actual_groups = pd.DataFrame(mapping_data)

    return mapping_actual_groups


def create_mapping_artificial_groups_bootstrapped(df, condition, group_size, bootstrap_reps=2):
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
                group_id = f"ID_AIB_{geno}_{group_id_counter}"

                # Store mapping (each individual -> multiple group IDs)
                for sub_dir in group_sub_dirs:
                    mapping_list.append((sub_dir, group_id))

                group_id_counter += 1

    # Convert mapping list into a DataFrame
    mapping_artificial_groups_bootstapped = pd.DataFrame(mapping_list, columns=['sub_dir', 'group_id'])

    return mapping_artificial_groups_bootstapped

def create_mapping_semi_artificial_groups_bootstrapped(df, condition, group_size, bootstrap_reps=2):
    """
    Creates artificial groups by bootstrapping within genotypes but ensures that no group
    contains data from the same sub_dir more than once.

    Instead of duplicating data, this function creates a separate mapping table linking individuals
    (sub_dir) to artificial group IDs.

    Args:
        df (pd.DataFrame): Multi-indexed DataFrame with levels ['sub_dir', 'condition', 'genotype', 'frame'].
        condition (str): The condition used to filter data.
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

    # Filter data for the given condition
    df_filtered = df[df.index.get_level_values('condition') == condition]

    mapping_list = []  # Stores (sub_dir, group_id) pairs

    # Iterate over each genotype in the filtered dataframe
    for geno, geno_df in df_filtered.groupby(level='genotype'):
        sub_dirs = list(geno_df.index.get_level_values('sub_dir').unique())
        group_id_counter = 0

        if len(sub_dirs) < group_size:
            raise ValueError(f"Not enough trials for genotype {geno} to form groups of size {group_size}.")

        for _ in range(bootstrap_reps):
            # Shuffle to ensure randomness
            random.shuffle(sub_dirs)
            used_sub_dirs = set()

            while len(used_sub_dirs) + group_size <= len(sub_dirs):
                # Select `group_size` sub_dirs that haven't been used together
                group_sub_dirs = random.sample([sub for sub in sub_dirs if sub not in used_sub_dirs], group_size)

                # Assign a unique group_id
                group_id = f"ID_AGB_{geno}_{group_id_counter}"

                # Store mapping (each individual -> multiple group IDs)
                for sub_dir in group_sub_dirs:
                    mapping_list.append((sub_dir, group_id))

                # Mark these sub_dirs as used in this round
                used_sub_dirs.update(group_sub_dirs)
                group_id_counter += 1

    # Convert mapping list into a DataFrame
    mapping_artificial_groups_exclusive = pd.DataFrame(mapping_list, columns=['sub_dir', 'group_id'])

    return mapping_artificial_groups_exclusive