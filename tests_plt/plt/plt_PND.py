import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import pdist, squareform
from concurrent.futures import ProcessPoolExecutor
import numpy as np

colors = [['#e41a1c', '#377eb8', '#4daf4a'], ['#fbb4ae', '#a6cee3', '#b2df8a']]

def calculate_pairwise_distances(df, group_id):
    """Calculate pairwise distances for a specific group_id using vectorized operations."""
    distances = []

    # Filter the DataFrame for the specific group_id
    group_df = df.xs(group_id, level='group_id')

    # Loop through unique frames
    for frame in group_df.index.get_level_values('frame').unique():
        # Extract positions (x, y) for this frame
        frame_df = group_df.xs(frame, level='frame')[['x', 'y']].dropna()

        if len(frame_df) > 1:  # Skip frames with fewer than 2 positions
            # Use pdist for efficient pairwise distance calculation
            distances.extend(pdist(frame_df.values))

    return distances


def process_group(group_id, df):
    """
    Process a single group to calculate distances.
    """
    group_df = df.xs(group_id, level='group_id')
    sub_dir = group_df.index.get_level_values('sub_dir')[0]
    condition = group_df.index.get_level_values('condition')[0]
    genotype = group_df.index.get_level_values('genotype')[0]
    group_type = group_df.index.get_level_values('group_type')[0]

    # Determine if it's an actual or artificial group
    distances = calculate_pairwise_distances(df, group_id)
    return genotype, group_type, distances


def process_groups_parallel(df, selected_lines=None):
    """
    Process groups in parallel and plot KDE for specified lines.
    """
    from concurrent.futures import ProcessPoolExecutor

    # Extract unique genotypes (from the 'genotype' index level)
    genotypes = df.index.get_level_values('genotype').unique()

    # Initialize dictionaries to store distances
    all_distances_actual = {genotype: [] for genotype in genotypes}
    all_distances_artificial = {genotype: [] for genotype in genotypes}

    # Use ProcessPoolExecutor to process groups in parallel
    with ProcessPoolExecutor() as executor:
        # Pass the DataFrame as an additional argument to process_group
        results = list(executor.map(process_group, df.index.get_level_values('group_id').unique(), [df] * len(df.index.get_level_values('group_id').unique())))

    # Collect results
    for genotype, group_type, distances in results:
        if group_type == "actual":
            all_distances_actual[genotype].extend(distances)
        elif group_type == "artificial":
            all_distances_artificial[genotype].extend(distances)

    # Plot KDE for selected lines
    plt.figure(figsize=(10, 6))

    if selected_lines is None:
        # Default: Plot all genotypes for both actual and artificial groups
        selected_lines = [(genotype, "actual") for genotype in genotypes] + [(genotype, "artificial") for genotype in genotypes]

    for idx, (genotype, group_type) in enumerate(selected_lines):
        if group_type == "actual":
            sns.kdeplot(
                all_distances_actual[genotype],
                label=f"Group {genotype}",
                color=colors[0][idx % len(colors[0])],  # Cycle through colors for "actual"
                linestyle='-'
            )
        elif group_type == "artificial":
            sns.kdeplot(
                all_distances_artificial[genotype],
                label=f"Singles {genotype}",
                color=colors[1][idx % len(colors[1])],  # Cycle through colors for "artificial"
                linestyle='--'
            )

    plt.title('Distribution of Pairwise Distances')
    plt.xlabel('Pairwise Distance')
    plt.ylabel('Density')
    plt.legend(title='Genotype and Group Type', loc='upper right')
    plt.tight_layout()
    plt.show()

# Example Usage
# Assuming df_final is already defined
if __name__ == "__main__":
    # Load or define your df_final
    # Example:
    # df_final = pd.DataFrame(...)
    df_final = pd.read_pickle(
        '/Users/aljoscha/Downloads/2012_nompC_Crimson_WT_4min_50p_TRex_beta_2101/results/data_frame_final.pkl')

    # Call the function with the appropriate arguments
    process_groups_parallel(
        df_final,
        selected_lines=[("WTxCrimson", "actual"), ("WTxCrimson", "artificial"), ("nompCxCrimson", "actual"), ("nompCxCrimson", "artificial"), ("nompCxWT", "actual"), ("nompCxWT", "artificial")]
    )