import pandas as pd
import os
import random
import numpy as np
from scipy.spatial.distance import euclidean
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Directories and parameters
main_directory_group = '/Users/aljoscha/Downloads/2012_nompC_Crimson_WT_4min_50p_TRex_beta_2101/group'
main_directory_single = '/Users/aljoscha/Downloads/2012_nompC_Crimson_WT_4min_50p_TRex_beta_2101/single'
results_singles_dir = '/Users/aljoscha/Downloads/Data_sorted/All_results/artificial_groups'
os.makedirs(results_singles_dir, exist_ok=True)

STRING = ["nompCxCrimson", "WTxCrimson", "nompCxWT"]
colors = ["dodgerblue", "turquoise", "limegreen", "red"]
line_styles = ["-", "--"]  # solid for group, dotted for single

def process_csv(file_path):
    """Load and clean CSV for X, Y centroids, and frame."""
    data = pd.read_csv(file_path)

    # Convert X and Y centroids to numeric, dropping any invalid entries
    X_centroid = pd.to_numeric(data['X#wcentroid (cm)'], errors='coerce')
    Y_centroid = pd.to_numeric(data['Y#wcentroid (cm)'], errors='coerce')

    # Return centroids and frame data
    return X_centroid.dropna().values, Y_centroid.dropna().values


# Function to create artificial groups
def create_artificial_groups(base_directory, group_size=5):
    """Creates artificial groups by combining trajectories from single data, ensuring same condition for all singles."""
    condition_subdirectories = {condition: [] for condition in STRING}

    # Loop through each condition in the STRING list and look for corresponding folders
    for condition in STRING:
        for folder in os.listdir(base_directory):
            if condition in folder:  # Check if the folder name contains the condition string
                condition_dir = os.path.join(base_directory, folder, 'data')
                condition_subdirectories[condition].append(condition_dir)

    artificial_groups = []

    # Loop through each condition in STRING and create artificial groups for each
    for condition, subdirectories in condition_subdirectories.items():
        random.shuffle(subdirectories)

        # Group size determines how many subdirectories are grouped together in each artificial group
        while len(subdirectories) >= group_size:
            group_dirs = subdirectories[:group_size]
            subdirectories = subdirectories[group_size:]

            group_X, group_Y = [], []

            # Process each subdirectory in the current group
            for subdir in group_dirs:
                # Loop over all CSV files in the subdirectory
                for csv_file in os.listdir(subdir):
                    if csv_file.endswith('.csv'):  # Match any CSV file
                        csv_path = os.path.join(subdir, csv_file)
                        X_clean, Y_clean = process_csv(csv_path)
                        group_X.append(X_clean)
                        group_Y.append(Y_clean)

            artificial_groups.append((group_X, group_Y, condition))  # Add the processed group to the list

    return artificial_groups


# Function to calculate pairwise neighbor distances
def calculate_pairwise_distances_over_time(dataframes):
    """
    Calculate pairwise distances at each time point and return them as a list of lists.
    """
    num_time_points = len(dataframes[0])  # Assuming all dataframes have the same number of time points
    distances_over_time = []

    for time_point in range(num_time_points):
        positions_list = []

        # Extract positions for the current time point from each dataframe
        for df in dataframes:
            if time_point < len(df):  # Ensure time_point is within bounds
                positions = df.iloc[time_point][['X#wcentroid (cm)', 'Y#wcentroid (cm)']].values
                if not np.any(np.isnan(positions)) and not np.any(np.isinf(positions)):
                    positions_list.append(positions)

        # Compute pairwise distances for this time point
        time_point_distances = []
        for i in range(len(positions_list)):
            for j in range(i + 1, len(positions_list)):
                distance = euclidean(positions_list[i], positions_list[j])
                time_point_distances.append(distance)
        distances_over_time.append(time_point_distances)

    return distances_over_time


# Main processing loop for groups and artificial groups with time-based analysis
def process_groups_over_time(selected_lines=None):
    """
    Process groups and generate scatter plots for pairwise distances over time.
    :param selected_lines: List of tuples specifying which lines to plot, e.g., [("CS_WT", "actual"), ("sNPFxWT", "artificial")]
    """
    all_distances_actual = {condition: [] for condition in STRING}
    all_distances_artificial = {condition: [] for condition in STRING}

    # Process groups from the "Group" folder
    for condition in STRING:
        for folder in os.listdir(main_directory_group):
            if condition in folder:
                group_subdir = os.path.join(main_directory_group, folder, 'data')

                # Load CSV files for the group
                dataframes = []
                for csv_file in os.listdir(group_subdir):
                    if csv_file.endswith('.csv'):
                        csv_path = os.path.join(group_subdir, csv_file)
                        X, Y = process_csv(csv_path)
                        df = pd.DataFrame({'X#wcentroid (cm)': X, 'Y#wcentroid (cm)': Y})
                        dataframes.append(df)

                # Calculate pairwise distances over time for this group
                distances = calculate_pairwise_distances_over_time(dataframes)
                all_distances_actual[condition].append(distances)

    # Create artificial groups from the "Single" folder
    artificial_groups = create_artificial_groups(main_directory_single)
    for group_index, (group_X, group_Y, condition) in enumerate(artificial_groups):
        dataframes = []
        for X, Y in zip(group_X, group_Y):
            df = pd.DataFrame({'X#wcentroid (cm)': X, 'Y#wcentroid (cm)': Y})
            dataframes.append(df)

        # Calculate pairwise distances over time for this artificial group
        distances = calculate_pairwise_distances_over_time(dataframes)
        all_distances_artificial[condition].append(distances)

    # Prepare the plot
    plt.figure(figsize=(12, 8))

    if selected_lines is None:
        selected_lines = [(condition, "actual") for condition in STRING] + [(condition, "artificial") for condition in STRING]

    for condition, group_type in selected_lines:
        if group_type == "actual":
            # Adjust color for "actual" (group)
            color = mcolors.to_rgba(colors[STRING.index(condition)], alpha=1)
            size = 9
        elif group_type == "artificial":
            # Adjust color for "artificial" (singles, lighter)
            #color = mcolors.to_rgba(colors[STRING.index(condition)], alpha=0.5)
            color = "gray"
            size = 9

        # Aggregate distances over all groups for this condition
        distances_per_time = []
        for distances in (all_distances_actual[condition] if group_type == "actual" else all_distances_artificial[condition]):
            for t, dist_list in enumerate(distances):
                if len(distances_per_time) <= t:
                    distances_per_time.append([])
                distances_per_time[t].extend(dist_list)

        # Compute mean distances over time
        mean_distances = [np.mean(dists) if dists else np.nan for dists in distances_per_time]
        time_indices = range(len(mean_distances))
        PND = pd.DataFrame({
            "time": time_indices,
            "distances": mean_distances,
        })

        # Specify the file path
        file_path = "/Users/aljoscha/Downloads/PND2.csv"

        # Save the DataFrame to a CSV file
        PND.to_csv(file_path, index=False)
        # Plot scatter points
        plt.scatter(time_indices, mean_distances, label=f"{group_type.capitalize()} {condition}", color=color, s=size)

    plt.title('Pairwise Distances Over Time')
    plt.xlabel('Time (Frame Index)')
    plt.ylabel('Pairwise Distance (cm)')
    plt.legend(title='Condition and Genotype', loc='upper right')
    plt.tight_layout()
    plt.show()

# Example Usage
# Plot red (actual) and red dotted (artificial) lines over time
process_groups_over_time(selected_lines=[
            ("WTxCrimson", "actual"),
            ("WTxCrimson", "artificial"),
            ("nompCxCrimson", "actual"),
            ("nompCxCrimson", "artificial"),
            ("nompCxWT", "actual"),
            ("nompCxWT", "artificial")])

# Plot specific lines, e.g., only blue (actual) and green dotted (artificial)
# process_groups_over_time(selected_lines=[("CS_WT", "actual"), ("WTxCrimson", "artificial")])

