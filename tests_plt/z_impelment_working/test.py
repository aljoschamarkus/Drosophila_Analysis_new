import pandas as pd
import os
import random
import numpy as np
from scipy.spatial.distance import euclidean
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

main_directory_group = '/Users/aljoscha/Downloads/2012_nompC_Crimson_WT_4min_50p_TRex_beta_2101/group'
main_directory_single = '/Users/aljoscha/Downloads/2012_nompC_Crimson_WT_4min_50p_TRex_beta_2101/single'
results_singles_dir = '/Users/aljoscha/Downloads/Data_sorted/All_results/artificial_groups'
os.makedirs(results_singles_dir, exist_ok=True)

STRING = ["nompCxCrimson", "WTxCrimson", "nompCxWT"]
colors = ["dodgerblue", "turquoise", "limegreen", "red"]
line_styles = ["-", "--"]  # solid for group, dotted for single

# Function to calculate pairwise distances over speed
def calculate_pairwise_distances_over_speed(dataframes):
    """
    Calculate pairwise distances at each time point and associate them with speed values.
    We ensure that distances and speeds are matched by 'frame' index.
    """
    num_time_points = len(dataframes[0])
    distances_over_speed = []
    speeds = []

    # Process each frame across all dataframes
    for time_point in range(num_time_points):
        positions_list = []
        speed_list = []
        frame_list = []

        for df in dataframes:
            if time_point < len(df):
                frame = df.iloc[time_point]['frame']
                positions = df.iloc[time_point][['X#wcentroid (cm)', 'Y#wcentroid (cm)']].values
                speed = df.iloc[time_point]['SPEED#wcentroid (cm/s)']

                if not np.any(np.isnan(positions)) and not np.any(np.isinf(positions)) and not np.isnan(speed):
                    positions_list.append(positions)
                    speed_list.append(speed)
                    frame_list.append(frame)

        # Now only calculate pairwise distances when we have at least two points
        if len(positions_list) > 1:
            # Compute pairwise distances for the positions
            time_point_distances = [euclidean(positions_list[i], positions_list[j])
                                    for i in range(len(positions_list))
                                    for j in range(i + 1, len(positions_list))]

            # Multiply the speed values by the number of distances
            speeds.extend(speed_list * (len(positions_list) - 1))  # Each speed corresponds to each pair
            distances_over_speed.extend(time_point_distances)

    return distances_over_speed, speeds


# Main processing function
def process_groups_over_speed(selected_lines=None):
    """
    Process groups and generate scatter plots for pairwise distances over speed.
    Only plot pairs where there is a matching frame index for both distance and speed.
    """
    all_distances_actual = {condition: [] for condition in STRING}
    all_speeds_actual = {condition: [] for condition in STRING}

    # Iterate over each condition
    for condition in STRING:
        for folder in os.listdir(main_directory_group):
            if condition in folder:
                group_subdir = os.path.join(main_directory_group, folder, 'data')
                dataframes = []

                # Read all CSV files for the group
                for csv_file in os.listdir(group_subdir):
                    if csv_file.endswith('.csv'):
                        csv_path = os.path.join(group_subdir, csv_file)
                        df = pd.read_csv(csv_path)
                        dataframes.append(df)

                distances, speeds = calculate_pairwise_distances_over_speed(dataframes)
                all_distances_actual[condition].extend(distances)
                all_speeds_actual[condition].extend(speeds)

    # Create the scatter plot
    plt.figure(figsize=(12, 8))
    for condition, group_type in selected_lines:
        color = mcolors.to_rgba(colors[STRING.index(condition)], alpha=1) if group_type == "actual" else "gray"

        # Ensure that for each condition we only plot matching speeds and distances
        x_data = all_speeds_actual[condition]
        y_data = all_distances_actual[condition]

        if len(x_data) == len(y_data):
            # If lengths match, plot the points
            plt.scatter(x_data, y_data, label=f"{group_type.capitalize()} {condition}", color=color, s=9)
        else:
            print(f"Warning: Mismatch in lengths for {condition}. Skipping this condition.")

    plt.title('Pairwise Distances Over Speed')
    plt.xlabel('Speed (cm/s)')
    plt.ylabel('Pairwise Distance (cm)')
    plt.legend(title='Condition and Genotype', loc='upper right')
    plt.tight_layout()
    plt.show()

# Example usage
process_groups_over_speed(selected_lines=[
    ("WTxCrimson", "actual"),
    ("nompCxCrimson", "actual"),
    ("nompCxWT", "actual")
])
