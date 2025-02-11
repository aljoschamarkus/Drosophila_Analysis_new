import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.spatial.distance import euclidean
import random
import numpy as np

# Paths and settings
main_directory_group = '/Users/aljoscha/Downloads/2012_nompC_Crimson_WT_4min_50p_TRex_beta_2101/group'
main_directory_single = '/Users/aljoscha/Downloads/2012_nompC_Crimson_WT_4min_50p_TRex_beta_2101/single'
results_singles_dir = '/Users/aljoscha/Downloads/Data_sorted/All_results/artificial_groups'
os.makedirs(results_singles_dir, exist_ok=True)

# Constants
STRING = ["nompCxCrimson", "WTxCrimson", "nompCxWT"]
colors = ['#e41a1c', '#377eb8', '#4daf4a']
distance_threshold = 47  # Distance threshold for encounters
frame_limit = 7190  # Number of frames to analyze

def process_csv(file_path):
    """Load and clean CSV for X, Y centroids, and SPEED."""
    data = pd.read_csv(file_path)

    # Limit to the first 5400 rows
    data = data.iloc[:5400]

    # Convert columns to numeric, dropping invalid entries
    X_centroid = pd.to_numeric(data['X#wcentroid (cm)'], errors='coerce')
    Y_centroid = pd.to_numeric(data['Y#wcentroid (cm)'], errors='coerce')

    # Drop NaN and Inf values explicitly
    X_centroid = X_centroid[~X_centroid.isin([np.inf, -np.inf])]
    Y_centroid = Y_centroid[~Y_centroid.isin([np.inf, -np.inf])]

    # Return cleaned centroids
    return (
        X_centroid.dropna().values,
        Y_centroid.dropna().values
    )

def create_bootstrapped_groups(base_directory, group_size=5):
    """Creates artificial groups by cycling through directories and randomly selecting others to form groups."""
    condition_subdirectories = {condition: [] for condition in STRING}

    # Organize subdirectories by condition
    for condition in STRING:
        for folder in os.listdir(base_directory):
            if condition in folder:
                condition_dir = os.path.join(base_directory, folder, 'data')
                condition_subdirectories[condition].append(condition_dir)

    artificial_groups = []

    # Create artificial groups for each condition
    for condition, subdirectories in condition_subdirectories.items():
        if len(subdirectories) < group_size:
            raise ValueError(f"Not enough directories for condition {condition} to form a group.")

        for i, base_dir in enumerate(subdirectories):
            # Randomly select group_size - 1 other directories
            other_dirs = random.sample(
                [d for j, d in enumerate(subdirectories) if j != i], group_size - 1
            )
            group_dirs = [base_dir] + other_dirs

            group_X, group_Y = [], []

            for subdir in group_dirs:
                for csv_file in os.listdir(subdir):
                    if csv_file.endswith('.csv'):
                        csv_path = os.path.join(subdir, csv_file)
                        X_clean, Y_clean = process_csv(csv_path)
                        group_X.append(X_clean)
                        group_Y.append(Y_clean)

            artificial_groups.append((group_X, group_Y, condition))

    return artificial_groups


# --- Utility Functions ---
def calculate_encounter_frequency(dataframes, distance_threshold=1):
    """Calculate the frequency of encounters (pairs within threshold)."""
    pair_combinations = list(combinations(dataframes, 2))
    encounter_count = 0
    total_count = 0

    for df1, df2 in pair_combinations:
        for frame in range(min(len(df1), len(df2))):
            x1, y1 = df1.iloc[frame]['X#wcentroid (cm)'], df1.iloc[frame]['Y#wcentroid (cm)']
            x2, y2 = df2.iloc[frame]['X#wcentroid (cm)'], df2.iloc[frame]['Y#wcentroid (cm)']
            distance = euclidean([x1, y1], [x2, y2])
            total_count += 1
            if distance <= distance_threshold:
                encounter_count += 1

    return encounter_count / total_count if total_count > 0 else 0


def calculate_encounter_duration(dataframes, distance_threshold=1):
    """Calculate the duration of encounters (continuous frames within threshold)."""
    pair_combinations = list(combinations(dataframes, 2))
    durations = []

    for df1, df2 in pair_combinations:
        current_duration = 0
        for frame in range(min(len(df1), len(df2))):
            x1, y1 = df1.iloc[frame]['X#wcentroid (cm)'], df1.iloc[frame]['Y#wcentroid (cm)']
            x2, y2 = df2.iloc[frame]['X#wcentroid (cm)'], df2.iloc[frame]['Y#wcentroid (cm)']
            distance = euclidean([x1, y1], [x2, y2])

            if distance <= distance_threshold:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                current_duration = 0
        if current_duration > 0:
            durations.append(current_duration)

    return durations


def process_groups(directory, conditions, is_actual=True):
    """Process actual or artificial groups and return encounter metrics."""
    group_metrics = {condition: [] for condition in conditions}

    # Process actual groups
    if is_actual:
        for condition in conditions:
            for folder in os.listdir(directory):
                if condition in folder:
                    group_subdir = os.path.join(directory, folder, 'data')
                    dataframes = []
                    for csv_file in os.listdir(group_subdir):
                        if csv_file.endswith('.csv'):
                            X, Y = process_csv(os.path.join(group_subdir, csv_file))
                            df = pd.DataFrame({'X#wcentroid (cm)': X, 'Y#wcentroid (cm)': Y})
                            dataframes.append(df)
                    group_metrics[condition].append((dataframes))
    # Process artificial groups
    else:
        artificial_groups = create_bootstrapped_groups(directory, group_size=5)  # Use the new function
        for group_X, group_Y, condition in artificial_groups:
            dataframes = []
            for X, Y in zip(group_X, group_Y):
                df = pd.DataFrame({'X#wcentroid (cm)': X, 'Y#wcentroid (cm)': Y})
                dataframes.append(df)
            group_metrics[condition].append((dataframes))

    return group_metrics


# --- Plotting Functions ---
def plot_encounter_analysis(metrics_actual, metrics_artificial, selected_lines):
    """Plot KDE for encounter frequencies and durations for selected pairs of conditions in separate plots."""

    # Plot Encounter Frequency KDE
    plt.figure(figsize=(10, 6))
    for condition, group_type in selected_lines:
        # Choose color and line style based on the condition and group type
        color = colors[STRING.index(condition)]
        if group_type == 'actual':
            metrics = metrics_actual
            label_suffix = 'Actual'
            linestyle = '-'  # Continuous line for actual
        else:
            metrics = metrics_artificial
            label_suffix = 'Artificial'
            linestyle = '--'  # Dotted line for artificial

        if condition in metrics:
            # Aggregate data for the selected condition and type
            frequencies = [calculate_encounter_frequency(group) for group in metrics[condition]]

            # Plot Frequency KDE
            sns.kdeplot(frequencies, label=f'{condition} {label_suffix}', color=color, fill=True, alpha=0.3,
                        clip=(0, None), linestyle=linestyle)

    # Finalize Frequency Plot
    plt.title('Encounter Frequency KDEs')
    plt.xlabel('Encounter Frequency')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

    # Plot Encounter Duration KDE
    plt.figure(figsize=(10, 6))
    for condition, group_type in selected_lines:
        # Choose color and line style based on the condition and group type
        color = colors[STRING.index(condition)]
        if group_type == 'actual':
            metrics = metrics_actual
            label_suffix = 'Actual'
            linestyle = '-'  # Continuous line for actual
        else:
            metrics = metrics_artificial
            label_suffix = 'Artificial'
            linestyle = '--'  # Dotted line for artificial

        if condition in metrics:
            # Aggregate data for the selected condition and type
            durations = [duration for group in metrics[condition] for duration in calculate_encounter_duration(group)]

            # Plot Duration KDE
            sns.kdeplot(durations, label=f'{condition} {label_suffix} Duration', color=color, linestyle=linestyle,
                        fill=True, alpha=0.3, clip=(0, None))

    # Finalize Duration Plot
    plt.title('Encounter Duration KDEs')
    plt.xlabel('Encounter Duration (Frames)')
    plt.ylabel('Density')
    plt.legend()

    # Add the ylim setting for the y-axis
    # plt.xlim(0, 500)  # Set the y-axis limit for the encounter duration plot

    plt.show()


# --- Main Execution ---
def main():
    # Define the conditions you want to compare (selected pairs of condition and group type)
    # selected_lines = [("nompCxWT", "actual")]  # Example selection
    # selected_lines = [("nompCxCrimson", "actual"), ("WTxCrimson", "actual"), ("nompCxCrimson", "artificial"), ("WTxCrimson", "artificial"), ("nompCxWT", "actual"), ("nompCxWT", "artificial")]  # Example selection
    selected_lines = [("nompCxCrimson", "actual"), ("nompCxWT", "actual"), ("WTxCrimson", "actual")]  # Example selection

    # Process actual and artificial groups for each condition
    print("Processing actual groups...")
    metrics_actual = process_groups(main_directory_group, conditions=STRING, is_actual=True)

    print("Creating and processing artificial groups...")
    metrics_artificial = process_groups(main_directory_single, conditions=STRING, is_actual=False)

    # Plot encounter analysis for the selected pairs
    print("Plotting encounter analysis...")
    plot_encounter_analysis(metrics_actual, metrics_artificial, selected_lines)


if __name__ == "__main__":
    main()