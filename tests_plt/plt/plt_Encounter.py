import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from concurrent.futures import ProcessPoolExecutor
import numpy as np

# Constants
# COLORS = [['#e41a1c', '#377eb8', '#4daf4a'], ['#fbb4ae', '#a6cee3', '#b2df8a']]
DISTANCE_THRESHOLD = 0.5  # Distance threshold for encounters

COLORS = ['red', 'orange', 'blue', 'cyan', 'green', 'limegreen']

MIN_ENCOUNTER_DURATION = 15
MAX_DURATION_THRESHOLD = 1200

# --- Utility Functions ---
def calculate_encounter_metrics(df, group_id, distance_threshold):
    group_df = df.xs(group_id, level='group_id')
    distances_per_frame = []

    for frame in group_df.index.get_level_values('frame').unique():
        frame_df = group_df.xs(frame, level='frame')[['x', 'y']].dropna()
        if len(frame_df) > 1:
            pairwise_distances = pdist(frame_df.values)
            distances_per_frame.append(pairwise_distances)

    all_distances = np.concatenate(distances_per_frame) if distances_per_frame else []
    encounter_frequency = np.mean(all_distances <= distance_threshold) if len(all_distances) > 0 else 0

    # Calculate encounter durations
    encounter_durations = []
    current_duration = 0

    for distances in distances_per_frame:
        if np.any(distances <= distance_threshold):
            current_duration += 1
        else:
            if current_duration > 0:
                encounter_durations.append(current_duration)
            current_duration = 0

    if current_duration > 0:
        encounter_durations.append(current_duration)

    # Apply the filter to remove long durations
    encounter_durations = [d for d in encounter_durations if d <= MAX_DURATION_THRESHOLD]

    return encounter_frequency, encounter_durations


def process_group(group_id, df, distance_threshold):
    """
    Process a single group to calculate encounter metrics.
    """
    group_df = df.xs(group_id, level='group_id')
    genotype = group_df.index.get_level_values('genotype')[0]
    group_type = group_df.index.get_level_values('group_type')[0]

    # Calculate metrics
    encounter_frequency, encounter_durations = calculate_encounter_metrics(df, group_id, distance_threshold)

    return genotype, group_type, encounter_frequency, encounter_durations


def process_groups_parallel(df, distance_threshold, selected_lines=None):
    """
    Process groups in parallel and plot KDEs for encounter metrics.
    """
    genotypes = df.index.get_level_values('genotype').unique()

    # Initialize dictionaries for results
    encounter_frequencies_actual = {genotype: [] for genotype in genotypes}
    encounter_frequencies_artificial = {genotype: [] for genotype in genotypes}
    encounter_durations_actual = {genotype: [] for genotype in genotypes}
    encounter_durations_artificial = {genotype: [] for genotype in genotypes}

    # Process groups in parallel
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(
            process_group,
            df.index.get_level_values('group_id').unique(),
            [df] * len(df.index.get_level_values('group_id').unique()),
            [distance_threshold] * len(df.index.get_level_values('group_id').unique())
        ))

    # Collect results
    for genotype, group_type, frequency, durations in results:
        if group_type == "actual":
            encounter_frequencies_actual[genotype].append(frequency)
            encounter_durations_actual[genotype].extend(durations)
        elif group_type == "artificial":
            encounter_frequencies_artificial[genotype].append(frequency)
            encounter_durations_artificial[genotype].extend(durations)

    # Plot results
    plot_encounter_metrics(
        genotypes,
        encounter_frequencies_actual,
        encounter_frequencies_artificial,
        encounter_durations_actual,
        encounter_durations_artificial,
        selected_lines
    )


def plot_encounter_metrics(genotypes, freq_actual, freq_artificial, dur_actual, dur_artificial, selected_lines):
    """
    Plot KDEs for encounter frequencies and durations.
    """
    plt.figure(figsize=(10, 6))

    if selected_lines is None:
        selected_lines = [(genotype, "actual") for genotype in genotypes] + [(genotype, "artificial") for genotype in genotypes]

    # # Plot encounter frequencies
    # for idx, (genotype, group_type) in enumerate(selected_lines):
    #     if group_type == "actual":
    #         sns.kdeplot(
    #             freq_actual[genotype],
    #             label=f"Actual {genotype}",
    #             color=COLORS[0][idx % len(COLORS)[0]],
    #             linestyle='-'
    #         )
    #     elif group_type == "artificial":
    #         sns.kdeplot(
    #             freq_artificial[genotype],
    #             label=f"Artificial {genotype}",
    #             color=COLORS[1][idx % len(COLORS[1])],
    #             linestyle='--'
    #         )
    #
    # plt.title('Encounter Frequency KDEs')
    # plt.xlabel('Encounter Frequency')
    # plt.ylabel('Density')
    # plt.legend(title='Genotype and Group Type')
    # plt.tight_layout()
    # plt.show()

    # Plot encounter durations
    plt.figure(figsize=(10, 6))
    for idx, (genotype, group_type) in enumerate(selected_lines):
        if group_type == "actual":
            if genotype == 'WTxCrimson':
                COLOR = COLORS[0]
            elif genotype == 'nompCxCrimson':
                COLOR = COLORS[2]
            elif genotype == 'nompCxWT':
                COLOR = COLORS[4]

            sns.kdeplot(
                dur_actual[genotype],
                label=f"Actual {genotype}",
                color=COLOR,
                linestyle='-'
            )
            sns.kdeplot(
                dur_actual[genotype],
                label=f"Actual {genotype}",
                color=COLOR,
                linestyle='-'
            )
        elif group_type == "artificial":
            if genotype == 'WTxCrimson':
                COLOR = COLORS[1]
            elif genotype == 'nompCxCrimson':
                COLOR = COLORS[3]
            elif genotype == 'nompCxWT':
                COLOR = COLORS[5]
            sns.kdeplot(
                dur_artificial[genotype],
                label=f"Artificial {genotype}",
                color=COLOR,
                linestyle='--'
            )

    plt.title('Encounter Duration KDEs')
    plt.xlabel('Encounter Duration (Frames)')
    plt.ylabel('Density')
    plt.xlim(-200, 400)
    plt.legend(title='Genotype and Group Type')
    plt.tight_layout()
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    # Load your DataFrame
    df_final = pd.read_pickle(
        '/Users/aljoscha/Downloads/2012_nompC_Crimson_WT_4min_50p_TRex_beta_2101/results/data_frame_final.pkl')

    # Process and plot encounter metrics
    process_groups_parallel(
        df_final,
        distance_threshold=DISTANCE_THRESHOLD,
        selected_lines=[
            ("WTxCrimson", "actual"),
            ("WTxCrimson", "artificial"),
            ("nompCxCrimson", "actual"),
            ("nompCxCrimson", "artificial"),
            ("nompCxWT", "actual"),
            ("nompCxWT", "artificial")
        ]
    )
