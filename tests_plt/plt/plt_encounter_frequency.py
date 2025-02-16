import pandas as pd
from package.config_settings import group_type

df = pd.read_pickle('/Users/aljoscha/Downloads/2012_nompC_Crimson_WT_4min_50p_TRex_beta_2101/results/data/df_group_parameters.pkl')
print("Index names:", df.index.names)
print("Columns:", df.columns.tolist())
df1 = df.groupby('group_type')
for group in group_type:
    df2 = df1.get_group(group)['encounter_count']
    print(group)
    print(0, df2.value_counts(0)[0], 1, df2.value_counts(0)[1])
    print(0, df2.value_counts(1)[0], 1, df2.value_counts(1)[1])

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define thresholds
encounter_duration_threshold = [10, 1800]
frames_per_minute = 1800

# Example: Assuming df is already sorted by index and frame
df = df.sort_index()

# Identify encounter start and end
df['encounter_start'] = (df['encounter_count'].diff() == 1)
df['encounter_end'] = (df['encounter_count'].diff() == -1)

# Assign an encounter ID to each encounter (continuous blocks of 1s)
df['encounter_id'] = df['encounter_count'] * (df['encounter_count'].diff() != 0).cumsum()
df.loc[df['encounter_count'] == 0, 'encounter_id'] = np.nan  # Remove non-encounters

# Group by encounter_id and calculate duration
encounter_durations = df.groupby('encounter_id').size()
encounter_durations = encounter_durations[
    (encounter_durations >= encounter_duration_threshold[0]) &
    (encounter_durations <= encounter_duration_threshold[1])
]

# Calculate encounter frequency per minute
# Calculate encounter frequency per minute (handling frame as an index)
encounter_starts = df[df['encounter_start']].groupby(level=['sub_dir', 'condition', 'genotype', 'group_type', 'group_id', 'individual_id'])
encounter_frequency = encounter_starts.apply(lambda x: x.index.get_level_values('frame').to_series().diff().fillna(frames_per_minute).lt(frames_per_minute).sum())
# Plot KDE
plt.figure(figsize=(12, 5))

# KDE for Encounter Duration
plt.subplot(1, 2, 1)
sns.kdeplot(encounter_durations, fill=True)
plt.xlabel("Encounter Duration (frames)")
plt.ylabel("Density")
plt.title("KDE Plot of Encounter Durations")

# KDE for Encounter Frequency
plt.subplot(1, 2, 2)
sns.kdeplot(encounter_frequency, fill=True)
plt.xlabel("Encounter Frequency (per minute)")
plt.ylabel("Density")
plt.title("KDE Plot of Encounter Frequency")

plt.tight_layout()
plt.show()

# import seaborn as sns
# import matplotlib.pyplot as plt
#
# # Get unique group types
# group_types = ["RGN", "AIB", "AGB"]
#
# # Prepare a figure with subplots
# plt.figure(figsize=(12, 5))
#
# # --- Encounter Duration KDE ---
# plt.subplot(1, 2, 1)
# for group in group_types:
#     # Filter df for the specific group type and get valid encounter durations
#     group_df = df.xs(group, level="group_type", drop_level=False)
#     valid_encounter_ids = group_df['encounter_id'].dropna().unique()
#
#     # Select durations corresponding to valid encounter IDs
#     durations = encounter_durations.loc[encounter_durations.index.intersection(valid_encounter_ids)]
#
#     sns.kdeplot(durations, fill=True, label=group)
#
# plt.xlabel("Encounter Duration (frames)")
# plt.ylabel("Density")
# plt.title("Encounter Duration KDE")
# plt.legend()
#
# # --- Encounter Frequency KDE ---
# plt.subplot(1, 2, 2)
# for group in group_types:
#     # Filter encounter frequency for the given group_type
#     freqs = encounter_frequency.xs(group, level="group_type", drop_level=False)
#
#     sns.kdeplot(freqs, fill=True, label=group)
#
# plt.xlabel("Encounter Frequency (per minute)")
# plt.ylabel("Density")
# plt.title("Encounter Frequency KDE")
# plt.legend()
#
# plt.tight_layout()
# plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Get unique group types
group_types = ["RGN", "AIB"]

# Define dataset length (number of frames per dataset)
dataset_length = 7192  # Frames per dataset

# Prepare a figure with subplots
plt.figure(figsize=(12, 5))

# Dictionary to store dataset counts (N)
dataset_counts = {}

# --- Encounter Duration KDE ---
plt.subplot(1, 2, 1)
for group in group_types:
    # Filter df for the specific group type
    group_df = df.xs(group, level="group_type", drop_level=False)

    # Determine number of datasets used
    if group == "RGN":
        dataset_counts[group] = group_df.index.get_level_values("individual_id").nunique()
    else:  # AIB & AGB -> Estimate dataset count using frame count
        total_frames = len(group_df)  # Total number of frames in bootstrapped dataset
        dataset_counts[group] = int(np.round(total_frames / dataset_length))  # Estimate dataset count

    # Get valid encounter durations
    valid_encounter_ids = group_df['encounter_id'].dropna().unique()
    durations = encounter_durations.loc[encounter_durations.index.intersection(valid_encounter_ids)]

    if len(durations) > 1:
        sns.kdeplot(durations, fill=True, label=f"{group} (N={dataset_counts[group]})")
    else:
        sns.histplot(durations, bins=20, kde=False, label=f"{group} (N={dataset_counts[group]})", alpha=0.5)

plt.xlabel("Encounter Duration (frames)")
plt.ylabel("Density")
plt.title("Encounter Duration KDE (Normalized)")
plt.legend()

# --- Encounter Frequency KDE ---
plt.subplot(1, 2, 2)
for group in group_types:
    # Get encounter frequencies per group
    freqs = encounter_frequency.xs(group, level="group_type", drop_level=False)

    if len(freqs) > 1:
        sns.kdeplot(freqs, fill=True, label=f"{group} (N={dataset_counts[group]})")
    else:
        sns.histplot(freqs, bins=20, kde=False, label=f"{group} (N={dataset_counts[group]})", alpha=0.5)

plt.xlabel("Encounter Frequency (per minute)")
plt.ylabel("Density")
plt.title("Encounter Frequency KDE (Normalized)")
plt.legend()

plt.tight_layout()
plt.show()

# Print the number of datasets used for the plots
print("\nNumber of Datasets Used per Group Type:")
for group, count in dataset_counts.items():
    print(f"{group}: {count} datasets")

for group in group_types:
    group_df = df.xs(group, level="group_type", drop_level=False)
    valid_encounter_ids = group_df['encounter_id'].dropna().unique()
    durations = encounter_durations.loc[encounter_durations.index.intersection(valid_encounter_ids)]

    print(f"{group}: {len(durations)} valid encounters")

for group in group_types:
    print(f"\n=== {group} ===")

    # Filter df for the specific group type
    group_df = df.xs(group, level="group_type", drop_level=False)
    valid_encounter_ids = group_df['encounter_id'].dropna().unique()

    # Get durations for valid encounter IDs
    group_durations = encounter_durations.loc[encounter_durations.index.intersection(valid_encounter_ids)]

    # Get all raw encounter durations before filtering
    all_durations = df.groupby("encounter_id").size()
    all_group_durations = all_durations.loc[all_durations.index.intersection(valid_encounter_ids)]

    # Count before and after filtering
    print(f"Total encounters (before filtering): {len(all_group_durations)}")
    print(f"Valid encounters (after filtering): {len(group_durations)}")

    if len(all_group_durations) > 0:
        print(f"Min Duration: {all_group_durations.min()} frames")
        print(f"Max Duration: {all_group_durations.max()} frames")
        print(f"Mean Duration: {all_group_durations.mean():.2f} frames")
        print(f"Median Duration: {all_group_durations.median()} frames")

        # Show some removed encounters (too short/long)
        removed_encounters = all_group_durations[
            (all_group_durations < encounter_duration_threshold[0]) |
            (all_group_durations > encounter_duration_threshold[1])
            ]
        print(f"Removed Encounters (too short/long): {len(removed_encounters)}")

        if not removed_encounters.empty:
            print("Examples of removed encounters:")
            print(removed_encounters.head(5))

    else:
        print("No encounters detected for this group.")