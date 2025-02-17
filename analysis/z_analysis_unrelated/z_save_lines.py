"""Generally helpful"""
# df.dropna(inplace=True)

# print(df.index.get_level_values('group_id').unique())
# plt.hlines(4, 0, 4, colors=None, linestyles='solid', label='', data=None)
# print("Index names:", df_final.index.names)
# print("Columns:", df_final.columns.tolist())

# print(df['encounter_count'].unique())
# print(df['encounter_count'].value_counts(0)[0])
# print(df['encounter_count'].value_counts(1))

# results_plt_specific_dir = os.path.join(condition_dir[2][1], 'specific')
#     os.makedirs(results_plt_specific_dir, exist_ok=True)

# map_RGN = create_mapping_actual_groups(
#     df_initial=df_initial,
#     condition=condition_dir[0][1]
# )
# map_RGN.to_pickle(os.path.join(condition_dir[2][0], "map_RGN.pkl"))

# for idx in data_files:
#     df = data_dict[idx]
#     print(f"--- {idx} ---")
#     print("Index names:", df.index.names)
#     print("Columns:", df.columns.tolist())
#     print("Shape:", df.shape)
# # print(data_dict['map_AGB']['group_id'].unique())

"""Encounter Extraction"""
# # Compute encounter frequency per individual
# df_final3['encounter_frequency'] = df_final3['encounter_count'] > 0
#
# # Identify encounter durations (frames with encounters grouped together)
# df_final3['encounter_event'] = df_final3['encounter_count'] > 0
# df_final3['encounter_duration'] = df_final3.groupby(['group_id', 'individual_id'])['encounter_event'].apply(lambda x: x.astype(int).groupby((x == 0).cumsum()).cumsum())
#
# # Filter durations over threshold
# df_final3['encounter_duration'] = df_final3['encounter_duration'].where(df_final3['encounter_duration'] <= MAX_DURATION_THRESHOLD, np.nan)
















# # Loop through file names and load them into the dictionary
# for name in data_files:
#     file_path = os.path.join(condition_dir[2][0], f"{name}.pkl")
#     data_dict[name] = pd.read_pickle(file_path)

# # Now you can access the data as:
# data_frame_initial = data_dict['data_frame_initial']
# map_RGN = data_dict['map_RGN']
# map_AIB = data_dict['map_AIB']
# map_AGB = data_dict['map_AGB']

# data_files = ['data_frame_initial', 'map_RGN', 'map_AIB', 'map_AGB']

data_dict = {}

# for i in range(len(mapping_list)):
#     data_dict[data_files[i]] = mapping_list[i]
# print(data_dict)
# mapping.to_pickle(os.path.join(condition_dir[2][0], f"{name}.pkl"))



"""hist for encounter debugging"""
# def plot_encounter_metrics(genotypes, freq_actual, freq_artificial, dur_actual, dur_artificial, selected_lines):
#     """
#     Plot histograms for encounter frequencies and durations.
#     """
#     plt.figure(figsize=(10, 6))
#
#     if selected_lines is None:
#         selected_lines = [(genotype, "actual") for genotype in genotypes] + [(genotype, "artificial") for genotype in genotypes]
#
#     # Plot encounter frequencies (Histogram)
#     for idx, (genotype, group_type) in enumerate(selected_lines):
#         if group_type == "actual":
#             plt.hist(
#                 freq_actual[genotype],
#                 bins=200,  # Adjust bins as needed
#                 alpha=0.6,
#                 label=f"Actual {genotype}",
#                 color=COLORS[0][idx % len(COLORS[0])],
#                 density=True  # Normalize to match KDE behavior
#             )
#         elif group_type == "artificial":
#             plt.hist(
#                 freq_artificial[genotype],
#                 bins=200,
#                 alpha=0.6,
#                 label=f"Artificial {genotype}",
#                 color=COLORS[1][idx % len(COLORS[1])],
#                 density=True
#             )
#
#     plt.title('Encounter Frequency Histogram')
#     plt.xlabel('Encounter Frequency')
#     plt.ylabel('Density')
#     plt.legend(title='Genotype and Group Type')
#     plt.tight_layout()
#     plt.show()
#
#     max_duration = 500  # Example: Set a cutoff for max encounter duration
#     bins = np.linspace(0, max_duration, 250)  # 30 bins from 0 to max_duration
#
#     # Plot encounter durations (Histogram)
#     plt.figure(figsize=(10, 6))
#     for idx, (genotype, group_type) in enumerate(selected_lines):
#         if group_type == "actual":
#             plt.hist(
#                 dur_actual[genotype],
#                 bins=bins,  # More bins for better resolution
#                 alpha=0.6,
#                 label=f"Actual {genotype}",
#                 color=COLORS[0][idx % len(COLORS[0])],
#                 density=True
#             )
#         elif group_type == "artificial":
#             plt.hist(
#                 dur_artificial[genotype],
#                 bins=bins,
#                 alpha=0.6,
#                 label=f"Artificial {genotype}",
#                 color=COLORS[1][idx % len(COLORS[1])],
#                 density=True
#             )
#
#     plt.title('Encounter Duration Histogram')
#     plt.xlabel('Encounter Duration (Frames)')
#     plt.ylabel('Density')
#     # Ensure that encounter durations are not empty before calling max()
#     max_actual_duration = max([max(d) for d in list(dur_actual.values()) if d], default=0)
#     max_artificial_duration = max([max(d) for d in list(dur_artificial.values()) if d], default=0)
#
#     plt.xlim(-100, 500)
#     plt.legend(title='Genotype and Group Type')
#     plt.tight_layout()
#     plt.show()

"""
group_id = "1_ID_WTxCrimson_0"
group_members = mapping_artificial_groups_bootstrapped[mapping_artificial_groups_bootstrapped['group_id'] == group_id]['sub_dir']
print(group_members)

group_data = df_initial.loc[df_initial.index.get_level_values('sub_dir').isin(group_members)]
print(group_data)

for idx in mapping_artificial_groups_bootstrapped['group_id'].unique():
    group_members = mapping_artificial_groups_bootstrapped[mapping_artificial_groups_bootstrapped['group_id'] == idx]['sub_dir']
    group_data = df_initial.loc[df_initial.index.get_level_values('sub_dir').isin(group_members)]
    print(group_data)
"""

"""
# Define the target condition and genotype
target_condition = "ConditionX"
target_genotype = "GenoA"

# Step 1: Filter df_initial to include only the target condition and genotype
df_filtered = df_initial[
    (df_initial.index.get_level_values('condition') == target_condition) &
    (df_initial.index.get_level_values('genotype') == target_genotype)
]

# Step 2: Identify sub_dirs that belong to the target condition & genotype
valid_sub_dirs = df_filtered.index.get_level_values('sub_dir').unique()

# Step 3: Filter df_artificial_groups_mapping to keep only groups containing these sub_dirs
filtered_groups = df_artificial_groups_mapping[
    df_artificial_groups_mapping['sub_dir'].isin(valid_sub_dirs)
]

# Step 4: Extract unique group_ids from the filtered groups
valid_group_ids = filtered_groups['group_id'].unique()

# Step 5: Iterate through only the valid group_ids and extract data
for idx in valid_group_ids:
    group_members = filtered_groups[filtered_groups['group_id'] == idx]['sub_dir']

    # Retrieve only relevant group data from df_filtered
    group_data = df_filtered.loc[df_filtered.index.get_level_values('sub_dir').isin(group_members)]

    print(f"Data for group {idx}:")
    print(group_data)
"""

"""actual grou creation df"""

# def create_actual_groups(df_initial, condition):
#     import pandas as pd
#     # actual_condition = condition_dir[0][1]  # Get the condition name for actual groups
#     df_filtered = df_initial[df_initial.index.get_level_values('condition') == condition]  # Filter data for actual groups
#
#     grouped_data_actual = []
#
#     # Iterate over each sub_dir in the filtered DataFrame
#     counter = 0
#     for sub_dir, sub_dir_df in df_filtered.groupby(level='sub_dir'):
#         sub_dir_df = sub_dir_df.copy()
#         geno_name = sub_dir_df.index.get_level_values('genotype').unique()[0]
#         sub_dir_df['group_id'] = f"5_{geno_name}_{counter}"  # Assign unique group_id based on sub_dir
#         grouped_data_actual.append(sub_dir_df)
#         counter += 1
#
#     # Concatenate all data for actual groups
#     df_actual_groups = pd.concat(grouped_data_actual, axis=0)
#
#     return df_actual_groups






#
# # Variable in Python
# input_value = "Hello from Python"
#
# # Pass the variable as an argument to the Bash script
# subprocess.run(["bash", "script.sh", input_value])



# Visualization
# for cond in condition:
#     for geno in genotype:
#         x_plt=df.xs((cond, geno), level = ['condition', 'genotype'])['X#wcentroid (cm)'].values
#         y_plt=df.xs((cond, geno), level=['condition', 'genotype'])['Y#wcentroid (cm)'].values
#         plt.scatter(x_plt, y_plt, s=1)
#         circle = plt.Circle((7, 7), dish_radius, color='red', fill=False, linewidth=2)
#         plt.gca().add_patch(circle)
#         plt.gca().set_aspect('equal', adjustable='box')
#         plt.xlim(0, 14)
#         plt.ylim(0, 14)
#         # plt.savefig(os.path.join(condition_dir[2], f"{cond}_{geno}.png"))
#     plt.show()
#         # plt.close()

# for cond in condition:
#     for geno in genotype:
#         df.dropna(inplace=True)
#         speed_plt = df.xs((cond, geno), level=['condition', 'genotype'])['SPEED#wcentroid (cm/s)'].values
#         midline_offset_plt = df.xs((cond, geno), level=['condition', 'genotype'])['MIDLINE_OFFSET'].values
#         # plt.figure(figsize=(8, 6))
#         plt.hist(speed_plt,bins=10000, color='blue', fill=False, alpha=0.3)
#         # sns.kdeplot(speed_plt, color='blue', fill=False, alpha=0.3)
#         # plt.hist(midline_offset_plt, bins=1000, color='red', fill=True, alpha=0.3)
#         # sns.kdeplot(data1, label=labels[0], color='blue', fill=False, alpha=0.3)
#         plt.xlim(-0.1, 1)
# plt.show()

#     plt.figure(figsize=(8, 6))
#     plt.title(title)
#     plt.xlabel(xlabel)
#     plt.ylabel('Density')
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(output_file)

# create artificial groups
# rename columns x/y
# add settings to plot
# plt Heatmaps
# plt PND hist
# plt PND KDE
# plt PND over time
# plt Encounter duration
# plt Encounter frequency
# decide on data to compare




# for i in range(len(grouped_df.index.get_level_values('group_id').unique())):
#     group_ID = 'AG_ID_WTxCrimson_' + str(i)
#
#     # Check if the group ID exists in the DataFrame's index
#     try:
#         df22 = grouped_df.xs(group_ID, level='group_id')
#         x_plt = df22['X#wcentroid (cm)'].values
#         y_plt = df22['Y#wcentroid (cm)'].values
#     except KeyError:
#         print(f"Group ID '{group_ID}' not found.")
#     plt.scatter(x_plt, y_plt, s=1)
#     circle = plt.Circle((7, 7), dish_radius, color='red', fill=False, linewidth=2)
#     plt.gca().add_patch(circle)
#     plt.gca().set_aspect('equal', adjustable='box')
#     plt.xlim(0, 14)
#     plt.ylim(0, 14)
#         # plt.savefig(os.path.join(condition_dir[2], f"{cond}_{geno}.png"))
#     plt.show()
#         # plt.close()


# # print(grouped_df.groupby(level='group_id').index.names)
# # print(grouped_df.groupby(level='group_id').columns.tolist())
# print(grouped_df.groupby(level='group_id').head())
# print(grouped_df.index.get_level_values('group_id').unique())

# ABC = df_filtered.xs('single', level='condition')
# print(123, ABC.index.get_level_values('genotype').unique())

# print(df_combined.index.names)
# print(df_combined.columns.tolist())
# # print(grouped_df.groupby(level='group_id').index.names)
# # print(grouped_df.groupby(level='group_id').columns.tolist())
# print(df_combined.groupby(level='group_id').head())
# print(df_combined.index.get_level_values('group_id').unique())


# df.dropna(inplace=True)









# df = pd.read_pickle('/Users/aljoscha/Downloads/2012_nompC_Crimson_WT_4min_50p_TRex_beta_2101/results/data_frame_final.pkl')

# access df.xs(('group', geno, 'actual', 'group_id'), level=['condition', 'genotype', 'group_type'])['x']
# access df.xs(('single', geno, 'artificial', 'group_id'), level=['condition', 'genotype', 'group_type'])['x']

# for group_id in df.index.get_level_values('group_id').unique():
#     for cond in condition:
#         if cond == "group":
#             group_type = "actual"
#         else:
#             group_type = "artificial"
#         for geno in genotype:
#             try:
#                 # Access x and y for the current group_id, cond, geno, and group_type
#                 x_plt = df.xs((cond, geno, group_type, group_id),
#                               level=['condition', 'genotype', 'group_type', 'group_id'])['x'].dropna().values
#                 y_plt = df.xs((cond, geno, group_type, group_id),
#                               level=['condition', 'genotype', 'group_type', 'group_id'])['y'].dropna().values
#             except KeyError:
#                 # Handle cases where the combination does not exist in the DataFrame
#                 continue


# for cond in condition:
#     for geno in genotype:
#         df.dropna(inplace=True)
#         speed_plt = df.xs((cond, geno), level=['condition', 'genotype'])['speed'].values
#         midline_offset_plt = df.xs((cond, geno), level=['condition', 'genotype'])['midline_offset'].values
#         # plt.figure(figsize=(8, 6))
#         plt.hist(speed_plt,bins=5000, color='blue', fill=True, alpha=0.3)
#         # sns.kdeplot(speed_plt, color='blue', fill=False, alpha=0.3)
#         # plt.hist(midline_offset_plt, bins=1000, color='red', fill=True, alpha=0.3)
#         # sns.kdeplot(data1, label=labels[0], color='blue', fill=False, alpha=0.3)
#         title = f"{cond} {geno}"
#         plt.title(title)
#         plt.xlim(-0.1, 1)
# plt.show()

'''DF Merging'''
# df_groups['group_type'] = 'actual'
# df_artificial_groups['group_type'] = 'artificial'
#
# # Concatenate the two DataFrames
# df_final = pd.concat([df_groups, df_artificial_groups], axis=0)
# # Make 'group_type' part of the index
# df_final.set_index('group_type', append=True, inplace=True)
# df_final = df_final.reorder_levels(['sub_dir', 'condition', 'genotype', 'group_type', 'group_id', 'frame'])
# # Sort the DataFrame by the new index for consistent organization
# df_final.sort_index(inplace=True)

# # Reset the index for both dataframes to enable comparison
# df_reset = df_initial.reset_index()
# df_actual_groups_reset = df_actual_groups.reset_index()
#
# # Ensure the same multi-index is used for both DataFrames
# df_actual_groups_reset.set_index(['sub_dir', 'condition', 'genotype', 'frame'], inplace=True)
# df_reset.set_index(['sub_dir', 'condition', 'genotype', 'frame'], inplace=True)
#
# # Now concatenate the DataFrames
# df_groups = pd.concat([
#     df_actual_groups_reset,
#     df_reset[~df_reset.index.isin(df_actual_groups_reset.index)]
# ], axis=0)
#
# # Set the group_id back as part of the MultiIndex
# df_groups.set_index('group_id', append=True, inplace=True)

'''print test'''
# print(df_initial.index.names)
# print(df_initial.columns.tolist())
# print(len(df_initial))
# print(df_initial.head())
# test = df_initial.groupby(['condition'])
# test2 = df_initial.groupby(['condition', 'genotype'])
# print(test.head())
# print(test2.head())


# print(df_artificial_groups_mapping['group_id'].unique())
# sub_dirs = geno_df.index.get_level_values('sub_dir').unique()

'''AGC_bootstrap_df'''
# def create_artificial_groups_bootstrapped(df, group_size, bootstrap_reps=2, condition_dir=None):
#     """
#     Creates artificial groups by bootstrapping within genotypes, but only for data where
#     condition == condition_dir[1][0]. Each trial is grouped with group_size - 1
#     additional trials, ensuring no duplicate trials within a group. The bootstrapping is applied bootstrap_reps times.
#
#     Args:
#         df (pd.DataFrame): Multi-indexed DataFrame with levels ['sub_dir', 'condition', 'genotype', 'frame'].
#         group_size (int): Size of each group.
#         bootstrap_reps (int): Number of times to apply bootstrapping.
#         condition_dir (list): A list containing the condition values, used to filter data.
#
#     Returns:
#         pd.DataFrame: A DataFrame containing the groups, allowing independent access by group IDs.
#     """
#     if not isinstance(df.index, pd.MultiIndex):
#         raise ValueError(
#             "The input DataFrame must have a MultiIndex with levels ['sub_dir', 'condition', 'genotype', 'frame'].")
#
#     if 'genotype' not in df.index.names:
#         raise ValueError("The input DataFrame must have a 'genotype' level in its MultiIndex.")
#
#     if condition_dir is None or len(condition_dir) < 2:
#         raise ValueError("condition_dir must be provided with at least two values.")
#     # Filter data for condition == condition_dir[1][0]
#     condition_value = condition_dir[1][1]
#     df_filtered = df[df.index.get_level_values('condition') == condition_value]
#
#     grouped_data = []
#
#     # Iterate over each genotype in the filtered dataframe
#     for geno, geno_df in df_filtered.groupby(level='genotype'):
#         sub_dirs = geno_df.index.get_level_values('sub_dir').unique()
#         group_id_counter = 0
#
#         if len(sub_dirs) < group_size:
#             raise ValueError(f"Not enough trials for genotype {geno} to form groups of size {group_size}.")
#
#         for _ in range(bootstrap_reps):
#             for base_sub_dir in sub_dirs:
#                 # Randomly select group_size - 1 other trials
#                 other_sub_dirs = random.sample(
#                     [sub for sub in sub_dirs if sub != base_sub_dir], group_size - 1
#                 )
#
#                 group_sub_dirs = [base_sub_dir] + other_sub_dirs
#
#                 # Extract data for the selected sub_dirs
#                 group_df = geno_df.loc[geno_df.index.get_level_values('sub_dir').isin(group_sub_dirs)]
#                 group_df = group_df.copy()
#
#                 if 'group_id' in group_df.index.names:
#                     if group_df.index.get_level_values('group_id').isna().all():
#                         # Override the existing 'NaN' group_id
#                         group_df = group_df.reset_index('group_id')  # Drop the existing 'group_id' index level
#                         group_df['group_id'] = f"1_ID_{geno}_{group_id_counter}"  # Assign new group_id
#                         group_df.set_index('group_id', append=True, inplace=True)
#                 else:
#                     # Add 'group_id' if it is not already in the index
#                     group_df['group_id'] = f"1_ID_{geno}_{group_id_counter}"
#                     group_df.set_index('group_id', append=True, inplace=True)
#
#                 grouped_data.append(group_df)
#                 group_id_counter += 1
#
#     # Concatenate all grouped data into a single DataFrame
#     result_df = pd.concat(grouped_data, axis=0)
#     # Ensure the group_id is part of the index without creating duplicates
#     if 'group_id' not in result_df.index.names:
#         result_df.set_index('group_id', append=True, inplace=True)
#     return result_df


