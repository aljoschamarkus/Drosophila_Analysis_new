"""Generally helpful"""
# df.dropna(inplace=True)
# print("Index names:", df.index.names)
# print("Columns:", df.columns.tolist())
# print(df['encounter_count'].value_counts(0)[0])
# print(df.index.get_level_values('group_id').unique())

########################################################################################################################
"""get_group()"""
# df1 = df.groupby('group_type')
# df2 = df1.get_group(group)['encounter_count']

########################################################################################################################
"""plot circles"""
#     circle = plt.Circle((7, 7), dish_radius, color='red', fill=False, linewidth=2)
#     plt.gca().add_patch(circle)
#     plt.gca().set_aspect('equal', adjustable='box')

########################################################################################################################
"""encounter feedback"""
# # Print the number of datasets used for the plots
# print("\nNumber of Datasets Used per Group Type:")
# for group, count in dataset_counts.items():
#     print(f"{group}: {count} datasets")
#
# for group in group_types:
#     group_df = df.xs(group, level="group_type", drop_level=False)
#     valid_encounter_ids = group_df['encounter_id'].dropna().unique()
#     durations = encounter_durations.loc[encounter_durations.index.intersection(valid_encounter_ids)]
#
#     print(f"{group}: {len(durations)} valid encounters")
#
# for group in group_types:
#     print(f"\n=== {group} ===")
#
#     # Filter df for the specific group type
#     group_df = df.xs(group, level="group_type", drop_level=False)
#     valid_encounter_ids = group_df['encounter_id'].dropna().unique()
#
#     # Get durations for valid encounter IDs
#     group_durations = encounter_durations.loc[encounter_durations.index.intersection(valid_encounter_ids)]
#
#     # Get all raw encounter durations before filtering
#     all_durations = df.groupby("encounter_id").size()
#     all_group_durations = all_durations.loc[all_durations.index.intersection(valid_encounter_ids)]
#
#     # Count before and after filtering
#     print(f"Total encounters (before filtering): {len(all_group_durations)}")
#     print(f"Valid encounters (after filtering): {len(group_durations)}")
#
#     if len(all_group_durations) > 0:
#         print(f"Min Duration: {all_group_durations.min()} frames")
#         print(f"Max Duration: {all_group_durations.max()} frames")
#         print(f"Mean Duration: {all_group_durations.mean():.2f} frames")
#         print(f"Median Duration: {all_group_durations.median()} frames")
#
#         # Show some removed encounters (too short/long)
#         removed_encounters = all_group_durations[
#             (all_group_durations < encounter_duration_threshold[0]) |
#             (all_group_durations > encounter_duration_threshold[1])
#             ]
#         print(f"Removed Encounters (too short/long): {len(removed_encounters)}")
#
#         if not removed_encounters.empty:
#             print("Examples of removed encounters:")
#             print(removed_encounters.head(5))
#
#     else:
#         print("No encounters detected for this group.")