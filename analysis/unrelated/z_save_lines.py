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

########################################################################################################################

"""speed filtering"""

# def remove_outliers(csv_data, speed_threshold, seed):
#     """Remove speed outliers beyond threshold * std deviation."""
#     # mean_speed = csv_data['speed_manual'].mean()
#     # std_speed = csv_data['speed_manual'].std()
#     # threshold = mean_speed + threshold_factor * std_speed
#     # csv_data.loc[csv_data[seed] < speed_threshold[0], seed] = np.nan
#     # csv_data.loc[csv_data[seed] > speed_threshold[1], seed] = np.nan
#     return csv_data

# def plot_speed(csv_data):
#     """Plot original and smoothed speed."""
#     import matplotlib.pyplot as plt
#     plt.figure(figsize=(10, 5))
#     plt.plot(csv_data['frame'], csv_data['speed_manual'], label='manual Speed', linewidth=2)
#     plt.plot(csv_data['frame'], csv_data['speed_processed'], label='Processed Speed', linewidth=2)
#     plt.xlabel('Frame')
#     plt.ylabel('Speed (cm/s)')
#     plt.legend()
#     plt.show()

# csv_data.loc[
#     csv_data['speed_processed'] > speed_initial_threshold, 'speed_processed'] = np.nan

########################################################################################################################

# # Convert 'frame' index level to a NumPy array
# frame_values = df.index.get_level_values('frame').to_numpy()
#
# # Create a mask to KEEP frames between 1800 and 3600
# mask = (frame_values >= 0) & (frame_values <= 7169)
#
# # Apply the mask
# df_filtered = df[mask]
# # df_WT = df_filtered.xs(('NANxCrimson', 'RGN'), level=['genotype', 'group_type'])
# # print(df_WT['encounter_count'].value_counts(0))
#
# # Verify unique frame values in the filtered DataFrame
# print(df_filtered.index.get_level_values('frame').unique())
# # Get unique group types

########################################################################################################################

def plt_nd_ot_cv(df, colors, selected, ND, rolling_window, len_end_intervall, num_trials=10000):
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    import numpy as np
    from scipy.spatial import distance_matrix
    from package.config_settings import circle_default
    from package.config_settings import group_size


    plt.figure(figsize=(10, 6))

    for i, (group, genotype) in enumerate(selected):

        df_pre = df.xs((group, genotype), level=['group_type', 'genotype'])
        df_plt = df_pre.groupby(['frame']).mean()
        df_clean = df_plt.dropna(subset=[ND])

        if rolling_window == 0:
            x_data = df_clean.index.get_level_values('frame').to_numpy() / 1800
            y_data_nd = df_clean[ND].to_numpy()
        elif rolling_window > 0:
            x_data = df_clean.index.get_level_values('frame').to_numpy() / 1800
            y_data_nd = df_clean[ND].rolling(window=91, center=True).mean().dropna().to_numpy()  # Remove NaNs

        # Ensure both x_data and y_data_nd have the same length
        min_length = min(len(x_data), len(y_data_nd))
        x_data = x_data[:min_length]
        y_data_nd = y_data_nd[:min_length]


        max_frame = df_pre.index.get_level_values('frame').max()
        df_last_1800 = df_pre.xs(slice(max_frame - len_end_intervall, max_frame), level='frame', drop_level=False)
        avg_nd_end = df_last_1800[ND].mean()

        def exp_growth(x, a, b, c):
            return a * (1 - np.exp(-b * x)) + c

        popt_exp, _ = curve_fit(exp_growth, x_data, y_data_nd)
        a_exp, b_exp, c_exp = popt_exp

        y_pred_exp = exp_growth(x_data, *popt_exp)
        ss_res_exp = np.sum((y_data_nd - y_pred_exp) ** 2)
        ss_tot_exp = np.sum((y_data_nd - np.mean(y_data_nd)) ** 2)
        r_squared_exp = 1 - (ss_res_exp / ss_tot_exp)

        x_0 = - (1 / b_exp) * np.log(1 + c_exp / a_exp)
        dt = abs(x_0) * 60

        # Extend x range for plotting
        x_extended = np.linspace(min(x_data) - abs(x_0), max(x_data), 500)

        plt.scatter(x_data, y_data_nd, label=f"{group}-{genotype} data", color=colors[i][1], s=10)
        plt.plot(x_data, exp_growth(x_data, *popt_exp),
                 label=f"fit (R²={r_squared_exp:.3f}), asymptote: {a_exp + c_exp:.3f}, slope parameter: {b_exp:.3f}",
                 color=colors[i][0])

        # plt.plot(x_extended, exp_growth(x_extended, *popt_exp),
        #                  label=f"{group}{genotype} (R²={r_squared_exp:.3f}), asymptote: {a_exp + c_exp:.3f}, slope parameter: {b_exp:.3f}",
        #                  color=colors[i][0])

        # plt.axvline(x=x_0, color=colors[i][0], linestyle='dashed')  # , label=f"x_0 = {x_0:.3f}")
        plt.axhline(avg_nd_end, color=colors[i][0], label=f' avg PND end: {avg_nd_end:.3f}', linestyle='dashed')

        # plt.text(x_pos, 0.5, f'$y = {a_exp + c_exp:.3f}(1 - e^{{-{b_exp:.3f}(x+{abs(x_0):.3f})}})$',
        #          transform=plt.gca().transAxes, fontsize=10, verticalalignment='center', color=colors[1])
        # plt.text(x_pos, 0.45, f'delayed start: {dt:.3f}',
        #          transform=plt.gca().transAxes, fontsize=10, verticalalignment='center', color=colors[1])

        print(f"{group}-{genotype}")
        print(f"R^2 {r_squared_exp}")
        print(f'y = {a_exp:.2f}(1 - e^(-{b_exp:.2f}x)) + {c_exp:.2f}')
        print(f"avg {ND} end", avg_nd_end)
        print(f" y=0 at x_0 = {x_0:.3f}")
        print(f"delayed start: {dt:.3f}")

    radius = circle_default[2]
    num_points = group_size

    # Function to generate random points inside a circle
    def generate_random_points_in_circle(radius, num_points):
        angles = np.random.uniform(0, 2 * np.pi, num_points)
        radii = np.sqrt(np.random.uniform(0, radius**2, num_points))  # Ensures uniform distribution
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        return np.column_stack((x, y))

    # Monte Carlo simulation
    avg_nd_mc = []

    if ND == 'NND':
        for _ in range(num_trials):
            points = generate_random_points_in_circle(radius, num_points)
            dist_matrix = distance_matrix(points, points)  # Compute pairwise distances
            np.fill_diagonal(dist_matrix, np.inf)  # Ignore self-distances
            avg_nnd_pre = np.mean(np.min(dist_matrix, axis=1))  # Average nearest-neighbor distance
            avg_nd_mc.append(avg_nnd_pre)
    elif ND == 'PND':
        for _ in range(num_trials):
            points = generate_random_points_in_circle(radius, num_points)
            dist_matrix = distance_matrix(points, points)  # Compute pairwise distances
            np.fill_diagonal(dist_matrix, np.nan)  # Ignore self-distances by setting them to NaN
            avg_pnd_pre = np.nanmean(dist_matrix)  # Average of all pairwise distances
            avg_nd_mc.append(avg_pnd_pre)

    avg_nd_mc = np.mean(avg_nd_mc)

    plt.axhline(avg_nd_mc, color='purple', label=f'statistical avg: {avg_nd_mc:.3f}', linestyle='dotted', linewidth=3)

    plt.legend()
    plt.xlabel("Time (min)")
    plt.ylabel("Pairwise Distance (cm)")
    plt.gcf().canvas.manager.set_window_title(f"{ND}_over_time_{selected[1][0]}-{selected[1][1]}")
    fig = plt.gcf()
    fig.patch.set_alpha(0)
    plt.show()

