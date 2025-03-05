def main_plot(select_function, *inputs):
    """Dynamically select and call a plot function."""
    if select_function in plot_functions:
        func = plot_functions[select_function]

        # Check function arguments
        required_args = func.__code__.co_varnames[:func.__code__.co_argcount]

        if len(inputs) >= len(required_args):  # Ensure correct number of inputs
            func(*inputs)  # Call the function dynamically
        else:
            print(f"Error: {select_function} requires {len(required_args)} inputs.")
    else:
        print(f"Error: Function '{select_function}' not found.")


def plt_heatmaps_density(df, selected, num_bins): # , result_dir):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    data_len = len(df.index.get_level_values('frame').unique())
    frame_bin_size = data_len / num_bins
    grid_size = 0.2

    for cond, geno in selected:
        x_plt = df.xs((cond, geno),
                      level=['condition', 'genotype'])['x'].dropna().values
        y_plt = df.xs((cond, geno),
                      level=['condition', 'genotype'])['y'].dropna().values
        subset = df.xs((cond, geno), level=['condition', 'genotype']).reset_index()
        frame_plt = subset['frame'].values
        min_frame = frame_plt.min()
        max_frame = frame_plt.max()
        frame_bins = np.arange(min_frame, max_frame + frame_bin_size, frame_bin_size)

        # Create a subplot for each frame analysis_unrelated
        num_bins = len(frame_bins) - 1
        fig, axes = plt.subplots(1, num_bins, figsize=(15, 7), sharey=True)

        if num_bins == 1:  # Handle case where only one frame analysis_unrelated exists
            axes = [axes]

        # Loop through each frame analysis_unrelated
        for i in range(num_bins):
            bin_start, bin_end = frame_bins[i], frame_bins[i + 1]

            # Filter data for the current frame analysis_unrelated
            df_bin = subset[(frame_plt >= bin_start) & (frame_plt < bin_end)]

            # Define grid for 1x1 cm bins (you can adjust based on the scale of your data)
            x_min, x_max = x_plt.min(), x_plt.max()
            y_min, y_max = y_plt.min(), y_plt.max()

            # Create a 2D histogram of point density (counts in each 1x1 cm analysis_unrelated)
            x_bin = df_bin['x'].values
            y_bin = df_bin['y'].values
            hist, xedges, yedges = np.histogram2d(
                x_bin, y_bin,
                bins=[
                    np.arange(x_min, x_max + grid_size, grid_size),
                    np.arange(y_min, y_max + grid_size, grid_size)
                ]
            )

            # Plot heatmap
            cax = axes[i].pcolormesh(
                xedges, yedges, hist.T,
                cmap='viridis', shading='auto',
                norm=LogNorm(vmin=1)  # Log scale normalization for better density visualization
            )

            # Set the title, labels, and grid
            axes[i].set_title(f"Frames {int(bin_start)}-{int(bin_end)}")
            axes[i].set_xlabel("X-coordinate (cm)")
            if i == 0:  # Only label the y-axis on the first plot
                axes[i].set_ylabel("Y-coordinate (cm)")
            axes[i].grid(True)

            # Ensure the aspect ratio is equal to prevent distortion
            axes[i].set_aspect('equal')

            # Set limits for x and y to fix them at 30cm x 30cm
            axes[i].set_xlim([0, 14])  # Set x limit to 30cm
            axes[i].set_ylim([0, 14])  # Set y limit to 30cm

            # ax = plt.gca()
            # ax.patch.set_alpha(0)

            # Add a colorbar to the figure, and position it to the right

            # # Add overall title
            # fig.suptitle(f"Heatmaps of Trajectories for Condition: {cond} {geno}", fontsize=16)
            plt.gcf().canvas.manager.set_window_title(f"Heatmaps_{cond}-{geno}")
        bar = fig.colorbar(cax, ax=axes, orientation='vertical', label='Density (log scale)', fraction=0.02, pad=0.04)
        plt.subplots_adjust(right=0.85, top=0.85)
        fig = plt.gcf()
        fig.patch.set_alpha(0)
        plt.show()

def plt_encounter_metrics(df, selected, encounter_duration_threshold, metric):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    frames_per_minute = 1800

    dataset_length = len(df.index.get_level_values('frame').unique())

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


    # Prepare a figure with subplots
    plt.figure(figsize=(12, 5))

    # Dictionary to store dataset counts (N)
    dataset_counts = {}


    if metric == "duration":
        # --- Encounter Duration KDE ---
        for group, geno in selected:
            group_df = df.xs((group, geno), level=["group_type", "genotype"], drop_level=False)
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
                sns.kdeplot(durations, fill=False, label=f"{group}-{geno} (N={dataset_counts[group]})")
            else:
                break

        plt.xlabel("Encounter Duration (frames)")
        plt.ylabel("Density")
        plt.legend()

    elif metric == "frequency":
        # --- Encounter Frequency KDE ---
        for group, geno in selected:  # Now iterating over (group, genotype)
            freqs = encounter_frequency.xs((group, geno), level=["group_type", "genotype"], drop_level=False)

            if len(freqs) > 1:
                sns.kdeplot(freqs, fill=False, label=f"{group}-{geno}")
            else:
                break

        plt.xlabel("Encounter Frequency (per minute)")
        plt.ylabel("Density")
        plt.legend(loc=1)
    plt.gcf().canvas.manager.set_window_title(f"encounter_{metric}_{selected[1][0]}-{selected[1][1]}_1")
    fig = plt.gcf()
    ax = plt.gca()
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    plt.show()


def plt_nd_kde(df, selected, ND):
    import seaborn as sns
    import matplotlib.pyplot as plt
    for group, geno in selected:
        df_plt = df.xs((geno, group), level=['genotype', 'group_type'])
        sns.kdeplot(df_plt[ND], fill=True, alpha=0.05, label=f'{group} {geno}')
    plt.gcf().canvas.manager.set_window_title(f"{ND}_kde_{selected[1][0]}-{selected[1][1]}")
    fig = plt.gcf()
    ax = plt.gca()
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    plt.legend(loc=1)
    plt.show()


def plt_nd_kde_fb(df, selected, ND, num_bins=4):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    # Extract the frame values from the index
    frames = df.index.get_level_values('frame')

    # Calculate the bin edges dynamically
    frame_min = frames.min()
    frame_max = frames.max()
    bin_edges = np.linspace(frame_min, frame_max, num=num_bins + 1)  # 5 edges for 4 bins
    bin_labels = [f'{int(bin_edges[i])}-{int(bin_edges[i + 1])}' for i in range(len(bin_edges) - 1)]

    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(1, num_bins, figsize=(num_bins * 4, num_bins))
    axes = axes.flatten()  # Flatten the 2x2 array to easily iterate over it

    for i, (bin_start, bin_end) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        ax = axes[i]
        for group, geno in selected:
            # Filter the data for the current frame bin and selected group/genotype
            df_bin = df[(frames >= bin_start) & (frames < bin_end)]
            df_bin = df_bin.xs((geno, group), level=['genotype', 'group_type'])
            sns.kdeplot(df_bin[ND], fill=True, alpha=0.05, label=f'{group} {geno}', ax=ax)
        ax.set_title(f'Frames {bin_labels[i]}')
        ax.legend(loc=1)

    plt.tight_layout()
    plt.gcf().canvas.manager.set_window_title(f"{ND}_kde_{selected[1][0]}{selected[1][1]}_{num_bins}")
    fig = plt.gcf()
    fig.patch.set_alpha(0)
    plt.show()

def plt_encounter_metrics_fb(df, selected, encounter_duration_threshold, metric, num_bins=4):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    frames_per_minute = 1800

    # Extract the frame values from the index
    frames = df.index.get_level_values('frame')

    # Calculate the bin edges dynamically
    frame_min = frames.min()
    frame_max = frames.max()
    bin_edges = np.linspace(frame_min, frame_max, num=num_bins + 1)  # 5 edges for 4 bins
    bin_labels = [f'{int(bin_edges[i])}-{int(bin_edges[i+1])}' for i in range(len(bin_edges)-1)]

    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(1, num_bins, figsize=(num_bins * 4, num_bins))
    axes = axes.flatten()  # Flatten the 2x2 array to easily iterate over it

    # Dictionary to store dataset counts (N)
    dataset_counts = {}

    for i, (bin_start, bin_end) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        ax = axes[i]
        # Filter the data for the current frame bin
        df_bin = df[(frames >= bin_start) & (frames < bin_end)].copy()  # Use .copy() to avoid SettingWithCopyWarning

        # Identify encounter start and end for the current bin
        df_bin.loc[:, 'encounter_start'] = (df_bin['encounter_count'].diff() == 1)
        df_bin.loc[:, 'encounter_end'] = (df_bin['encounter_count'].diff() == -1)

        # Assign an encounter ID to each encounter (continuous blocks of 1s)
        df_bin.loc[:, 'encounter_id'] = df_bin['encounter_count'] * (df_bin['encounter_count'].diff() != 0).cumsum()
        df_bin.loc[df_bin['encounter_count'] == 0, 'encounter_id'] = np.nan  # Remove non-encounters

        # Group by encounter_id and calculate duration for the current bin
        encounter_durations = df_bin.groupby('encounter_id').size()
        encounter_durations = encounter_durations[
            (encounter_durations >= encounter_duration_threshold[0]) &
            (encounter_durations <= encounter_duration_threshold[1])
        ]

        # Calculate encounter frequency per minute for the current bin
        encounter_starts = df_bin[df_bin['encounter_start']].groupby(level=['sub_dir', 'condition', 'genotype', 'group_type', 'group_id', 'individual_id'])
        encounter_frequency = encounter_starts.apply(lambda x: x.index.get_level_values('frame').to_series().diff().fillna(frames_per_minute).lt(frames_per_minute).sum())

        if metric == "duration":
            # --- Encounter Duration KDE ---
            for group, geno in selected:
                try:
                    group_df = df_bin.xs((group, geno), level=["group_type", "genotype"], drop_level=False)
                    # Determine number of datasets used
                    if group == "RGN":
                        dataset_counts[group] = group_df.index.get_level_values("individual_id").nunique()
                    else:  # AIB & AGB -> Estimate dataset count using frame count
                        total_frames = len(group_df)  # Total number of frames in bootstrapped dataset
                        dataset_counts[group] = int(np.round(total_frames / (frame_max - frame_min + 1)))  # Estimate dataset count


                    # Get valid encounter durations
                    valid_encounter_ids = group_df['encounter_id'].dropna().unique()
                    durations = encounter_durations.loc[encounter_durations.index.intersection(valid_encounter_ids)]

                    if len(durations) > 1:
                        sns.kdeplot(durations, fill=False, label=f"{group}-{geno} (N={dataset_counts[group]})", ax=ax)
                        # ax = plt.gca()
                        # ax.patch.set_alpha(0)
                except KeyError:
                    # Skip if the group/genotype combination doesn't exist in this bin
                    continue

            ax.set_xlabel("Encounter Duration (frames)")
            ax.set_title(f'Frames {bin_labels[i]}')

        elif metric == "frequency":
            # --- Encounter Frequency KDE ---
            for group, geno in selected:
                try:
                    freqs = encounter_frequency.xs((group, geno), level=["group_type", "genotype"], drop_level=False)

                    if len(freqs) > 1:
                        sns.kdeplot(freqs, fill=False, label=f"{group}-{geno}", ax=ax)
                except KeyError:
                    # Skip if the group/genotype combination doesn't exist in this bin
                    continue

            ax.set_xlabel("Encounter Frequency (per minute)")
            ax.set_title(f'Frames {bin_labels[i]}')
            # ax = plt.gca()
            # ax.patch.set_alpha(0)

        ax.set_ylabel("Density")
        ax.legend(loc=1)

    plt.tight_layout()
    plt.gcf().canvas.manager.set_window_title(f"encounter_{metric}_{selected[1][0]}-{selected[1][1]}_{num_bins}")
    fig = plt.gcf()
    fig.patch.set_alpha(0)
    plt.show()

def plt_nd_ot_cv(df, colors, selected, ND, rolling_window, len_end_intervall, bin_len, num_trials=10000):
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

        # Group data into bins of 10 frames
        df_clean['frame_bin'] = (df_clean.index.get_level_values('frame') // bin_len) * bin_len
        df_binned = df_clean.groupby('frame_bin').mean().reset_index()

        if rolling_window == 0:
            x_data = df_binned['frame_bin'].to_numpy() / 1800
            y_data_nd = df_binned[ND].to_numpy()
        elif rolling_window > 0:
            x_data = df_binned['frame_bin'].to_numpy() / 1800
            y_data_nd = df_binned[ND].rolling(window=91, center=True).mean().dropna().to_numpy()  # Remove NaNs

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

        plt.axhline(avg_nd_end, color=colors[i][0], label=f' avg PND end: {avg_nd_end:.3f}', linestyle='dashed')

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

def plt_speed_mo(df, selected, metric):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    for group, genotype in selected:
        speed_plt = df.xs((group, genotype), level=['condition', 'genotype'])[metric].values
        sns.kdeplot(speed_plt, fill=True, alpha=0.05, label=f"{group} - {genotype}")
        # sns.histplot(speed_plt, kde=True, fill=True, alpha=0.5, label=f"{group} - {genotype}")

    fig = plt.gcf()
    ax = plt.gca()
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    if metric == 'speed_manual':
        plt.xlim(-0.1, 1)
    elif metric == 'midline_offset_signless':
        plt.xlim(-0.1, 1)
    elif metric == 'midline_offset':
        plt.xlim(-1, 1)
    plt.legend(loc=1)
    plt.gcf().canvas.manager.set_window_title(f"{metric}_kde_{selected[1][0]}-{selected[1][1]}")
    plt.ylabel("Density")
    if metric == 'speed_manual':
        plt.xlabel(f"{metric} (cm/s)")
    elif metric == 'midline_offset_signless':
        plt.xlabel(f"{metric}")
    plt.show()


plot_functions = {
    "heatmaps": plt_heatmaps_density,
    "encounter_metrics": plt_encounter_metrics,
    "nd_kde": plt_nd_kde,
    "nd_kde_fb": plt_nd_kde_fb,
    "encounter_metrics_fb": plt_encounter_metrics_fb,
    "nd_ot_cv": plt_nd_ot_cv,
    "speed_mo": plt_speed_mo
}

# def plot(x, y, midpoint, radius):
#     plt.figure(figsize=(8, 6))
#     plt.scatter(x, y, alpha=0.5, label="Data")
#     plt.xlabel('Nearest Neighbor Distance (NND)')
#     plt.ylabel('Rolling-Averaged Speed')
#     plt.title('Speed vs. Nearest Neighbor Distance')
#     plt.xlim(0, 1)
#     plt.ylim(0, 1)
#     plt.gca().set_aspect('equal', adjustable ='box')
#     plt.tight_layout()
#     plt.legend()
#     plt.grid(True)
#     plt.show()
#     circle = plt.Circle(midpoint, radius, color='red', fill=False, linewidth=2)
#     plt.gca().add_patch(circle)
#     # plt.savefig(os.path.join(condition_dir[2], f"{cond}_{geno}.png"))
#     # plt.close()
#     # cbar = fig.colorbar(cax, ax=axes, orientation='vertical', label='Density (log scale)', fraction=0.02, pad=0.04)
#     return None