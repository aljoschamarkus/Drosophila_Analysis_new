from numpy.ma.core import append


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

        # Add a colorbar to the figure, and position it to the right
        cbar = fig.colorbar(cax, ax=axes, orientation='vertical', label='Density (log scale)', fraction=0.02, pad=0.04)

        # Add overall title
        fig.suptitle(f"Heatmaps of Trajectories for Condition: {cond} {geno}", fontsize=16)
        fig.canvas.manager.set_window_title(f"Heatmap_{cond}_{geno}_{num_bins}")
        plt.subplots_adjust(right=0.85, top=0.85)
        # plt.savefig(os.path.join(result_dir, f"{cond}_{geno}.png"))
        # plt.close()
        plt.show()

def plt_encounter_metrics(df, selected, encounter_duration_threshold):
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

    # --- Encounter Duration KDE ---
    plt.subplot(1, 2, 1)
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
            sns.kdeplot(durations, fill=True, label=f"{group}-{geno} (N={dataset_counts[group]})")
        else:
            sns.histplot(durations, bins=20, kde=False, label=f"{group}-{geno} (N={dataset_counts[group]})", alpha=0.3)

    plt.xlabel("Encounter Duration (frames)")
    plt.ylabel("Density")
    plt.title("Encounter Duration KDE (Normalized)")
    plt.legend()

    # --- Encounter Frequency KDE ---
    plt.subplot(1, 2, 2)
    for group, geno in selected:  # Now iterating over (group, genotype)
        freqs = encounter_frequency.xs((group, geno), level=["group_type", "genotype"], drop_level=False)

        if len(freqs) > 1:
            sns.kdeplot(freqs, fill=True, label=f"{group}-{geno} (N={dataset_counts[group]})")
        else:
            sns.histplot(freqs, bins=20, kde=False, label=f"{group}-{geno} (N={dataset_counts[group]})", alpha=0.3)

    plt.xlabel("Encounter Frequency (per minute)")
    plt.ylabel("Density")
    plt.title("Encounter Frequency KDE (Normalized)")
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_functions = {
    "heatmaps": plt_heatmaps_density,
    "encounter_metrics": plt_encounter_metrics,
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