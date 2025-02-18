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




def plt_heatmaps(df, data_len, selected): # , result_dir):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    frame_bin_size = (data_len + 1) / 2
    grid_size = 0.2

    for cond, geno in selected.items():
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

        plt.subplots_adjust(right=0.85, top=0.85)
        # plt.savefig(os.path.join(result_dir, f"{cond}_{geno}.png"))
        # plt.close()
        plt.show()



plot_functions = {
    "heatmaps": plt_heatmaps,
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