import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from package.util_data_preperation import handle_main_dir

main_dir = "/Users/aljoscha/Downloads/2012_nompC_Crimson_WT_4min_50p_TRex_beta_2101"
condition = ["group", "single"]
genotype = ["nompCxCrimson", "WTxCrimson", "nompCxWT"]
quality = [1296, 972]
dish_radius = 6.5

FPS = 30
group_size = 5
data_len = 7191
stimulation_used = "625nm 1µW/mm^2"
colors = [['#e41a1c', '#377eb8', '#4daf4a'], ['#fbb4ae', '#a6cee3', '#b2df8a']]
line_styles = ["-", "--"]

condition_dir = handle_main_dir(main_dir, condition)
print(condition_dir)

df = pd.read_pickle('/Users/aljoscha/Downloads/2012_nompC_Crimson_WT_4min_50p_TRex_beta_2101/results/data/data_frame_initial.pkl')

print("Index names:", df.index.names)
print("Columns:", df.columns.tolist())

frame_bin_size = (data_len + 1) / 2
grid_size = 0.2

for cond in condition:
    for geno in genotype:
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

        # Adjust layout to prevent overlap while keeping the colorbar in place
        plt.subplots_adjust(right=0.85, top=0.85)

        # Show the plot
        plt.show()