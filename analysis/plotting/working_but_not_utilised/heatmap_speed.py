import pandas as pd

df = pd.read_pickle('/Users/aljoscha/Downloads/results/data/df_initial.pkl')
selected = {
    'group': 'nompCxCrimson'
}

print("Index names:", df.index.names)
print("Columns:", df.columns.tolist())

def plt_heatmaps_density(df, selected): # , result_dir):
                    import numpy as np
                    import matplotlib.pyplot as plt
                    from matplotlib.colors import LogNorm, ListedColormap

                    frame_bin_size = len(df.index.get_level_values('frame').unique())
                    grid_size = 0.2

                    for cond, geno in selected.items():
                        x_plt = df.xs((cond, geno), level=['condition', 'genotype'])['x'].dropna().values
                        y_plt = df.xs((cond, geno), level=['condition', 'genotype'])['y'].dropna().values
                        speed_plt = df.xs((cond, geno), level=['condition', 'genotype'])['speed_manual'].dropna().values
                        subset = df.xs((cond, geno), level=['condition', 'genotype']).reset_index()
                        frame_plt = subset['frame'].values
                        min_frame = frame_plt.min()
                        max_frame = frame_plt.max()
                        frame_bins = np.arange(min_frame, max_frame + frame_bin_size, frame_bin_size)

                        num_bins = len(frame_bins) - 1
                        fig, axes = plt.subplots(1, num_bins, figsize=(15, 7), sharey=True)

                        if num_bins == 1:
                            axes = [axes]

                        for i in range(num_bins):
                            bin_start, bin_end = frame_bins[i], frame_bins[i + 1]
                            df_bin = subset[(frame_plt >= bin_start) & (frame_plt < bin_end)]

                            x_min, x_max = x_plt.min(), x_plt.max()
                            y_min, y_max = y_plt.min(), y_plt.max()

                            x_bin = df_bin['x'].values
                            y_bin = df_bin['y'].values
                            speed_bin = df_bin['speed'].values

                            # Filter out speeds above 30
                            valid_indices = speed_bin <= 20
                            x_bin = x_bin[valid_indices]
                            y_bin = y_bin[valid_indices]
                            speed_bin = speed_bin[valid_indices]

                            hist, xedges, yedges = np.histogram2d(
                                x_bin, y_bin,
                                bins=[
                                    np.arange(x_min, x_max + grid_size, grid_size),
                                    np.arange(y_min, y_max + grid_size, grid_size)
                                ]
                            )

                            speed_hist, _, _ = np.histogram2d(
                                x_bin, y_bin,
                                bins=[
                                    np.arange(x_min, x_max + grid_size, grid_size),
                                    np.arange(y_min, y_max + grid_size, grid_size)
                                ],
                                weights=speed_bin
                            )

                            avg_speed = np.divide(speed_hist, hist, out=np.zeros_like(speed_hist), where=hist != 0)

                            # Filter out bins with less than 2 data points and speeds above 10
                            avg_speed[(hist < 2) | (avg_speed > 0.25)] = np.nan

                            # Create a colormap with white for grids without data
                            cmap = plt.cm.viridis
                            cmap.set_under('white')

                            cax = axes[i].pcolormesh(
                                xedges, yedges, avg_speed.T,
                                cmap=cmap, shading='auto', vmin=0.0001
                            )

                            axes[i].set_title(f"Frames {int(bin_start)}-{int(bin_end)}")
                            axes[i].set_xlabel("X-coordinate (cm)")
                            if i == 0:
                                axes[i].set_ylabel("Y-coordinate (cm)")
                            axes[i].grid(True)
                            axes[i].set_aspect('equal')
                            axes[i].set_xlim([0, 14])
                            axes[i].set_ylim([0, 14])

                        cbar = fig.colorbar(cax, ax=axes, orientation='vertical', label='Average Speed', fraction=0.02, pad=0.04)
                        fig.suptitle(f"Heatmaps of Average Speed for Condition: {cond} {geno}", fontsize=16)
                        plt.subplots_adjust(right=0.85, top=0.85)
                        plt.show()


input = df, selected
plt_heatmaps_density(*input)  # Calls plot_sine(x_values)