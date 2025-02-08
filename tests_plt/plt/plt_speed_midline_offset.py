import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from package import config_settings

condition = config_settings.condition
genotype = config_settings.genotype

df = pd.read_pickle('/Users/aljoscha/Downloads/2012_nompC_Crimson_WT_4min_50p_TRex_beta_2101/results/data_frame_final.pkl')

colors = [['#e41a1c', '#377eb8', '#4daf4a'], ['black', 'gray', 'darkgray']]

# Initialize an empty list to store legend entries
legend_entries = []
counter = 0
color_list = [
    (255, 0, 0),    # Red
    (0, 114, 178),  # Blue
    (0, 158, 115),  # Green
    (230, 159, 0),  # Orange
    (204, 121, 167),# Purple
    (86, 180, 233)  # Cyan
]
normalized_color_list = [(r / 255.0, g / 255.0, b / 255.0) for r, g, b in color_list]

for cond_idx, cond in enumerate(condition):
    for geno_idx, geno in enumerate(genotype):
        df.dropna(inplace=True)
        speed_plt = df.xs((cond, geno, 'actual'), level=['condition', 'genotype', 'group_type'])['speed'].values
        # speed_plt = speed_plt[speed_plt != 0]
        # color_speed = colors[cond_idx][geno_idx]  # Use colors[0] for speed_plt
        # sns.kdeplot(speed_plt, color=color_speed, fill=False, alpha=1)
        plt.hist(
            speed_plt,
            bins=4000,
            color=normalized_color_list[counter],
            alpha=1 - counter * 0.12,
            label=f"{cond} - {geno}",
            density=True
        )
        counter += 1

# Set the legend after plotting all histograms
plt.legend(loc="upper right")
plt.title("Speed Distributions by Condition and Genotype")
plt.xlim(-1, 18)
plt.xlabel("Value")
plt.ylabel("Density")
plt.show()

legend_entries = []
counter = 0
window_size = 30  # Set the size of the rolling window

alpha = [1, 0.3, 0.3, 0.7, 0.4, 0.4]

for cond_idx, cond in enumerate(condition):
    for geno_idx, geno in enumerate(genotype):
        df.dropna(inplace=True)
        speed_plt = df.xs((cond, geno, 'actual'), level=['condition', 'genotype', 'group_type'])['speed'].values
        speed_plt = speed_plt[speed_plt != 0]

        # Apply rolling average
        speed_plt_rolling = pd.Series(speed_plt).rolling(window=window_size, center=False).mean()
        speed_plt_rolling = speed_plt_rolling.dropna().values  # Drop NaN values introduced by rolling

        # Plot the smoothed data
        plt.hist(
            speed_plt_rolling,
            bins=2000,
            color=normalized_color_list[counter],
            # alpha=1 - counter * 0.15,
            alpha=alpha[counter],
            label=f"{cond} - {geno}",
            density=True
        )
        counter += 1

# Set the legend after plotting all histograms
plt.legend(loc="upper right")
plt.title("Speed Distributions by Condition and Genotype")
plt.xlim(0, 8)
plt.xlabel("Value")
plt.ylabel("Density")
plt.show()

counter = 0
for cond_idx, cond in enumerate(condition):
    for geno_idx, geno in enumerate(genotype):
        df.dropna(inplace=True)
        midline_offset_plt = df.xs((cond, geno, 'actual'), level=['condition', 'genotype', 'group_type'])['midline_offset'].values
        # color_plt = colors[cond_idx][geno_idx]
        sns.kdeplot(midline_offset_plt ** 2, color=normalized_color_list[counter], fill=True, alpha=0.08, label=f"{cond}_{geno}")

        # plt.hist(
        #     midline_offset_plt,
        #     bins=500,
        #     color=normalized_color_list[counter],
        #     alpha=alpha[counter],
        #     label=f"{cond}_{geno}",
        #     density=True,
        #     fill=True# Normalize the histogram
        # )
        counter += 1

# Set the legend after plotting all histograms
plt.legend(loc="upper right")
plt.title("Midline Offset Distributions by Condition and Genotype")
plt.xlim(-1, 18)
plt.xlabel("Value")
plt.ylabel("Density")
plt.xlim(-0.1,0.75)
plt.show()

counter = 0
for cond_idx, cond in enumerate(condition):
    for geno_idx, geno in enumerate(genotype):
        df.dropna(inplace=True)
        midline_offset_plt = df.xs((cond, geno, 'actual'), level=['condition', 'genotype', 'group_type'])['midline_offset'].values
        # color_plt = colors[cond_idx][geno_idx]
        speed_plt = df.xs((cond, geno, 'actual'), level=['condition', 'genotype', 'group_type'])['speed'].values
        # speed_plt = speed_plt[speed_plt != 0]
        # speed_plt_rolling = pd.Series(speed_plt).rolling(window=window_size, center=False).mean()
        # speed_plt_rolling = speed_plt_rolling.dropna().values  # Drop NaN values introduced by rolling
        plt.scatter(midline_offset_plt, speed_plt, color=normalized_color_list[counter], alpha=0.8, label=f"{cond}_{geno}")

        # plt.hist(
        #     midline_offset_plt,
        #     bins=500,
        #     color=normalized_color_list[counter],
        #     alpha=alpha[counter],
        #     label=f"{cond}_{geno}",
        #     density=True,
        #     fill=True# Normalize the histogram
        # )
        counter += 1

# Set the legend after plotting all histograms
plt.legend(loc="upper right")
plt.title("Midline Offset Distributions by Condition and Genotype")
plt.xlim(-1, 18)
plt.xlabel("Value")
plt.ylabel("Density")
plt.xlim(-1,1)
plt.show()

