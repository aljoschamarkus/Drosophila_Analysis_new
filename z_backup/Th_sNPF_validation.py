import os
import pandas as pd
import numpy as np
from package.util_data_preperation import handle_main_dir_p2
import matplotlib.pyplot as plt
import seaborn as sns
from package.util_significance import bootstrap_test

main_dir = "/Users/aljoscha/Downloads/Th_Chrimson"

fps = 30

data_frame = []

results_data_dir, results_plt_dir = handle_main_dir_p2()

for sub_dir in os.listdir(main_dir):
    if os.path.isdir(os.path.join(main_dir, sub_dir)):
        for csv in os.listdir(os.path.join(main_dir, sub_dir)):
            if "WT" in csv:
                genotype = "WT"
            elif "Th" in csv:
                genotype = "Th"
            else:
                print(f"Skipping: {csv} (Unknown condition)")
                continue
            csv_path = os.path.join(main_dir, sub_dir, csv)
            csv_data = pd.read_csv(csv_path, usecols=['frame',
                                                      'X#wcentroid (cm)',
                                                      'Y#wcentroid (cm)',
                                                      'SPEED#wcentroid (cm/s)',
                                                      'MIDLINE_OFFSET'])

            csv_data['speed_trex'] = csv_data['SPEED#wcentroid (cm/s)']
            csv_data['midline_offset_signless'] = np.abs(csv_data['MIDLINE_OFFSET'])
            csv_data['genotype'] = genotype

            data_frame.append(csv_data)

            def compute_speed(csv_data, fps=fps):
                """Calculate speed in cm/s from position data."""
                dx = csv_data['X#wcentroid (cm)'].diff()
                dy = csv_data['Y#wcentroid (cm)'].diff()
                csv_data['speed_manual'] = np.sqrt(dx ** 2 + dy ** 2) * fps
                return csv_data

            csv_data = compute_speed(csv_data, fps)
            # csv_data = smooth_speed(csv_data, speed_avg_window, method='rolling', input_speed='speed_processed')
            # csv_data = smooth_speed(csv_data, speed_avg_window, method='rolling', input_speed='speed_manual')
            csv_data['midline_offset_signless'] = np.abs(csv_data['MIDLINE_OFFSET'])

df_initial = pd.concat(data_frame, ignore_index=True)
df_initial.set_index(['genotype', 'frame'], inplace=True)
df_initial.rename(columns={
    'MIDLINE_OFFSET': 'midline_offset',
    'SPEED#wcentroid (cm/s)': 'speed',
    'X#wcentroid (cm)': 'x',
    'Y#wcentroid (cm)': 'y'
}, inplace=True)


# df_initial.to_pickle(os.path.join(results_data_dir, f"df_initial_Th.pkl"))
#
# df = pd.read_pickle('/Users/aljoscha/Downloads/results/data/df_initial_Th.pkl')

selected = ['Th', 'WT']
metric = 'speed_manual'

colors = [['salmon', 'red'], ['cornflowerblue', 'blue']]
bins = 2

inputs_Th = df_initial, selected, metric, bins, colors

data = []

def plot_metric(df_initial, selected, metric, bins, colors):
    for i, genotype in enumerate(selected):
        # Extract the data for the current genotype
        genotype_data = df_initial.xs(genotype, level='genotype')

        # Get the frame index values
        frames = genotype_data.index.get_level_values('frame')

        # Split the frames into 2 equally sized bins
        bin_labels = pd.qcut(frames, q=bins, labels=False)

        # Plot the KDE for each bin
        for ii, bin_label in enumerate(range(bins)):
            bin_data = genotype_data[bin_labels == bin_label][metric]
            if i == 0:
                light = "off"
            elif i == 1:
                light = "on"
            sns.kdeplot(bin_data, fill=True, alpha=0.05, label=f"{genotype} light {light}", color=colors[i][ii])
            bin_data = bin_data.replace([np.inf, -np.inf], np.nan).dropna()
            bin_data = bin_data.dropna().values
            print(bin_data)
            data.append(bin_data)

    fig = plt.gcf()
    ax = plt.gca()
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    plt.xlim(-0.1, 1)
    plt.legend(loc=1)
    plt.gcf().canvas.manager.set_window_title(f"{metric}_kde_{selected[1][0]}-{selected[1][1]}")
    plt.ylabel("Density")
    if metric == 'speed_manual':
        plt.xlabel(f"{metric} (cm/s)")
    elif metric == 'midline_offset_signless':
        plt.xlabel(f"{metric}")
    plt.show()

plot_metric(*inputs_Th)

p_value_th, cohen_d_th = bootstrap_test(data[0], data[1])
print(p_value_th, cohen_d_th)

p_value_wt, cohen_d_wt = bootstrap_test(data[2], data[3])
print(p_value_wt, cohen_d_wt)

csv_dir = "/Users/aljoscha/Downloads/sNPF_test/data/sNPFsingle_onoff_Video_fish0.csv"

csv_sNPF = pd.read_csv(csv_dir, usecols=['frame', 'MIDLINE_OFFSET', 'SPEED#wcentroid (cm/s)', 'X#wcentroid (cm)', 'Y#wcentroid (cm)'])
print("Columns:", csv_sNPF.columns.tolist())


def compute_speed(csv_data, fps=fps):
    """Calculate speed in cm/s from position data."""
    dx = csv_data['X#wcentroid (cm)'].diff()
    dy = csv_data['Y#wcentroid (cm)'].diff()
    csv_data['speed_manual'] = np.sqrt(dx ** 2 + dy ** 2) * fps
    return csv_data
csv_sNPF['midline_offset_signless'] = np.abs(csv_sNPF['MIDLINE_OFFSET'])

csv_sNPF = compute_speed(csv_sNPF, 30)

print(csv_sNPF['speed_manual'])


bin_on = (
    (csv_sNPF["frame"].between(0, 299)) |
    (csv_sNPF["frame"].between(2999, 3297)) |
    (csv_sNPF["frame"].between(5999, 6297))
)

# Split the data into two bins
bin_on_pre = bin_on.astype(bool)
bin_on = csv_sNPF[bin_on_pre]  # Contains specified frame ranges
bin_off = csv_sNPF[~bin_on_pre] # Contains the rest

# print(bin_on)
# print(bin_off)

colors_sNPF = ['red', 'salmon']
selected_sNPF = [bin_on, bin_off]
metric_sNPF = 'speed_manual'
inputs_sNPF = selected_sNPF, metric_sNPF, colors_sNPF

def plot_metric(selected, metric, colors):
    for i, bin in enumerate(selected):
        # Extract the data for the current genotype
        if i == 0:
            light = "on"
        elif i == 1:
            light = "off"
        sns.kdeplot(bin[metric], fill=True, alpha=0.05, label=f"sNPFxCrimson light {light}", color=colors[i])

    fig = plt.gcf()
    ax = plt.gca()
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    plt.xlim(-0.1, 1)
    plt.legend(loc=1)
    plt.gcf().canvas.manager.set_window_title(f"{metric}_kde_sNPF_on_off")
    plt.ylabel("Density")
    if metric == 'speed_manual':
        plt.xlabel(f"{metric} (cm/s)")
    elif metric == 'midline_offset_signless':
        plt.xlabel(f"{metric}")
    plt.show()

plot_metric(*inputs_sNPF)

bin_on1 = bin_on[metric_sNPF].replace([np.inf, -np.inf], np.nan).dropna()
data1 = bin_on1.dropna().values

bin_off1 = bin_off[metric_sNPF].replace([np.inf, -np.inf], np.nan).dropna()
data2 = bin_off1.dropna().values

p_value_sNPF1, cohen_d_sNPF1 = bootstrap_test(data1, data2)
print(p_value_sNPF1, cohen_d_sNPF1)
