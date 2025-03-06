import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from package.util_data_preperation import handle_main_dir_p2
from package.util_significance import bootstrap_test

# Constants
MAIN_DIR = "/Users/aljoscha/Downloads/Data_Drosophila/Th"
FPS = 30
RESULTS_DATA_DIR, RESULTS_PLT_DIR = handle_main_dir_p2()

def load_and_preprocess_data(main_dir):
    """Load and preprocess data from CSV files in the main directory."""
    data_frame = []

    for sub_dir in os.listdir(main_dir):
        sub_dir_path = os.path.join(main_dir, sub_dir)
        if os.path.isdir(sub_dir_path):
            for csv_file in os.listdir(sub_dir_path):
                genotype = "WT" if "WT" in csv_file else "Th" if "Th" in csv_file else None
                if genotype is None:
                    print(f"Skipping: {csv_file} (Unknown condition)")
                    continue

                csv_path = os.path.join(sub_dir_path, csv_file)
                csv_data = pd.read_csv(csv_path, usecols=['frame', 'X#wcentroid (cm)', 'Y#wcentroid (cm)',
                                                           'SPEED#wcentroid (cm/s)', 'MIDLINE_OFFSET'])

                # Rename columns immediately after loading
                csv_data.rename(columns={
                    'X#wcentroid (cm)': 'x',
                    'Y#wcentroid (cm)': 'y',
                    'SPEED#wcentroid (cm/s)': 'speed',
                    'MIDLINE_OFFSET': 'midline_offset'
                }, inplace=True)

                csv_data['speed_trex'] = csv_data['speed']
                csv_data['midline_offset_signless'] = np.abs(csv_data['midline_offset'])
                csv_data['genotype'] = genotype
                csv_data = compute_speed(csv_data, FPS)

                # Replace 0 values in speed_manual with NaN
                # csv_data['speed_manual'] = csv_data['speed_manual'].replace(0, np.nan)

                data_frame.append(csv_data)

    df_initial = pd.concat(data_frame, ignore_index=True)
    df_initial.set_index(['genotype', 'frame'], inplace=True)
    return df_initial

def compute_speed(csv_data, fps):
    """Calculate speed in cm/s from position data."""
    dx = csv_data['x'].diff()
    dy = csv_data['y'].diff()
    csv_data['speed_manual'] = np.sqrt(dx ** 2 + dy ** 2) * fps
    return csv_data

def plot_metric(df, selected, metric, bins, colors):
    """Plot KDE for the given metric and genotypes."""
    data = []
    plt.figure()

    for i, genotype in enumerate(selected):
        genotype_data = df.xs(genotype, level='genotype')
        frames = genotype_data.index.get_level_values('frame')
        bin_labels = pd.qcut(frames, q=bins, labels=False)

        for ii in range(bins):
            bin_data = genotype_data[bin_labels == ii][metric]
            light = "off" if ii == 0 else "on"

            sns.kdeplot(bin_data, fill=True, alpha=0.05, label=f"{genotype} light {light}", color=colors[i][ii], clip=(0, 1))
            bin_data = bin_data.replace([np.inf, -np.inf], np.nan).dropna().values
            data.append(bin_data)

    if metric == 'speed_manual':
        plt.xlim(-0.1, 4)
    elif metric == 'midline_offset':
        plt.xlim(0, 1)
    plt.legend(loc=1)
    plt.ylabel("Density")
    plt.xlabel(f"{metric} (cm/s)" if metric == 'speed_manual' else metric)
    plt.show()

    return data

def analyze_sNPF_data(csv_path, fps):
    """Load and analyze sNPF data."""
    csv_sNPF = pd.read_csv(csv_path, usecols=['frame', 'MIDLINE_OFFSET', 'SPEED#wcentroid (cm/s)',
                                              'X#wcentroid (cm)', 'Y#wcentroid (cm)'])
    csv_sNPF.rename(columns={
        'X#wcentroid (cm)': 'x',
        'Y#wcentroid (cm)': 'y',
        'SPEED#wcentroid (cm/s)': 'speed',
        'MIDLINE_OFFSET': 'midline_offset'
    }, inplace=True)

    csv_sNPF['midline_offset_signless'] = np.abs(csv_sNPF['midline_offset'])
    csv_sNPF = compute_speed(csv_sNPF, fps)

    # Replace 0 values in speed_manual with NaN
    # csv_sNPF['speed_manual'] = csv_sNPF['speed_manual'].replace(0, np.nan)

    bin_on = (
        (csv_sNPF["frame"].between(0, 299)) |
        (csv_sNPF["frame"].between(2999, 3297)) |
        (csv_sNPF["frame"].between(5999, 6297)))
    bin_on_pre = bin_on.astype(bool)
    bin_on = csv_sNPF[bin_on_pre]
    bin_off = csv_sNPF[~bin_on_pre]

    return bin_on, bin_off

def main():
    df_initial = load_and_preprocess_data(MAIN_DIR)

    selected = ['Th', 'WT']
    metric = 'midline_offset'
    colors = [['salmon', 'red'], ['cornflowerblue', 'blue']]
    bins = 2

    data = plot_metric(df_initial, selected, metric, bins, colors)

    p_value_th_on_off, cohen_d_th_on_off = bootstrap_test(data[0], data[1])
    p_value_wt_on_off, cohen_d_wt_on_off = bootstrap_test(data[2], data[3])
    p_value_th_on_wt_on, cohen_d_th_on_wt_on = bootstrap_test(data[1], data[3])
    p_value_th_off_wt_off, cohen_d_th_off_wt_off = bootstrap_test(data[0], data[2])

    print(f"th_on_off: p-value={p_value_th_on_off}, Cohen's d={cohen_d_th_on_off}")
    print(f"wt_on_off: p-value={p_value_wt_on_off}, Cohen's d={cohen_d_wt_on_off}")
    print(f"th_on_wt_on: p-value={p_value_th_on_wt_on}, Cohen's d={cohen_d_th_on_wt_on}")
    print(f"th_off_wt_off: p-value={p_value_th_off_wt_off}, Cohen's d={cohen_d_th_off_wt_off}")


    csv_sNPF_path = "/Users/aljoscha/Downloads/Data_Drosophila/sNPF/data/sNPFsingle_onoff_Video_fish0.csv"
    bin_on, bin_off = analyze_sNPF_data(csv_sNPF_path, FPS)

    colors_sNPF = ['red', 'salmon']
    selected_sNPF = [bin_on, bin_off]
    metric_sNPF = 'midline_offset'

    plt.figure()
    for i, bin_data in enumerate(selected_sNPF):
        light = "on" if i == 0 else "off"
        sns.kdeplot(bin_data[metric_sNPF], fill=True, alpha=0.05, label=f"sNPFxCrimson light {light}", color=colors_sNPF[i], clip=(0, 1))

    if metric_sNPF == 'speed_manual':
        plt.xlim(-0.1, 100)
    elif metric_sNPF == 'midline_offset':
        plt.xlim(0, 1)
    plt.legend(loc=1)
    plt.ylabel("Density")
    plt.xlabel(f"{metric_sNPF}")
    plt.show()

    bin_on_data = bin_on[metric_sNPF].replace([np.inf, -np.inf], np.nan).dropna().values
    bin_off_data = bin_off[metric_sNPF].replace([np.inf, -np.inf], np.nan).dropna().values

    p_value_sNPF, cohen_d_sNPF = bootstrap_test(bin_on_data, bin_off_data)
    print(f"sNPF: p-value={p_value_sNPF}, Cohen's d={cohen_d_sNPF}")

if __name__ == "__main__":
    main()