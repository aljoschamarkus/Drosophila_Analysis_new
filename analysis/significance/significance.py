import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from package.config_settings import encounter_duration_threshold

df = pd.read_pickle('/Users/aljoscha/Downloads/Data_Drosophila/results/data/df_group_parameters_NAN_IAV.pkl')
print("Index names:", df.index.names)
print("Columns:", df.columns.tolist())
df_WT = df.xs(('WTxCrimson', 'RGN'), level=['genotype', 'group_type'])
df_IAV = df.xs(('IAVxCrimson', 'RGN'), level=['genotype', 'group_type'])



def get_encounter_metrics(df, encounter_duration_threshold):
    frames_per_minute = 1800

    # Ensure dataframe is sorted by index
    df = df.sort_index()

    # Identify encounter start and end
    df['encounter_start'] = (df['encounter_count'].diff() == 1)
    df['encounter_end'] = (df['encounter_count'].diff() == -1)

    # Assign an encounter ID to each encounter
    df['encounter_id'] = df['encounter_count'] * (df['encounter_count'].diff() != 0).cumsum()
    df.loc[df['encounter_count'] == 0, 'encounter_id'] = np.nan  # Remove non-encounters

    # Group by encounter_id and calculate duration
    encounter_durations = df.groupby('encounter_id').size()
    encounter_durations = encounter_durations[
        (encounter_durations >= encounter_duration_threshold[0]) &
        (encounter_durations <= encounter_duration_threshold[1])
        ]

    # Calculate encounter frequency per minute
    encounter_starts = df[df['encounter_start']].groupby(
        level=['sub_dir', 'condition', 'group_id', 'individual_id'])
    encounter_frequency = encounter_starts.apply(
        lambda x: x.index.get_level_values('frame').to_series().diff().fillna(frames_per_minute).lt(
            frames_per_minute).sum())

    return encounter_durations, encounter_frequency


WT_dur = get_encounter_metrics(df_WT, encounter_duration_threshold)[1]
IAV_dur = get_encounter_metrics(df_IAV, encounter_duration_threshold)[1]




def bootstrap_test(data1, data2, n_bootstrap=10000, one_sided=False):
    observed_diff = np.mean(data1) - np.mean(data2)
    pooled_data = np.concatenate([data1, data2])
    bootstrap_diff = np.array([
        np.mean(np.random.choice(pooled_data, size=len(data1), replace=True)) -
        np.mean(np.random.choice(pooled_data, size=len(data2), replace=True))
        for _ in range(n_bootstrap)
    ])
    # Compute p-value
    if one_sided:
        p_value = (np.sum(bootstrap_diff >= observed_diff) + 1) / (n_bootstrap + 1)
    else:
        p_value = (np.sum(np.abs(bootstrap_diff) >= np.abs(observed_diff)) + 1) / (n_bootstrap + 1)
    # Effect size: Cohen’s d
    n1, n2 = len(data1), len(data2)
    s1, s2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    cohen_d = observed_diff / pooled_std
    # # Plot bootstrap distribution
    # plt.hist(bootstrap_diff, bins=50, edgecolor='black', alpha=0.7)
    # plt.axvline(observed_diff, color='red', linestyle='dashed', label='Observed Diff')
    # plt.legend()
    # plt.xlabel('Bootstrapped Mean Differences')
    # plt.ylabel('Frequency')
    # plt.title('Bootstrap Distribution of Mean Differences')
    # plt.show()
    return p_value, cohen_d
# Run the test
p_value, effect_size = bootstrap_test(WT_dur, IAV_dur, n_bootstrap=10000)
print(f'P-value: {p_value}')
print(f'Cohen’s d (Effect Size): {effect_size}')

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, kruskal, friedmanchisquare
from scikit_posthocs import posthoc_dunn
import seaborn as sns
import matplotlib.pyplot as plt

# Example data
data = pd.DataFrame({
    'Group': ['A', 'A', 'B', 'B', 'C', 'C'],  # Group labels
    'Speed': [1.2, 1.5, 1.8, 1.6, 1.4, 1.3],  # Example measure
    'Midline_Offset': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    # Add other measures here
})

# KDE Plot for Speed
sns.kdeplot(data=data, x='Speed', hue='Group', fill=True)
plt.title('KDE Plot of Speed by Group')
plt.show()

# Mann-Whitney U Test for Speed (Group A vs. Group B)
group_A = data[data['Group'] == 'A']['Speed']
group_B = data[data['Group'] == 'B']['Speed']
stat, p = mannwhitneyu(group_A, group_B)
print(f"Mann-Whitney U for Speed: stat={stat}, p={p}")

# Kruskal-Wallis Test for Speed (Group A, B, C)
group_A = data[data['Group'] == 'A']['Speed']
group_B = data[data['Group'] == 'B']['Speed']
group_C = data[data['Group'] == 'C']['Speed']
stat, p = kruskal(group_A, group_B, group_C)
print(f"Kruskal-Wallis for Speed: stat={stat}, p={p}")

# Post-hoc Dunn's Test
posthoc = posthoc_dunn(data, val_col='Speed', group_col='Group', p_adjust='bonferroni')
print(posthoc)

# Friedman Test for PND Over Time (example with repeated measures)
pnd_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # Example data
stat, p = friedmanchisquare(*pnd_data)
print(f"Friedman Test for PND Over Time: stat={stat}, p={p}")