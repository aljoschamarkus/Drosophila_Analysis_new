from scipy.stats import mannwhitneyu, kruskal
from scikit_posthocs import posthoc_dunn
import numpy as np
import pandas as pd
from package.config_settings import encounter_duration_threshold

def get_encounter_metrics(df, encounter_duration_threshold, metric):
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
    # print(len(encounter_durations))
    # print(len(encounter_frequency))
    if metric == 'encounter_duration':
        return encounter_durations
    elif metric == 'encounter_frequency':
        return encounter_frequency

def cliffs_delta(data1, data2):
    """Calculate Cliff's Delta effect size."""
    n1, n2 = len(data1), len(data2)
    wins = 0
    for x in data1:
        for y in data2:
            if x > y:
                wins += 1
            elif x < y:
                wins -= 1
    delta = wins / (n1 * n2)
    return delta

def compare_groups(data_list):
    """
    Compare groups using non-parametric tests.

    Parameters:
    - data_list: A list of lists, where each inner list contains the data for a group.
                The length of the outer list should be 2 or 3.
    """
    # Check the number of groups
    num_groups = len(data_list)
    if num_groups not in [2, 3]:
        raise ValueError("The input list must contain 2 or 3 groups.")

    # Remove NaN values (if any) from each group
    cleaned_data = [np.array(group)[~np.isnan(group)] for group in data_list]

    # Choose the appropriate test based on the number of groups
    if num_groups == 2:
        # Mann-Whitney U Test for two groups
        stat, p = mannwhitneyu(cleaned_data[0], cleaned_data[1])
        print(f"Mann-Whitney U Test: U = {stat}, p = {p:.4f}")
        if p < 0.05:
            print("Significant difference found between the two groups.")
        else:
            print("No significant difference found between the two groups.")
    else:
        # Kruskal-Wallis Test for three groups
        stat, p = kruskal(cleaned_data[0], cleaned_data[1], cleaned_data[2])
        print(f"Kruskal-Wallis Test: H = {stat}, p = {p:.4f}")
        if p < 0.05:
            print("Significant difference found among the groups.")
            # Perform post-hoc Dunn's test
            print("\nPerforming post-hoc Dunn's test with Bonferroni correction:")
            posthoc_results = posthoc_dunn(cleaned_data, p_adjust='bonferroni')
            print(posthoc_results)
        else:
            print("No significant difference found among the groups.")

    print("\n" + "="*50 + "\n")

def choose(df, selected, metric, encounter_duration_threshold):

    data = []

    if metric == 'speed_maunal' or metric == 'midline_offset_signless':
        group_parameter = 'condition'
    else:
        group_parameter = 'group_type'

    for group, geno in selected:
        df_data = df.xs((geno, group), level=['genotype', group_parameter])
        if metric == 'encounter_duration' or metric == 'encounter_frequency':
            df_data = get_encounter_metrics(df_data, encounter_duration_threshold, metric)
            df_final = df_data.replace([np.inf, -np.inf], np.nan).dropna().values
            print(len(df_final))
            data.append(df_final)
        else:
            df_final = df_data.replace([np.inf, -np.inf], np.nan).dropna().values
            print(len(df_final))
            data.append(df_final)

    return data

df = pd.read_pickle('/Users/aljoscha/Downloads/Data_Drosophila/results/data/df_group_parameters_NAN_IAV.pkl')
# df = pd.read_pickle('/Users/aljoscha/Downloads/Data_Drosophila/results/data/df_initial_NAN_IAV.pkl')

print("Index names:", df.index.names)
print("Columns:", df.columns.tolist())
selected_group = [
    ('RGN', 'WTxCrimson'),
    ('AIB', 'WTxCrimson'),
    # ('RGN', 'IAVxCrimson'),
]

metric = 'encounter_duration'

# def main():
#     inputs = df, selected_group, metric, encounter_duration_threshold
#     get_data = choose(*inputs)
#     compare_groups(get_data)

if __name__ == "__main__":
    inputs = df, selected_group, metric, encounter_duration_threshold
    get_data = choose(*inputs)
    # print(get_data)
    compare_groups(get_data)