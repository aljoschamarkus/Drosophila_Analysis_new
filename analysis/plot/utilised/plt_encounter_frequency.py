import pandas as pd
from package.util_plot import main_plot

df = pd.read_pickle('/Users/aljoscha/Downloads/2012_nompC_Crimson_WT_4min_50p_TRex_beta_2101/results/data/df_group_parameters.pkl')
# Get unique group types
group_types = ["RGN", "AIB"]
selected = {
    'RGN': 'nompCxCrimson',
    'AIB': 'nompCxCrimson'
}

encounter_duration_threshold = [10, 1800]

inputs = df, selected, group_types, encounter_duration_threshold  # This creates a tuple (df, data_len, selected)
main_plot("encounter_metrics", *inputs)  # Calls plot_sine(x_values)