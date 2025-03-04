import pandas as pd
from package.util_plot import main_plot

df = pd.read_pickle('/Users/aljoscha/Downloads/results/data/df_group_parameters_p2.pkl')
# Get unique group types
selected = {
    ('RGN','WTxCrimson'),
    ('AIB', 'WTxCrimson'),
    # ('RGN', 'WTxCrimson'),
}

encounter_duration_threshold = [10, 1800]

inputs = df, selected, encounter_duration_threshold  # This creates a tuple (df, data_len, selected)
main_plot("encounter_metrics", *inputs)  # Calls plot_sine(x_values)