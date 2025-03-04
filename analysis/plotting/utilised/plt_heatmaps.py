import pandas as pd
from package.util_plot import main_plot
selected = [
    ('group', 'WTxCrimson'),
    ('group', 'NANxCrimson'),
    ('group', 'IAVxCrimson'),
]

df = pd.read_pickle('/Users/aljoscha/Downloads/results/data/df_initial_p2.pkl')
inputs = df, selected, 1  # This creates a tuple (df, data_len, selected)

main_plot("heatmaps", *inputs)  # Calls plot_sine(x_values)