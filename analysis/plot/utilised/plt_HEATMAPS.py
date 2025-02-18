import pandas as pd
from package.util_plot import main_plot
from package.util_plot import plt_heatmaps

data_len = 7191
df = pd.read_pickle('/Users/aljoscha/Downloads/2012_nompC_Crimson_WT_4min_50p_TRex_beta_2101/results/data/data_frame_initial.pkl')
selected = {
    'group': 'nompCxCrimson',
    'single': 'nompCxCrimson'
}
inputs = df, data_len, selected  # This creates a tuple (df, data_len, selected)

main_plot("heatmaps", *inputs)  # Calls plot_sine(x_values)