import pandas as pd
from package.util_plot import main_plot

# df_initial = pd.read_pickle('/Users/aljoscha/Downloads/results/data/df_initial_nompC.pkl')
df_initial = pd.read_pickle('/Users/aljoscha/Downloads/results/data/df_initial_NAN_IAV.pkl')
# df_groups = pd.read_pickle('/Users/aljoscha/Downloads/results/data/df_group_parameters_nompC.pkl')
df_groups = pd.read_pickle('/Users/aljoscha/Downloads/results/data/df_group_parameters_NAN_IAV.pkl')

# df_initial = pd.read_pickle('/Users/aljoscha/Downloads/results/data/df_initial_sNPF.pkl')
# print(df_initial)

selected_initial = [
    ('group', '_ATR_pulse4'),
    ('group', '_ATR_'),
    # ('group', 'nompCxWT'),
    # ('single', 'WTxCrimson'),
    # ('single', 'nompCxCrimson'),
    # ('single', 'nompCxWT'),
    # ('group', 'IAVxCrimson'),
    # ('group', 'CS_WT'),
    # ('single', 'CS_WT'),
]
selected_group = [
    ('AIB', 'WTxCrimson'),
    ('AIB', 'NANxCrimson'),
    ('AIB', 'NANxWT'),
]

# plot_functions = {
#     "heatmaps": plt_heatmaps_density,
#     "encounter_metrics": plt_encounter_metrics,
#     "nd_kde": plt_nd_kde,
#     "nd_kde_fb": plt_nd_kde_fb,
#     "encounter_metrics_fb": plt_encounter_metrics_fb
# }

# num_bins = 1
num_bins = 4
encounter_duration_threshold = [10, 1800] # frames
metric1 = 'frequency'
metric2 = 'duration'
ND1 = 'PND'
ND2 = 'NND'
metric3 = 'speed_manual'

colors = [['blue','cornflowerblue'], ['red', 'salmon'], ['green', 'mediumseagreen'], ['purple', 'violet']]

#
# name = "heatmaps"
# arguments = df_initial, selected_initial, num_bins
# inputs = name, *arguments

# name = "encounter_metrics"
# arguments = df_groups, selected_group, encounter_duration_threshold, metric2
# inputs = name, *arguments
# #
# name = "nd_kde"
# arguments = df_groups, selected_group, ND1
# inputs = name, *arguments
#
# name = "nd_kde_fb"
# arguments = df_groups, selected_group, ND1, num_bins
# inputs = name, *arguments
#
# name = "encounter_metrics_fb"
# arguments = df_groups, selected_group, encounter_duration_threshold, metric1, num_bins
# inputs = name, *arguments

name = "nd_ot_cv"
arguments = df_groups, colors, selected_group, 'PND', 0, 2700, 10, 10000
inputs = name, *arguments

# name = "speed_mo"
# arguments = df_initial, selected_initial, metric3
# inputs = name, *arguments



main_plot(*inputs)  # Calls plot_sine(x_values)