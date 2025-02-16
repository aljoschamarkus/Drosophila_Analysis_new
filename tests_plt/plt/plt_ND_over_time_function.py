import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_pickle('/Users/aljoscha/Downloads/2012_nompC_Crimson_WT_4min_50p_TRex_beta_2101/results/data/df_group_parameters.pkl')
selected_option = [['nompCxCrimson', 'RGN'], ['nompCxCrimson', 'AIB']]

def plt_nd(df, selected_option):
    for idx in selected_option:
        df_plt = df.xs((idx[0], idx[1]), level=['genotype', 'group_type'])
        df_plt = df_plt.groupby(['frame']).mean()
        plt.scatter(df_plt.index / 1800, df_plt['nearest_neighbor_distance'], s=5)
        plt.xlabel('Time [min]')
        plt.ylabel('Nearest Neighbor Distance [cm]')
    plt.show()
    for idx in selected_option:
        df_plt = df.xs((idx[0], idx[1]), level=['genotype', 'group_type'])
        sns.kdeplot(df_plt['nearest_neighbor_distance'], fill=True, color='r')
        plt.xlabel('Nearest Neighbor Distance [cm]')
    plt.show()

    print('plt_ND - Done')

plt_nd(df, selected_option)