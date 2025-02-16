import pandas as pd

# Load your DataFrame
df = pd.read_pickle('/Users/aljoscha/Downloads/2012_nompC_Crimson_WT_4min_50p_TRex_beta_2101/results/data/df_group_parameters.pkl')
df1 = df.xs('RGN', level='group_type')
df2 = df.xs('AIB', level='group_type')
df3 = df.xs(('RGN', 'nompCxCrimson'), level=['group_type', 'genotype'])
df4 = df.xs(('RGN', 'WTxCrimson'), level=['group_type', 'genotype'])

df5 = [df1, df2, df3, df4]
PND = []
for df in df5:

    max_frame = df.index.get_level_values('frame').max()
    df_last_1800 = df.xs(slice(max_frame - 1799, max_frame), level='frame', drop_level=False)
    df_plt = df_last_1800.groupby(['frame']).mean()

    from scipy.stats import shapiro


    # # Calculate the average pairwise_distance
    # average_pairwise_distance = df_last_1800['pairwise_distance'].mean()
    # df_f = df_last_1800['pairwise_distance'].dropna()
    # df_f = df_f[np.isfinite(df_f)]
    # stat, p = shapiro(df_f)


    average_pairwise_distance = df_plt['pairwise_distance'].mean()
    # df_f = df_last_1800['pairwise_distance'].dropna()
    # df_f = df_f[np.isfinite(df_f)]
    stat, p = shapiro(df_plt['pairwise_distance'])


    if p > 0.05:
        print("Data is normally distributed (p-value:", p, ")")
    else:
        print("Data is NOT normally distributed (p-value:", p, ")")
    PND.append(df_plt['pairwise_distance'])

    print("Average pairwise distance for last 1800 frames:", average_pairwise_distance)

from scipy.stats import mannwhitneyu

u_stat, p_value = mannwhitneyu(PND[0], PND[1])

print(f"U-statistic: {u_stat}, P-value: {p_value}")

if p_value < 0.05:
    print("Significant difference found!")
else:
    print("No significant difference.")