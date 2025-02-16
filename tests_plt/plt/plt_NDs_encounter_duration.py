import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_pickle('/Users/aljoscha/Downloads/2012_nompC_Crimson_WT_4min_50p_TRex_beta_2101/results/data/df_group_parameters.pkl')

df2 = df.xs(('nompCxCrimson', 'RGN'), level=['genotype', 'group_type'])
df22 = df.xs(('nompCxCrimson', 'AIB'), level=['genotype', 'group_type'])
df3 = df2.groupby(['frame']).mean()
df32 = df22.groupby(['frame']).mean()
print("Columns:", df2.columns.tolist())
print("Index names:", df2.index.names)

plt.scatter(df3.index ,df3['nearest_neighbor_distance'], c='r', s=5)
plt.scatter(df32.index ,df32['nearest_neighbor_distance'], c='salmon', s=5)
plt.scatter(df3.index ,df3['pairwise_distance'], c='b', s=5)
plt.scatter(df32.index ,df32['pairwise_distance'], c='cornflowerblue', s=5)

# plt.scatter(df3.index ,df3['x'], c='r', s=5)
# plt.scatter(df3.index ,df3['y'], c='b', s=5)
plt.show()

sns.kdeplot(df2['nearest_neighbor_distance'], fill=True, color='r')
sns.kdeplot(df22['nearest_neighbor_distance'], fill=True, color='b')
plt.show()

