import pandas as pd

df = pd.read_pickle('/Users/aljoscha/Downloads/2012_nompC_Crimson_WT_4min_50p_TRex_beta_2101/results/data/mapping_artificial_groups_bootstrapped.pkl')

print("Index names:", df.index.names)
print("Columns:", df.columns.tolist())
print("Shape:", df.shape)
print("Head:", df.head())
print("Tail:", df.tail())
print("Group IDs:", df['group_id'].unique())
print("Group Counts:", df['group_id'].value_counts())
print("Group Sizes:", df.groupby('group_id').size())
print(df)