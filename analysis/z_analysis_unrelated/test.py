import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import expon

df = np.random.random(size=500) * 5
x = np.array(list(range(0, 500, 1))) * 0.01
y = (x**2 + df)
plt.scatter(x, y)
plt.show()

df = pd.read_pickle('/Users/aljoscha/Downloads/results/data/df_initial.pkl')
print("Columns:", df.columns.tolist())
print("Index names:", df.index.names)

df2 = pd.read_pickle('/Users/aljoscha/Downloads/results/data/df_groups.pkl')
print("Columns:", df2.columns.tolist())
print("Index names:", df2.index.names)

df3 = pd.read_pickle('/Users/aljoscha/Downloads/2012_nompC_Crimson_WT_4min_50p_TRex_beta_2101/results/data/df_group_parameters.pkl')
print("Columns:", df3.columns.tolist())
print("Index names:", df3.index.names)

