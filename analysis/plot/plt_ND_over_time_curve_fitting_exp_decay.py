import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

df = pd.read_pickle('/Users/aljoscha/Downloads/2012_nompC_Crimson_WT_4min_50p_TRex_beta_2101/results/data/df_group_parameters.pkl')
print("Index names:", df.index.names)
print("Columns:", df.columns.tolist())
df_plt = df.xs('RGN', level='group_type')
df_plt = df_plt.groupby(['frame']).mean()
x_data = df_plt.index / 1800
y_data = df_plt['pairwise_distance']

plt.scatter(x_data, y_data, s=5)
plt.show()

# Define functions
def exp_growth(x, a, b, c):
    return a * (1 - np.exp(-b * x)) + c

# Initialize a dictionary to store R^2 values
r_squared_dict = {}

# Fit the exponential growth function
popt_exp, _ = curve_fit(exp_growth, x_data, y_data)
y_pred_exp = exp_growth(x_data, *popt_exp)
ss_res_exp = np.sum((y_data - y_pred_exp) ** 2)
ss_tot_exp = np.sum((y_data - np.mean(y_data)) ** 2)
r_squared_exp = 1 - (ss_res_exp / ss_tot_exp)
r_squared_dict["Exponential Growth"] = r_squared_exp

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, label="Data", color="cornflowerblue", s=10)

# Plot the exponential growth function with the formula on the plot
a_exp, b_exp, c_exp = popt_exp
plt.plot(x_data, exp_growth(x_data, *popt_exp), label=f"Exponential Growth (R²={r_squared_exp:.3f})", color="red")
plt.text(0.05, 0.95, f'$y = {a_exp:.3f}(1 - e^{{-{b_exp:.3f}x}}) + {c_exp:.3f}$',
         transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', color='red')

# Customize plot
plt.legend()
plt.xlabel("Time (x)")
plt.ylabel("Pairwise Distance (y)")
plt.title("Function Fitting with R² Values")
# plt.grid(True)
plt.show()

# Print R^2 values for reference
print("R^2 Values:")
for func, r_squared in r_squared_dict.items():
    print(f"{func}: {r_squared}")

print(f'y = {a_exp:.2f}(1 - e^(-{b_exp:.2f}x)) + {c_exp:.2f}')