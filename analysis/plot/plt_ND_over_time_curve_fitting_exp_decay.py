import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

df = pd.read_pickle('/Users/aljoscha/Downloads/2012_nompC_Crimson_WT_4min_50p_TRex_beta_2101/results/data/df_group_parameters.pkl')
# df_plt = df.xs('RGN', level='group_type')
df_plt = df.groupby(['frame']).mean()
x_data = df_plt.index / 1800
y_data = df_plt['pairwise_distance']
# y_data = df_plt['nearest_neighbor_distance']

import numpy as np
from scipy.spatial import distance_matrix

# Define parameters
radius = 6.5  # cm
num_points = 5
num_trials = 100000  # Monte Carlo simulations

# Function to generate random points inside a circle
def generate_random_points_in_circle(radius, num_points):
    angles = np.random.uniform(0, 2 * np.pi, num_points)
    radii = np.sqrt(np.random.uniform(0, radius**2, num_points))  # Ensures uniform distribution
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack((x, y))

# Monte Carlo simulation
avg_distances = []
for _ in range(num_trials):
    points = generate_random_points_in_circle(radius, num_points)
    dist_matrix = distance_matrix(points, points)  # Compute pairwise distances
    np.fill_diagonal(dist_matrix, np.inf)  # Ignore self-distances
    avg_neighbor_dist = np.mean(np.min(dist_matrix, axis=1))  # Average nearest-neighbor distance
    avg_distances.append(avg_neighbor_dist)

avg_pairwise_distances = []
for _ in range(num_trials):
    points = generate_random_points_in_circle(radius, num_points)
    dist_matrix = distance_matrix(points, points)  # Compute pairwise distances
    np.fill_diagonal(dist_matrix, np.nan)  # Ignore self-distances by setting them to NaN
    avg_pairwise_dist = np.nanmean(dist_matrix)  # Average of all pairwise distances
    avg_pairwise_distances.append(avg_pairwise_dist)

# Compute final average neighbor distance
average_neighbor_distance = np.mean(avg_distances)
average_pairwise_distance = np.mean(avg_pairwise_distances)
print(f"Average nearest-neighbor distance: {average_neighbor_distance} cm")
print(f"Average pairwise distance: {average_pairwise_distance} cm")

max_frame = df.index.get_level_values('frame').max()
df_last_1800 = df.xs(slice(max_frame - 1799, max_frame), level='frame', drop_level=False)
df1 = df_last_1800.groupby(['frame'])
df2 = df_last_1800.groupby(['frame']).mean()
average_pairwise_distance = df_last_1800['pairwise_distance'].mean()
average_pairwise_distance1 = df2['pairwise_distance'].mean()
print("Average pairwise distance for last 1800 frames:", average_pairwise_distance)
print("Average pairwise distance for last 1800 frames:", average_pairwise_distance1)
print(average_pairwise_distance / average_pairwise_distance1)

def plt_ND_over_time_curve_fitting_exp_decay(x_data, y_data):

    def exp_growth(x, a, b, c):
        return a * (1 - np.exp(-b * x)) + c

    popt_exp, _ = curve_fit(exp_growth, x_data, y_data)
    y_pred_exp = exp_growth(x_data, *popt_exp)
    ss_res_exp = np.sum((y_data - y_pred_exp) ** 2)
    ss_tot_exp = np.sum((y_data - np.mean(y_data)) ** 2)
    r_squared_exp = 1 - (ss_res_exp / ss_tot_exp)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, label="Data", color="cornflowerblue", s=10)

    # Plot the exponential growth function with the formula on the plot

    a_exp, b_exp, c_exp = popt_exp
    plt.plot(x_data, exp_growth(x_data, *popt_exp), label=f"Exponential Growth (R²={r_squared_exp:.3f})", color="red")
    plt.hlines(average_pairwise_distance, 0, 4, color='purple', linestyles='solid', label='avg PND last minute', data=None)

    plt.text(0.05, 0.95, f'fitted curve: $y = {a_exp:.3f}(1 - e^{{-{b_exp:.3f}x}}) + {c_exp:.3f}$',
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='bottom')
    plt.text(0.05, 0.95, f'start midpoint: $y = {a_exp + c_exp:.3f}(1 - e^{{-{b_exp:.3f}x}})$',
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

    plt.xlim(0, 4)

    plt.legend()
    plt.xlabel("Time (x)")
    plt.ylabel("Pairwise Distance (y)")
    plt.show()

    print(f"R^2 value {r_squared_exp}")

    print(f'y = {a_exp:.2f}(1 - e^(-{b_exp:.2f}x)) + {c_exp:.2f}')

plt_ND_over_time_curve_fitting_exp_decay(x_data, y_data)
