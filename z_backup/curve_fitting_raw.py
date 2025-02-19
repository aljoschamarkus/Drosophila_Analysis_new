import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

df = pd.read_pickle('/Users/aljoscha/Downloads/results/data/df_group_parameters.pkl')
df_RGN_nomp = df.xs(('RGN', 'nompCxCrimson'), level=['group_type', 'genotype'])
df_AIB_nomp = df.xs(('AIB', 'nompCxCrimson'), level=['group_type', 'genotype'])
df_RGN_WT = df.xs(('RGN', 'WTxCrimson'), level=['group_type', 'genotype'])
df_AIB_WT = df.xs(('AIB', 'WTxCrimson'), level=['group_type', 'genotype'])
df_RGN = df.xs('RGN', level='group_type')
df_AIB = df.xs('AIB', level='group_type')
plt.figure(figsize=(10, 6))

colors = [['red', 'salmon'], ['blue','cornflowerblue'], ['green', 'mediumseagreen'], ['purple', 'violet']]
def plt_ND_over_time_curve_fitting_exp_decay(df_list, labels, colors, x_pos):
    import numpy as np
    from scipy.spatial import distance_matrix

    plt.figure(figsize=(10, 6))

    all_x_data = []
    all_y_data = []

    for df, label, color in zip(df_list, labels, colors):
        df_clean = df.dropna(subset=['PND'])
        x_data = df_clean.index.get_level_values('frame').to_numpy() / 1800
        y_data_pnd = df_clean['PND'].to_numpy()

        all_x_data.extend(x_data)
        all_y_data.extend(y_data_pnd)

        plt.scatter(x_data, y_data_pnd, label=f"{label} Raw Data", color=color[1], s=10, alpha=0.6)

    all_x_data = np.array(all_x_data)
    all_y_data = np.array(all_y_data)

    def exp_growth(x, a, b, c):
        return a * (1 - np.exp(-b * x)) + c

    popt_exp, _ = curve_fit(exp_growth, all_x_data, all_y_data)
    a_exp, b_exp, c_exp = popt_exp

    y_pred_exp = exp_growth(all_x_data, *popt_exp)
    ss_res_exp = np.sum((all_y_data - y_pred_exp) ** 2)
    ss_tot_exp = np.sum((all_y_data - np.mean(all_y_data)) ** 2)
    r_squared_exp = 1 - (ss_res_exp / ss_tot_exp)

    x_fit = np.linspace(min(all_x_data), max(all_x_data), 500)
    y_fit = exp_growth(x_fit, *popt_exp)

    plt.plot(x_fit, y_fit, label=f"Fit (R²={r_squared_exp:.3f})", color="black", linewidth=2)
    # plt.plot(x_data, exp_growth(x_data, *popt_exp),
    #          label=f"{option} (R²={r_squared_exp:.3f}), asymptote: {a_exp + c_exp:.3f}, slope parameter: {b_exp:.3f}",
    #          color=colors[0])


    plt.legend()
    plt.xlabel("Time (min)")
    plt.ylabel("Pairwise Distance (cm)")
    plt.title("Raw Data with Single Fit")
    radius = 6.5  # cm
    num_points = 5
    num_trials = 1000  # Monte Carlo simulations

    # Function to generate random points inside a circle
    def generate_random_points_in_circle(radius, num_points):
        angles = np.random.uniform(0, 2 * np.pi, num_points)
        radii = np.sqrt(np.random.uniform(0, radius ** 2, num_points))  # Ensures uniform distribution
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        return np.column_stack((x, y))

    # Monte Carlo simulation
    avg_nnd_mc = []
    for _ in range(num_trials):
        points = generate_random_points_in_circle(radius, num_points)
        dist_matrix = distance_matrix(points, points)  # Compute pairwise distances
        np.fill_diagonal(dist_matrix, np.inf)  # Ignore self-distances
        avg_nnd_pre = np.mean(np.min(dist_matrix, axis=1))  # Average nearest-neighbor distance
        avg_nnd_mc.append(avg_nnd_pre)

    avg_pnd_mc = []
    for _ in range(num_trials):
        points = generate_random_points_in_circle(radius, num_points)
        dist_matrix = distance_matrix(points, points)  # Compute pairwise distances
        np.fill_diagonal(dist_matrix, np.nan)  # Ignore self-distances by setting them to NaN
        avg_pnd_pre = np.nanmean(dist_matrix)  # Average of all pairwise distances
        avg_pnd_mc.append(avg_pnd_pre)

    # Compute final average neighbor distance
    avg_nnd_mc = np.mean(avg_nnd_mc)
    avg_pnd_mc = np.mean(avg_pnd_mc)
    print(f"Average nearest-neighbor distance: {avg_nnd_mc} cm")
    print(f"Average pairwise distance: {avg_pnd_mc} cm")

    max_frame = df.index.get_level_values('frame').max()
    df_last_1800 = df.xs(slice(max_frame - 1799, max_frame), level='frame', drop_level=False)
    avg_nnd = df_last_1800['NND'].mean()
    avg_pnd = df_last_1800['PND'].mean()
    # df2 = df_last_1800.groupby(['frame']).mean()
    # average_pairwise_distance1 = df2['pairwise_distance'].mean()
    print("Average pairwise distance for last 1800 frames:", avg_nnd)
    print("Average pairwise distance for last 1800 frames:", avg_pnd)
    print(a_exp + c_exp)
    print(avg_pnd_mc)
    print(avg_pnd)
    plt.axhline(avg_pnd, color='purple', label=f'avg PND min 3-4: {avg_pnd:.3f}', linestyle='dashed')
    plt.axhline(avg_pnd_mc, color='purple', label=f'statistical avg: {avg_pnd_mc:.3f}', linestyle='dotted')
    plt.show()

# Combine datasets for a single fit
plt_ND_over_time_curve_fitting_exp_decay(
    [df_AIB, df_RGN],  # List of datasets
    ['Individuals', 'Groups'],  # Labels
    [['red', 'salmon'], ['blue', 'cornflowerblue']],  # Colors
    0.7
)