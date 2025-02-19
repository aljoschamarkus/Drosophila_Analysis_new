import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

df = pd.read_pickle('/Users/aljoscha/Downloads/2012_nompC_Crimson_WT_4min_50p_TRex_beta_2101/results/data/df_group_parameters.pkl')
df_RGN_nomp = df.xs(('RGN', 'nompCxCrimson'), level=['group_type', 'genotype'])
df_AIB_nomp = df.xs(('AIB', 'nompCxCrimson'), level=['group_type', 'genotype'])
df_RGN_WT = df.xs(('RGN', 'WTxCrimson'), level=['group_type', 'genotype'])
df_AIB_WT = df.xs(('AIB', 'WTxCrimson'), level=['group_type', 'genotype'])
df_RGN = df.xs('RGN', level='group_type')
df_AIB = df.xs('AIB', level='group_type')
plt.figure(figsize=(10, 6))

colors = [['red', 'salmon'], ['blue','cornflowerblue'], ['green', 'mediumseagreen'], ['purple', 'violet']]
def plt_ND_over_time_curve_fitting_exp_decay(df, colors, x_pos, option):

    import numpy as np
    from scipy.spatial import distance_matrix

    df_plt = df.groupby(['frame']).mean()
    df_clean = df_plt.dropna(subset=['pairwise_distance'])

    # Extract x_data from the index (assuming 'frame' is part of the index)
    x_data = df_clean.index.get_level_values('frame').to_numpy() / 1800

    # Extract y_data from the column
    y_data_pnd = df_clean['pairwise_distance'].to_numpy()
    # x_data = df_clean.index / 1800
    # y_data_pnd = df_clean['pairwise_distance']
    # y_data_nnd = df_clean['nearest_neighbor_distance']

    radius = 6.5  # cm
    num_points = 5
    num_trials = 1000  # Monte Carlo simulations

    # Function to generate random points inside a circle
    def generate_random_points_in_circle(radius, num_points):
        angles = np.random.uniform(0, 2 * np.pi, num_points)
        radii = np.sqrt(np.random.uniform(0, radius**2, num_points))  # Ensures uniform distribution
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
    avg_nnd = df_last_1800['nearest_neighbor_distance'].mean()
    avg_pnd = df_last_1800['pairwise_distance'].mean()
    # df2 = df_last_1800.groupby(['frame']).mean()
    # average_pairwise_distance1 = df2['pairwise_distance'].mean()
    print("Average pairwise distance for last 1800 frames:", avg_nnd)
    print("Average pairwise distance for last 1800 frames:", avg_pnd)
    # print(average_pairwise_distance / average_pairwise_distance1)

    def exp_growth(x, a, b, c):
        return a * (1 - np.exp(-b * x)) + c

    popt_exp, _ = curve_fit(exp_growth, x_data, y_data_pnd)
    a_exp, b_exp, c_exp = popt_exp

    y_pred_exp = exp_growth(x_data, *popt_exp)
    ss_res_exp = np.sum((y_data_pnd - y_pred_exp) ** 2)
    ss_tot_exp = np.sum((y_data_pnd - np.mean(y_data_pnd)) ** 2)
    r_squared_exp = 1 - (ss_res_exp / ss_tot_exp)

    x_0 = - (1 / b_exp) * np.log(1 + c_exp / a_exp)
    print(f"The curve crosses y=0 at x_0 = {x_0:.3f}")
    dt = abs(x_0) * 60

    # Extend x range for plotting
    # x_extended = np.linspace(min(x_data) - abs(x_0), max(x_data), 500)

    # Plot results
    # plt.scatter(x_data, y_data_pnd, label=f"{option}", color=colors[1], s=10)
    plt.plot(x_data, exp_growth(x_data, *popt_exp), label=f"{option} (R²={r_squared_exp:.3f}), asymptote: {a_exp + c_exp:.3f}, slope parameter: {b_exp:.3f}",
             color=colors[0])

    # Mark x_0 on the plot
    # plt.axvline(x=x_0, color=colors, linestyle='dashed')#, label=f"x_0 = {x_0:.3f}")
    # plt.axhline(y=, color='black', linewidth=0.8)

    # Labels and legend
    # plt.title("Extended Exponential Growth Fit")
    # plt.show()
    plt.axhline(avg_pnd, x_0, 4, color=colors[1], label=f'{option} avg PND min 3-4: {avg_pnd:.3f}', linestyle='dashed')
    # plt.axhline(avg_pnd_mc, x_0, 4, color='green', label=f'avg PND mc simulation: {avg_pnd_mc:.3f}')

    # plt.text(x_pos, 0.2, f'fitted curve: $y = {a_exp:.3f}(1 - e^{{-{b_exp:.3f}x}}) + {c_exp:.3f}$',
    #          transform=plt.gca().transAxes, fontsize=10, verticalalignment='center', color=colors[1])
    # plt.text(x_pos, 0.35, f'$y = {a_exp + c_exp:.3f}(1 - e^{{-{b_exp:.3f}(x+{abs(x_0):.3f})}})$',
    #          transform=plt.gca().transAxes, fontsize=10, verticalalignment='center', color=colors[1])
    # plt.text(x_pos, 0.1, f'delayed start: {dt:.3f}',
    #          transform=plt.gca().transAxes, fontsize=10, verticalalignment='center', color=colors[1])

    plt.legend(facecolor=None, edgecolor=None, framealpha=0.0)
    plt.xlabel("Time (min)")
    plt.ylabel("Pairwise Distance (cm)")
    # plt.show()

    print(f"R^2 value {r_squared_exp}")

    print(f'y = {a_exp:.2f}(1 - e^(-{b_exp:.2f}x)) + {c_exp:.2f}')

plt_ND_over_time_curve_fitting_exp_decay(df_RGN_nomp, colors[0], 0, 'groups-nompCxCrimson')
plt_ND_over_time_curve_fitting_exp_decay(df_AIB_nomp, colors[1], 0, 'individuals-nompCxCrimson')
plt_ND_over_time_curve_fitting_exp_decay(df_RGN_WT, colors[2], 0, 'groups-WTxCrimson')
plt_ND_over_time_curve_fitting_exp_decay(df_AIB_WT, colors[3], 0, 'individuals-WTxCrimson')

plt.savefig("/Users/aljoscha/Downloads/plot3.pdf", transparent=True)
plt.show()
