import pandas as pd


df = pd.read_pickle('/Users/aljoscha/Downloads/results/data/df_group_parameters_p2.pkl')

colors = [['blue','cornflowerblue'], ['green', 'mediumseagreen'], ['red', 'salmon'], ['purple', 'violet']]

selected = [
    ('AIB', 'NANxCrimson'),
    ('AIB', 'IAVxCrimson'),
    ('AIB', 'WTxCrimson'),
    # ('RGN', 'CS_WT'),
]

inputs = df, colors, selected, 'NND', 3, 2700, 10000

def plt_nd_ot_cv(df, colors, selected, ND, rolling_window, end_intervall, num_trials=10000):
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    import numpy as np
    from scipy.spatial import distance_matrix
    from package.config_settings import circle_default
    from package.config_settings import group_size


    plt.figure(figsize=(10, 6))

    for i, (group, genotype) in enumerate(selected):

        df_pre = df.xs((group, genotype), level=['group_type', 'genotype'])
        df_plt = df_pre.groupby(['frame']).mean()
        df_clean = df_plt.dropna(subset=[ND])

        if rolling_window == 0:
            x_data = df_clean.index.get_level_values('frame').to_numpy() / 1800
            y_data_nd = df_clean[ND].to_numpy()
        elif rolling_window > 0:
            x_data = df_clean.index.get_level_values('frame').to_numpy() / 1800
            y_data_nd = df_clean[ND].rolling(window=91, center=True).mean().dropna().to_numpy()  # Remove NaNs

        # Ensure both x_data and y_data_nd have the same length
        min_length = min(len(x_data), len(y_data_nd))
        x_data = x_data[:min_length]
        y_data_nd = y_data_nd[:min_length]


        max_frame = df_pre.index.get_level_values('frame').max()
        df_last_1800 = df_pre.xs(slice(max_frame - end_intervall, max_frame), level='frame', drop_level=False)
        avg_nd_end = df_last_1800[ND].mean()

        def exp_growth(x, a, b, c):
            return a * (1 - np.exp(-b * x)) + c

        popt_exp, _ = curve_fit(exp_growth, x_data, y_data_nd)
        a_exp, b_exp, c_exp = popt_exp

        y_pred_exp = exp_growth(x_data, *popt_exp)
        ss_res_exp = np.sum((y_data_nd - y_pred_exp) ** 2)
        ss_tot_exp = np.sum((y_data_nd - np.mean(y_data_nd)) ** 2)
        r_squared_exp = 1 - (ss_res_exp / ss_tot_exp)

        x_0 = - (1 / b_exp) * np.log(1 + c_exp / a_exp)
        dt = abs(x_0) * 60

        # Extend x range for plotting
        x_extended = np.linspace(min(x_data) - abs(x_0), max(x_data), 500)

        plt.scatter(x_data, y_data_nd, label=f"{group}-{genotype} data", color=colors[i][1], s=10)
        plt.plot(x_data, exp_growth(x_data, *popt_exp),
                 label=f"fit (R²={r_squared_exp:.3f}), asymptote: {a_exp + c_exp:.3f}, slope parameter: {b_exp:.3f}",
                 color=colors[i][0])

        # plt.plot(x_extended, exp_growth(x_extended, *popt_exp),
        #                  label=f"{group}{genotype} (R²={r_squared_exp:.3f}), asymptote: {a_exp + c_exp:.3f}, slope parameter: {b_exp:.3f}",
        #                  color=colors[i][0])

        # plt.axvline(x=x_0, color=colors[i][0], linestyle='dashed')  # , label=f"x_0 = {x_0:.3f}")
        plt.axhline(avg_nd_end, color=colors[i][0], label=f' avg PND end: {avg_nd_end:.3f}', linestyle='dashed')

        # plt.text(x_pos, 0.5, f'$y = {a_exp + c_exp:.3f}(1 - e^{{-{b_exp:.3f}(x+{abs(x_0):.3f})}})$',
        #          transform=plt.gca().transAxes, fontsize=10, verticalalignment='center', color=colors[1])
        # plt.text(x_pos, 0.45, f'delayed start: {dt:.3f}',
        #          transform=plt.gca().transAxes, fontsize=10, verticalalignment='center', color=colors[1])

        print(f"{group}-{genotype}")
        print(f"R^2 {r_squared_exp}")
        print(f'y = {a_exp:.2f}(1 - e^(-{b_exp:.2f}x)) + {c_exp:.2f}')
        print(f"avg {ND} end", avg_nd_end)
        print(f" y=0 at x_0 = {x_0:.3f}")
        print(f"delayed start: {dt:.3f}")

    radius = circle_default[2]
    num_points = group_size

    # Function to generate random points inside a circle
    def generate_random_points_in_circle(radius, num_points):
        angles = np.random.uniform(0, 2 * np.pi, num_points)
        radii = np.sqrt(np.random.uniform(0, radius**2, num_points))  # Ensures uniform distribution
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        return np.column_stack((x, y))

    # Monte Carlo simulation
    avg_nd_mc = []

    if ND == 'NND':
        for _ in range(num_trials):
            points = generate_random_points_in_circle(radius, num_points)
            dist_matrix = distance_matrix(points, points)  # Compute pairwise distances
            np.fill_diagonal(dist_matrix, np.inf)  # Ignore self-distances
            avg_nnd_pre = np.mean(np.min(dist_matrix, axis=1))  # Average nearest-neighbor distance
            avg_nd_mc.append(avg_nnd_pre)
    elif ND == 'PND':
        for _ in range(num_trials):
            points = generate_random_points_in_circle(radius, num_points)
            dist_matrix = distance_matrix(points, points)  # Compute pairwise distances
            np.fill_diagonal(dist_matrix, np.nan)  # Ignore self-distances by setting them to NaN
            avg_pnd_pre = np.nanmean(dist_matrix)  # Average of all pairwise distances
            avg_nd_mc.append(avg_pnd_pre)

    avg_nd_mc = np.mean(avg_nd_mc)

    plt.axhline(avg_nd_mc, color='purple', label=f'statistical avg: {avg_nd_mc:.3f}', linestyle='dotted', linewidth=3)

    plt.legend()
    plt.xlabel("Time (min)")
    plt.ylabel("Pairwise Distance (cm)")
    plt.show()

plt_nd_ot_cv(*inputs)



