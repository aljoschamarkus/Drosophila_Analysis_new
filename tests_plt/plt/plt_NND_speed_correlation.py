import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.stats import pearsonr, linregress

# Load data
df = pd.read_pickle(
    '/Users/aljoscha/Downloads/2012_nompC_Crimson_WT_4min_50p_TRex_beta_2101/results/data_frame_initial.pkl')

# Extract subset of data
id = '2024.10.12_06-09-11_4min_50p_WTxCrimson_group_5'
df_new = df.xs((id, 'WTxCrimson', 'group'), level=['sub_dir', 'genotype', 'condition'])

# Ensure necessary columns exist
if not {'x', 'y', 'speed'}.issubset(df_new.columns):
    raise ValueError("Missing required columns in DataFrame")


# Compute Nearest Neighbor Distance (NND)
def compute_nnd(df):
    """Computes the nearest neighbor distance for each individual at each frame."""
    nnd_list = []

    for frame, group in df.groupby(level='frame'):
        coords = group[['x', 'y']].values
        if len(coords) > 1:
            dist_matrix = distance_matrix(coords, coords)
            np.fill_diagonal(dist_matrix, np.inf)  # Ignore self-distance
            nnd_values = dist_matrix.min(axis=1)
        else:
            nnd_values = np.full(len(group), np.nan)  # If only one individual, no NND

        nnd_list.extend(nnd_values)

    df['NND'] = nnd_list
    return df


df_new = compute_nnd(df_new)

# Apply Rolling Average to Speed
window_size = 30  # Adjust window size as needed
df_new['speed_rolling'] = df_new.groupby(level='individual_id')['speed'].rolling(
    window=window_size, center=True, min_periods=1).mean().reset_index(level=0, drop=True)
# df_new['speed_rolling'] = df_new.groupby(level='individual_id')['midline_offset']

# Filter: Exclude speeds > 15 and NND outside [0,2]
df_filtered = df_new[(df_new['speed_rolling'] <= 30) & (df_new['NND'].between(0, 8))].dropna()

# Calculate Pearson Correlation
corr_coeff, p_value = pearsonr(df_filtered['NND'], df_filtered['speed_rolling'])

# Print results
print(f"Pearson Correlation Coefficient: {corr_coeff:.3f}")
print(f"P-value: {p_value:.3e}")  # Scientific notation for readability

if p_value < 0.05:
    print("The correlation is statistically significant.")
else:
    print("The correlation is not statistically significant.")

if corr_coeff > 0:
    print("The correlation is positive: As NND increases, speed tends to increase.")
elif corr_coeff < 0:
    print("The correlation is negative: As NND increases, speed tends to decrease.")
else:
    print("No correlation detected.")

# Fit a Linear Regression Line
slope, intercept, _, _, _ = linregress(df_filtered['NND'], df_filtered['speed_rolling'])

# Scatter plot with regression line
plt.figure(figsize=(8, 6))
plt.scatter(df_filtered['NND'], df_filtered['speed_rolling'], alpha=0.5, label="Data")
plt.plot(df_filtered['NND'], slope * df_filtered['NND'] + intercept, color='red',
         label=f"Fit: y={slope:.2f}x + {intercept:.2f}")
plt.xlabel('Nearest Neighbor Distance (NND)')
plt.ylabel('Rolling-Averaged Speed')
plt.title('Speed vs. Nearest Neighbor Distance')
plt.legend()
plt.grid(True)
plt.show()