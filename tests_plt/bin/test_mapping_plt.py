import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_initial = pd.read_pickle(
    '/Users/aljoscha/Downloads/2012_nompC_Crimson_WT_4min_50p_TRex_beta_2101/results/data_frame_initial.pkl')

df_artificial_groups_mapping = pd.read_pickle('/Users/aljoscha/Downloads/2012_nompC_Crimson_WT_4min_50p_TRex_beta_2101/results/mapping_actual_groups.pkl')

# Define the target condition and genotype
target_condition = "group"
target_genotype = "WTxCrimson"

# Step 1: Filter df_initial to include only the target condition and genotype
df_filtered = df_initial[
    (df_initial.index.get_level_values('condition') == target_condition) &
    (df_initial.index.get_level_values('genotype') == target_genotype)
]

# Step 2: Identify sub_dirs that belong to the target condition & genotype
valid_sub_dirs = df_filtered.index.get_level_values('sub_dir').unique()

# Step 3: Filter df_artificial_groups_mapping to keep only groups containing these sub_dirs
filtered_groups = df_artificial_groups_mapping[
    df_artificial_groups_mapping['sub_dir'].isin(valid_sub_dirs)
]

# Step 4: Extract unique group_ids from the filtered groups
valid_group_ids = filtered_groups['group_id'].unique()
print(valid_group_ids)

# Step 5: Iterate through only the valid group_ids and extract data
for idx in valid_group_ids:
    group_members = filtered_groups[filtered_groups['group_id'] == idx]['sub_dir']

    # Retrieve only relevant group data from df_filtered
    group_data = df_filtered.loc[df_filtered.index.get_level_values('sub_dir').isin(group_members)]

    df.dropna(inplace=True)
    speed_plt = df.xs((cond, geno, 'actual'), level=['condition', 'genotype', 'group_type'])['speed'].values
    # speed_plt = speed_plt[speed_plt != 0]
    # color_speed = colors[cond_idx][geno_idx]  # Use colors[0] for speed_plt
    # sns.kdeplot(speed_plt, color=color_speed, fill=False, alpha=1)
    sns.kdeplot(midline_offset_plt ** 2, color=normalized_color_list[counter], fill=True, alpha=0.08,
                label=f"{cond}_{geno}")
    plt.hist(
        speed_plt,
        bins=4000,
        color='red',
        alpha= 0.5,
        label=f"{idx}",
        density=True
    )
    # Set the legend after plotting all histograms
    plt.legend(loc="upper right")
    plt.title("Speed Distributions by Condition and Genotype")
    plt.xlim(-1, 18)
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.show()
