import os
import cv2
import numpy as np
import pandas as pd

from package.util_data_preperation import handle_main_dir
from package.util_data_preperation import create_mapping_actual_groups
from package.util_data_preperation import create_mapping_artificial_groups_bootstrapped
from package.util_data_preperation import create_mapping_semi_artificial_groups_bootstrapped
from package.util_data_preperation import compute_pairwise_distances_and_encounters

from package import config_settings

# Constants and configuration
main_dir = config_settings.main_dir
condition = config_settings.condition
genotype = config_settings.genotype
quality = config_settings.quality
dish_radius = config_settings.dish_radius
group_size = config_settings.group_size
bootstrap_reps = config_settings.bootstrap_reps
distance_threshold_encounter = config_settings.distance_threshold_encounter

counter = 0

# Prepare the output data
data_frame = []

# Get the directory structure
group_dir, single_dir, results_data_dir, results_plt_dir = handle_main_dir(main_dir, condition)
condition_dir = [group_dir, single_dir]

# Process each condition directory
for i in range(len(condition)):
    for sub_dir in os.listdir(condition_dir[i]):

        # Identify the genotype from the folder name
        geno = next((g for g in genotype if g in sub_dir), None)
        if not geno:
            print(f"Skipping: {sub_dir}")
            continue  # Skip directories without a matching genotype

        sub_dir_path = os.path.join(condition_dir[i], sub_dir)
        if not os.path.isdir(sub_dir_path):
            continue

        # Process images in the subdirectory
        shift = [0, 0]
        x_mid, y_mid, radius = None, None, None  # Initialize circle values
        conversion_factor = None
        for png_file in os.listdir(sub_dir_path):
            if png_file.endswith('.png'):
                image_path = os.path.join(sub_dir_path, png_file)
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 2)

                # Detect circles
                detected_circles = cv2.HoughCircles(
                    blurred_image, cv2.HOUGH_GRADIENT,
                    dp=1.2, minDist=100,
                    param1=50, param2=30,
                    minRadius=10, maxRadius=500
                )

                # Calculate the shift based on the largest circle
                if detected_circles is not None:
                    detected_circles = np.uint16(np.around(detected_circles))
                    largest_circle = max(detected_circles[0, :], key=lambda c: c[2])
                    x_mid, y_mid, radius = largest_circle
                    conversion_factor = dish_radius / radius
                break  # Stop after processing the first image

        # Process CSV files in the subdirectory
        data_dir = os.path.join(sub_dir_path, 'data')
        if os.path.isdir(data_dir):
            for csv_file in os.listdir(data_dir):
                if csv_file.endswith('.csv'):
                    csv_path = os.path.join(data_dir, csv_file)
                    csv_data = pd.read_csv(csv_path, usecols=['frame', 'X#wcentroid (cm)', 'Y#wcentroid (cm)', 'SPEED#wcentroid (cm/s)', 'MIDLINE_OFFSET'])

                    # Mask rows where points fall within the circle
                    distances = np.sqrt((csv_data['X#wcentroid (cm)'] - x_mid)**2 + (csv_data['Y#wcentroid (cm)'] - y_mid)**2)
                    mask = distances > 1.05 * radius
                    csv_data.loc[mask, ['X#wcentroid (cm)', 'Y#wcentroid (cm)']] = np.nan

                    # Apply shift and conversion factor
                    csv_data['X#wcentroid (cm)'] -= x_mid
                    csv_data['Y#wcentroid (cm)'] -= y_mid
                    csv_data['X#wcentroid (cm)'] *= conversion_factor
                    csv_data['Y#wcentroid (cm)'] *= conversion_factor
                    csv_data['X#wcentroid (cm)'] += 7
                    csv_data['Y#wcentroid (cm)'] += 7
                    csv_data['SPEED#wcentroid (cm/s)'] *= conversion_factor * 30

                    # Add metadata to the DataFrame
                    csv_data['midline_offset_signless'] = np.abs(csv_data['MIDLINE_OFFSET'])
                    csv_data['sub_dir'] = sub_dir
                    csv_data['condition'] = condition[i]
                    csv_data['genotype'] = geno
                    csv_data['individual_id'] = f'I_ID_{counter}'


                    # Append the data to the main list
                    data_frame.append(csv_data)

                    counter += 1

# Concatenate all data into a single DataFrame
df_initial = pd.concat(data_frame, ignore_index=True)
indices = ['sub_dir', 'condition', 'genotype', 'individual_id', 'frame']
df_initial.set_index(indices, inplace=True)
df_initial = df_initial.rename(columns={
    'MIDLINE_OFFSET': 'midline_offset',
    'SPEED#wcentroid (cm/s)': 'speed',
    'X#wcentroid (cm)': 'x',
    'Y#wcentroid (cm)': 'y'
})
df_initial.to_pickle(os.path.join(results_data_dir, "df_initial.pkl"))

print("Index names:", df_initial.index.names)
print("Columns:", df_initial.columns.tolist())
print("Data frame shape:", df_initial.shape)

# Define a dictionary mapping names to functions
mapping_functions = {
    "map_RGN": create_mapping_actual_groups,
    "map_AIB": create_mapping_artificial_groups_bootstrapped,
    "map_AGB": create_mapping_semi_artificial_groups_bootstrapped,
}

# Define the corresponding conditions for each function
mapping_conditions = {
    "map_RGN": condition[0],
    "map_AIB": condition[1],
    "map_AGB": condition[0],
}

mapping_list = []
# Loop through the mapping functions
for name, func in mapping_functions.items():
    # Determine function arguments dynamically
    kwargs = {
        "df": df_initial,
        "condition": mapping_conditions[name],
    }

    # Add optional arguments if they exist for the function
    if "group_size" in func.__code__.co_varnames:
        kwargs["group_size"] = group_size
    if "bootstrap_reps" in func.__code__.co_varnames:
        kwargs["bootstrap_reps"] = bootstrap_reps

    # Call function dynamically
    mapping = func(**kwargs)
    mapping_list.append(mapping)

# Now you can access the data as:
map_RGN = mapping_list[0]
map_AIB = mapping_list[1]
map_AGB = mapping_list[2]

map_RGN['group_type'] = 'RGN'
map_AIB['group_type'] = 'AIB'
map_AGB['group_type'] = 'AGB'

map_combined = pd.concat([map_RGN, map_AIB, map_AGB])

df_merged = df_initial.reset_index().merge(map_combined, on='individual_id', how='inner')

df_groups = df_merged.set_index(['sub_dir', 'condition', 'genotype', 'group_type', 'group_id', 'individual_id', 'frame'])

df_groups.to_pickle(os.path.join(results_data_dir, "df_groups.pkl"))

print("Index names:", df_groups.index.names)
print("Columns:", df_groups.columns.tolist())
print("Data frame shape:", df_groups.shape)

df_group_parameters = compute_pairwise_distances_and_encounters(df_groups, distance_threshold_encounter)
df_group_parameters.to_pickle(os.path.join(results_data_dir, "df_group_parameters.pkl"))

print("Index names:", df_group_parameters.index.names)
print("Columns:", df_group_parameters.columns.tolist())
print("Data frame shape:", df_group_parameters.shape)
