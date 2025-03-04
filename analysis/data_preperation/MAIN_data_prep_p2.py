import os
import cv2
import numpy as np
import pandas as pd

from package.config_settings import bootstrap_reps_p2
from package.util_data_preperation import handle_main_dir_p2
from package.util_data_preperation import create_mapping_actual_groups
from package.util_data_preperation import create_mapping_artificial_groups_bootstrapped
from package.util_data_preperation import compute_pairwise_distances_and_encounters

from package import config_settings

# Constants and configuration
main_dir = config_settings.main_dir_p2
genotype = config_settings.genotype_p2
quality = config_settings.quality
dish_radius = config_settings.dish_radius
group_size = config_settings.group_size
bootstrap_reps = config_settings.bootstrap_reps_p2

fps = config_settings.fps
speed_initial_threshold = config_settings.speed_initial_threshold
speed_avg_threshold = config_settings.speed_avg_threshold
speed_avg_window = config_settings.speed_avg_window
encounter_distance_threshold = config_settings.encounter_distance_threshold

counter = 0

# Prepare the output data
data_frame = []

# Get the directory structure
results_data_dir, results_plt_dir = handle_main_dir_p2()

# Process each subdirectory in main_dir
for sub_dir in os.listdir(main_dir):
    sub_dir_path = os.path.join(main_dir, sub_dir)
    if not os.path.isdir(sub_dir_path):
        continue

    # Identify condition from folder name
    if "group" in sub_dir.lower():
        cond = "group"
    elif "single" in sub_dir.lower():
        cond = "single"
    else:
        print(f"Skipping: {sub_dir} (Unknown condition)")
        continue  # Skip unknown conditions

    # Identify the genotype from the folder name
    geno = next((g for g in genotype if g in sub_dir), None)
    if not geno:
        print(f"Skipping: {sub_dir} (No matching genotype)")
        continue

    # Process images in the subdirectory
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

                distances = np.sqrt((csv_data['X#wcentroid (cm)'] - x_mid)**2 + (csv_data['Y#wcentroid (cm)'] - y_mid)**2)
                mask = distances > 1.05 * radius
                csv_data.loc[mask, ['X#wcentroid (cm)', 'Y#wcentroid (cm)']] = np.nan

                csv_data['X#wcentroid (cm)'] -= x_mid
                csv_data['Y#wcentroid (cm)'] -= y_mid
                csv_data['X#wcentroid (cm)'] *= conversion_factor
                csv_data['Y#wcentroid (cm)'] *= conversion_factor
                csv_data['X#wcentroid (cm)'] += 7
                csv_data['Y#wcentroid (cm)'] += 7
                csv_data['SPEED#wcentroid (cm/s)'] *= conversion_factor
                csv_data['speed_processed'] = csv_data['SPEED#wcentroid (cm/s)']

                csv_data['midline_offset_signless'] = np.abs(csv_data['MIDLINE_OFFSET'])
                csv_data['sub_dir'] = sub_dir
                csv_data['condition'] = cond
                csv_data['genotype'] = geno
                csv_data['individual_id'] = f'I_ID_{counter}'

                data_frame.append(csv_data)
                counter += 1

# Combine all data
df_initial = pd.concat(data_frame, ignore_index=True)
df_initial.set_index(['sub_dir', 'condition', 'genotype', 'individual_id', 'frame'], inplace=True)
df_initial.rename(columns={
    'MIDLINE_OFFSET': 'midline_offset',
    'SPEED#wcentroid (cm/s)': 'speed',
    'X#wcentroid (cm)': 'x',
    'Y#wcentroid (cm)': 'y'
}, inplace=True)

df_initial.to_pickle(os.path.join(results_data_dir, "df_initial_p2.pkl"))

# Mapping Functions
mapping_functions = {
    "map_RGN": create_mapping_actual_groups,
    "map_AIB": create_mapping_artificial_groups_bootstrapped,
}

mapping_conditions = {
    "map_RGN": "group",
    "map_AIB": "single",
}

mapping_list = []
for name, func in mapping_functions.items():
    kwargs = {
        "df": df_initial,
    }
    if "group_size" in func.__code__.co_varnames:
        kwargs["group_size"] = group_size
    if "bootstrap_reps" in func.__code__.co_varnames:
        kwargs["bootstrap_reps"] = bootstrap_reps_p2
    mapping_list.append(func(**kwargs))

map_RGN, map_AIB = mapping_list
map_RGN['group_type'] = 'RGN'
map_AIB['group_type'] = 'AIB'

map_combined = pd.concat([map_RGN, map_AIB])
df_merged = df_initial.reset_index().merge(map_combined, on='individual_id', how='inner')
df_groups = df_merged.set_index(['sub_dir', 'condition', 'genotype', 'group_type', 'group_id', 'individual_id', 'frame'])
df_groups.to_pickle(os.path.join(results_data_dir, "df_groups_p2.pkl"))

df_group_parameters = compute_pairwise_distances_and_encounters(df_groups, encounter_distance_threshold)
df_group_parameters.to_pickle(os.path.join(results_data_dir, "df_group_parameters_p2.pkl"))
