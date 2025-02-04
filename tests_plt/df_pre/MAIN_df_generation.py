import os
import cv2
import numpy as np
import pandas as pd

from package.util_df_generation import handle_main_dir
from package.util_df_generation import create_actual_groups
from package.util_df_generation import create_artificial_groups_bootstrapped

from package import config_settings

# Constants and configuration
main_dir = config_settings.main_dir
condition = config_settings.condition
genotype = config_settings.genotype
quality = config_settings.quality
dish_radius = config_settings.dish_radius
group_size = config_settings.group_size
bootstrap_reps = config_settings.bootstrap_reps

counter = 0

# Prepare the output data
data_frame = []

# Get the directory structure
condition_dir = handle_main_dir(main_dir, condition)

# Process each condition directory
for i in range(len(condition_dir) - 1):
    for sub_dir in os.listdir(condition_dir[i][0]):

        # Identify the genotype from the folder name
        geno = next((g for g in genotype if g in sub_dir), None)
        if not geno:
            print(f"Skipping: {sub_dir}")
            continue  # Skip directories without a matching genotype

        sub_dir_path = os.path.join(condition_dir[i][0], sub_dir)
        if not os.path.isdir(sub_dir_path):
            continue

        # Process images in the subdirectory
        shift = [0, 0]
        x_mid, y_mid, radius = None, None, None  # Initialize circle values
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
                    csv_data['sub_dir'] = sub_dir
                    csv_data['condition'] = condition_dir[i][1]
                    csv_data['genotype'] = geno
                    csv_data['individual_id'] = f'individual_{counter}'


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

# Save the final DataFrame to a pickle file
df_initial_path = os.path.join(condition_dir[2], "data_frame_initial.pkl")
df_initial.to_pickle(df_initial_path)

df_actual_groups = create_actual_groups(df_initial, condition_dir[0][1])

# Example usage of create_artificial_groups_bootstrapped
df_artificial_groups_mapping = create_artificial_groups_bootstrapped(
    df=df_initial,
    group_size=group_size,
    bootstrap_reps=bootstrap_reps,  # Apply bootstrapping twice
    condition=condition_dir[1][1]  # Replace with your condition values
)

group_id = "1_ID_WTxCrimson_0"
group_members = df_artificial_groups_mapping[df_artificial_groups_mapping['group_id'] == group_id]['sub_dir']
print(group_members)

group_data = df_initial.loc[df_initial.index.get_level_values('sub_dir').isin(group_members)]
print(group_data)

for idx in df_artificial_groups_mapping['group_id'].unique():
    group_members = df_artificial_groups_mapping[df_artificial_groups_mapping['group_id'] == idx]['sub_dir']
    group_data = df_initial.loc[df_initial.index.get_level_values('sub_dir').isin(group_members)]
    print(group_data)