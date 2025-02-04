import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

def handle_main_dir(main_directory, condition):
    group_dir, single_dir = None, None  # Initialize variables with default values

    results_dir = os.path.join(main_directory, 'results')
    os.makedirs(results_dir, exist_ok=True)
    # Ensure condition2 has at least two elements
    if not isinstance(condition, (list, tuple)) or len(condition) < 2:
        raise ValueError("condition2 must be a list or tuple with at least two elements.")

    for folder in os.listdir(main_directory):
        folder_lower = folder.lower()  # Avoid recalculating `folder.lower()`
        if any(part in folder_lower for part in condition):
            if condition[0] in folder_lower:
                group_dir = [os.path.join(main_directory, folder), condition[0]]
            elif condition[1] in folder_lower:
                single_dir = [os.path.join(main_directory, folder), condition[1]]

    # Return the directories (or None if not found)
    return group_dir, single_dir, results_dir

main_dir = "/Users/aljoscha/Downloads/2012_nompC_Crimson_WT_4min_50p_TRex_beta_2101"
condition = ["group", "single"]
genotype = ["nompCxCrimson", "WTxCrimson", "nompCxWT"]
stimulation_used = "625nm 1µW/mm^2"
colors = [['#e41a1c', '#377eb8', '#4daf4a'], ['#fbb4ae', '#a6cee3', '#b2df8a']]
line_styles = ["-", "--"]
quality = [1296, 972]
group_size = 5
data_len = 7190
dish_radius = 6.5

genotype_mapping = {
    "nompCxCrimson": "nompCxCrimson",
    "WTxCrimson": "WTxCrimson",
    "nompCxWT": "nompCxWT"
}

# counter = 1

condition_dir = handle_main_dir(main_dir, condition)
print(condition_dir)
for i in range(len(condition_dir)-1):
    for sub_dir in os.listdir(condition_dir[i][0]):

        geno = None
        for key, value in genotype_mapping.items():
            if key in sub_dir:
                geno = value
                break

        if not geno:
            print(f"Genotype not found for directory: {sub_dir}")
            continue  # Skip directories without a matching genotype

            # Check if the sub_dir_path is a directory
        if os.path.isdir(os.path.join(condition_dir[i][0], sub_dir)):
            for png_file in os.listdir(os.path.join(condition_dir[i][0], sub_dir)):
                if png_file.endswith('.png'):
                    image = cv2.imread(os.path.join(condition_dir[i][0], sub_dir, png_file), cv2.IMREAD_COLOR)
                    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 2)

                    detected_circles = cv2.HoughCircles(
                        blurred_image,
                        cv2.HOUGH_GRADIENT,
                        dp=1.2,  # Inverse ratio of the accumulator resolution to the image resolution
                        minDist=100,  # Minimum distance between circle centers
                        param1=50,  # Higher threshold for Canny edge detector
                        param2=30,  # Threshold for center detection
                        minRadius=10,  # Minimum radius of detected circles
                        maxRadius=500  # Set to 0 to allow OpenCV to detect any circle size
                    )

                    output_image = image.copy()

                    # If circles are detected, find the largest circle
                    if detected_circles is not None:
                        detected_circles = np.uint16(np.around(detected_circles))
                        # Sort circles by radius in descending order
                        largest_circle = max(detected_circles[0, :], key=lambda c: c[2])
                        x, y, radius = largest_circle
                        # Draw the largest circle
                        cv2.circle(output_image, (x, y), radius, (255, 255, 255), 3)
                        # Draw the center of the largest circle
                        cv2.circle(output_image, (x, y), 3, (255, 255, 255), 3)

                    # Convert the image to RGB for displaying with matplotlib
                    output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

        data_trial = []
        data_dir = os.path.join(condition_dir[i][0], sub_dir, 'data')
        if os.path.isdir(data_dir):
            for csv_file in os.listdir(data_dir):
                if csv_file.endswith('.csv'):
                    csv_path = os.path.join(data_dir, csv_file)
                    csv_data = pd.read_csv(csv_path, usecols=['frame', 'X#wcentroid (cm)', 'Y#wcentroid (cm)'])

                    # Mask rows where points fall within the circle
                    distances = np.sqrt(
                        (csv_data['X#wcentroid (cm)'] - x) ** 2 + (csv_data['Y#wcentroid (cm)'] - y) ** 2)
                    mask = distances > 1.05 * radius
                    csv_data.loc[mask, ['X#wcentroid (cm)', 'Y#wcentroid (cm)']] = np.nan

                    data_trial.append(csv_data)

        plt.figure(figsize=(10, 7))
        for csv_data in data_trial:
            x_coords = csv_data['X#wcentroid (cm)']
            y_coords = csv_data['Y#wcentroid (cm)']
            frame = csv_data['frame']

            plt.scatter(x_coords, y_coords, c=frame, s=5, cmap='viridis', alpha=0.7, edgecolor='none')

            # Inner part of the dot (default color)
        plt.colorbar(label='Frame')
        plt.imshow(output_image_rgb)
        plt.xlim(x - (radius + 100), x + (radius + 100))
        plt.ylim(y - (radius + 20), y + (radius + 20))
        plt.show()
        print(sub_dir)




print()