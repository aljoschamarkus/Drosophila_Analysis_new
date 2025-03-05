import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
main_dir = "/Users/aljoscha/Downloads/2012_nompC_4min_50p"
dish_radius = 6.5

# Define conditions to filter subdirectories
conditions = ["nompCxCrimson", "group"]  # Add your desired conditions here

results_dir = os.path.join(main_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

for sub_dir in os.listdir(main_dir):
    # Check if the subdirectory name includes all the specified conditions
    if not all(condition.lower() in sub_dir.lower() for condition in conditions):
        continue  # Skip this subdirectory if it doesn't match all conditions

    sub_dir_path = os.path.join(main_dir, sub_dir)
    if os.path.isdir(sub_dir_path):
        for png_file in os.listdir(sub_dir_path):
            if png_file.endswith('.png') and not png_file.lower().startswith('background'):
                image = cv2.imread(os.path.join(sub_dir_path, png_file), cv2.IMREAD_COLOR)
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

                # If circles are detected, find the largest circle
                if detected_circles is not None:
                    detected_circles = np.uint16(np.around(detected_circles))
                    # Sort circles by radius in descending order
                    largest_circle = max(detected_circles[0, :], key=lambda c: c[2])
                    x, y, radius = largest_circle

                    # Create output_image with the circle drawn (only if needed for plt.imshow)
                    output_image = image.copy()
                    cv2.circle(output_image, (x, y), radius, (255, 255, 255), 3)  # Draw the largest circle
                    cv2.circle(output_image, (x, y), 3, (255, 255, 255), 3)  # Draw the center of the largest circle
                    output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib

                data_trial = []
                data_dir = os.path.join(sub_dir_path, 'data')
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

                # Draw the circle and its midpoint
                circle = plt.Circle((x, y), radius, color='grey', fill=False, linewidth=3)
                plt.gca().add_patch(circle)
                plt.scatter(x, y, color='grey', s=30, label='Midpoint')  # Midpoint as a red dot

                # Uncomment the line below if you want to include the PNG in the plot
                plt.imshow(output_image_rgb)

                plt.colorbar(label='Frame')
                plt.xlim(x - (radius + 100), x + (radius + 100))
                plt.ylim(y - (radius + 20), y + (radius + 20))
                plt.gca().set_aspect('equal', adjustable='box')  # Ensure the circle is not distorted
                plt.legend()  # Show legend for the midpoint
                plt.show()
                print(sub_dir)