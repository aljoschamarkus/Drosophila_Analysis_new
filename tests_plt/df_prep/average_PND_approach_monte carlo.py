import numpy as np
from scipy.spatial import distance_matrix

# Define parameters
radius = 6.5  # cm
num_points = 5
num_trials = 1000000  # Monte Carlo simulations

# Function to generate random points inside a circle
def generate_random_points_in_circle(radius, num_points):
    angles = np.random.uniform(0, 2 * np.pi, num_points)
    radii = np.sqrt(np.random.uniform(0, radius**2, num_points))  # Ensures uniform distribution
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack((x, y))

# Monte Carlo simulation
avg_distances = []
for _ in range(num_trials):
    points = generate_random_points_in_circle(radius, num_points)
    dist_matrix = distance_matrix(points, points)  # Compute pairwise distances
    np.fill_diagonal(dist_matrix, np.inf)  # Ignore self-distances
    avg_neighbor_dist = np.mean(np.min(dist_matrix, axis=1))  # Average nearest-neighbor distance
    avg_distances.append(avg_neighbor_dist)

# Compute final average neighbor distance
average_neighbor_distance = np.mean(avg_distances)
print(f"Average nearest-neighbor distance: {average_neighbor_distance} cm")