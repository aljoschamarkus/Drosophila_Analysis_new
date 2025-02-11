import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tqdm

# Load the DataFrame from the pickle file
df_final = pd.read_pickle(
    '/Users/aljoscha/Downloads/2012_nompC_Crimson_WT_4min_50p_TRex_beta_2101/results/data_frame_initial.pkl'
)

print("Index names:", df_final.index.names)
print("Columns:", df_final.columns.tolist())

# Extract data based on your specific condition, genotype, and sub_dir
condition = "group"
genotype = "nompCxCrimson"
sub_dir = "2024.10.12_02-46-05_4min_50p_nompCxCrimson_group_5"

# Filter the dataframe to select the specific data
df_final = df_final.sort_index()
df_filtered = df_final.loc[(sub_dir, condition, genotype), :]

num_frames2 = df_filtered.index.get_level_values('frame').nunique()
progress_bar = tqdm.tqdm(total=num_frames2, desc="Animating Frames", unit="frame", dynamic_ncols=True)

# Create the figure and axis for plotting
fig, ax = plt.subplots()
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)
ax.set_xlim(0, 14)  # Adjust for padding
ax.set_ylim(0, 14)  # Adjust for padding
# ax.set_xlabel('cm')
# ax.set_ylabel('cm')
plt.gca().set_aspect('equal', adjustable='box')
circle = plt.Circle((7, 7), 6.5, color='blue', fill=False, linewidth=2)
plt.gca().add_patch(circle)
# ax.set_title(f"Trajectory of {genotype} - {condition} - {sub_dir}")

# Create a dictionary for line objects for current trajectories
trajectories = {}
# Create a dictionary for previous trajectory (60 frames) for each individual
previous_trajectories = {}

# Buffer to store the previous 60 frames for each individual
history = {individual: ([], [], []) for individual in df_filtered.index.get_level_values('individual_id').unique()}

# Initialize the plot limits for the animation
def init():
    # Initialize empty data for each individual trajectory
    for individual_id in df_filtered.index.get_level_values('individual_id').unique():
        trajectories[individual_id], = ax.plot([], [], 'ro', markersize=4)  # Current trajectory (individual)
        previous_trajectories[individual_id], = ax.plot([], [], 'go', markersize=1, alpha=0.5)  # Initial transparency
    return [line for line in trajectories.values()] + [line for line in previous_trajectories.values()]

# Function to update the plot in each frame of the animation
def update(frame):
    # Filter the DataFrame for the current frame
    progress_bar.update(1)  # Manually update tqdm progress
    ax.set_title(f"Frame {frame}/{num_frames}")  # Optional: Update title for progress visualization
    frame_data = df_filtered.loc[df_filtered.index.get_level_values('frame') == frame]

    all_x = []
    all_y = []

    # Iterate over each individual to update their current and previous trajectory
    for individual_id in frame_data.index.get_level_values('individual_id').unique():
        individual_data = frame_data.loc[individual_id]
        x_data = individual_data['x'].tolist() if isinstance(individual_data['x'], pd.Series) else [
            individual_data['x']]
        y_data = individual_data['y'].tolist() if isinstance(individual_data['y'], pd.Series) else [
            individual_data['y']]

        # Update the current trajectory for this individual
        trajectories[individual_id].set_data(x_data, y_data)

        # Store the current data in the history buffer
        history[individual_id][0].append(x_data)  # Store x position
        history[individual_id][1].append(y_data)  # Store y position

        # Limit the buffer size to 60 frames for each individual
        if len(history[individual_id][0]) > 400:
            history[individual_id][0].pop(0)
            history[individual_id][1].pop(0)

        # Update the previous trajectory for this individual
        previous_trajectories[individual_id].set_data(history[individual_id][0], history[individual_id][1])

        # Accumulate all x and y for the previous trajectories to animate them together
        all_x.extend(history[individual_id][0])
        all_y.extend(history[individual_id][1])

    # Update the previous trajectory (showing the last 60 frames for all individuals)
    return [line for line in trajectories.values()] + [line for line in previous_trajectories.values()]

# Set up the animation (number of frames corresponds to the range of frames in the DataFrame)
num_frames = df_filtered.index.get_level_values('frame').nunique()  # Number of unique frames
ani = animation.FuncAnimation(
    fig, update, frames=range(1, num_frames + 1),
    init_func=init, blit=True, repeat=True, interval=0.5 # Faster animation
)
# ani.save('/Users/aljoscha/Downloads/animation5.mp4', writer='ffmpeg', fps=30, dpi=600)
# Show the animation
plt.show()
# plt.close()