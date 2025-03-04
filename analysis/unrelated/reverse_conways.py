import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import ImageOps

def load_image(image_path):
    """Load an image and convert it to a binary black-and-white numpy array."""
    image = Image.open(image_path).convert('L')
    image_original = Image.open(image_path)

    image = ImageOps.invert(image)
    image = image.resize((315, 45))
    plt.imshow(image_original)
    plt.show()
    binary_image = np.array(image) > 0  # Threshold to create a binary image
    return binary_image.astype(int)

def conways_game_of_life(grid):
    """Run one iteration of Conway's Game of Life."""
    neighbors = sum(np.roll(np.roll(grid, i, 0), j, 1)
              for i in (-1, 0, 1) for j in (-1, 0, 1) if (i != 0 or j != 0))
    return (neighbors == 3) | (grid & (neighbors == 2))

def generate_frames(initial_grid, num_frames):
    """Generate frames by running Conway's Game of Life."""
    frames = [initial_grid]
    current_grid = initial_grid.copy()
    for _ in range(num_frames - 1):
        current_grid = conways_game_of_life(current_grid)
        frames.append(current_grid)
    return frames

def animate_frames(frames):
    """Animate the frames in reverse order."""
    fig, ax = plt.subplots()
    im = ax.imshow(frames[0], cmap='binary', interpolation='nearest')
    plt.axis('off')

    def update(frame):
        im.set_data(frames[-(frame + 1)])# Play backwards
        return im,

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=100, blit=True, repeat=False)
    ani.save('/Users/aljoscha/Downloads/animation1.mp4', writer='ffmpeg', fps=20, dpi=600)
    plt.show()

def main(image_path, num_frames=40):
    """Main function to load, process, and animate the image."""
    binary_image = load_image(image_path)
    frames = generate_frames(binary_image, num_frames)
    animate_frames(frames)

if __name__ == "__main__":
    image_path = "/Users/aljoscha/Downloads/ABDCE.jpeg"  # Replace with your image path
    main(image_path)