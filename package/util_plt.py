import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def plot(x, y, midpoint, radius):
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.5, label="Data")
    plt.xlabel('Nearest Neighbor Distance (NND)')
    plt.ylabel('Rolling-Averaged Speed')
    plt.title('Speed vs. Nearest Neighbor Distance')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal', adjustable ='box')
    plt.tight_layout()
    plt.legend()
    plt.grid(True)
    plt.show()
    circle = plt.Circle(midpoint, radius, color='red', fill=False, linewidth=2)
    plt.gca().add_patch(circle)
    # plt.savefig(os.path.join(condition_dir[2], f"{cond}_{geno}.png"))
    # plt.close()
    # cbar = fig.colorbar(cax, ax=axes, orientation='vertical', label='Density (log scale)', fraction=0.02, pad=0.04)
    return None