import matplotlib.pyplot as plt
import numpy as np

# Example data
groups = ['Treatment', 'Control 1', 'Control 2']
condition_1 = [75, 50, 55]
condition_2 = [80, 45, 60]

x = np.arange(len(groups))  # Group positions
width = 0.35  # Width of the bars

# Colors
colors = [['#e41a1c', '#377eb8', '#4daf4a'], ['#fbb4ae', '#a6cee3', '#b2df8a']]

# Create bar plot
fig, ax = plt.subplots(figsize=(8, 6))
bar1 = ax.bar(x - width/2, condition_1, width, label='Condition 1', color=colors[0])
bar2 = ax.bar(x + width/2, condition_2, width, label='Condition 2', color=colors[1])

# Add labels, title, and legend
ax.set_xlabel('Groups')
ax.set_ylabel('Values')
ax.set_title('Treatment and Control Groups by Condition')
ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.legend()

# Display plot
plt.tight_layout()
plt.show()