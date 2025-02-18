import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import expon

df = np.random.random(size=500) * 5
x = np.array(list(range(0, 500, 1))) * 0.01
y = (x**2 + df)
plt.scatter(x, y)
plt.show()

