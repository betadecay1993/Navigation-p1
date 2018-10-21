import pickle
from scipy.signals import savgol_filter
import numpy as np
from matplotlib import pyplot as plt
scores = pickle.load(open('banana_scores', 'rb+'))


#PLOT SCORES
x = np.arange(len(scores))
xs = 13.0*np.ones(len(scores))
plt.xkcd()
plt.plot(xs,'k--', linewidth = 2)
plt.plot(scores, linewidth = 3, color = 'Red')

plt.title("Gathered bananas per episode")
plt.xlabel("Number of an episode")
plt.ylabel("Bananas")
plt.grid(True)
plt.show()