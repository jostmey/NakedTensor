import numpy as np
import matplotlib.pyplot as plt

m = 0.3	# Slope
b = -0.8 # Intercept

xs = np.arange(0.0, 8.0, 1.0) # Generate features
r = np.random.normal(scale=0.25, size=len(xs))
ys = np.round(m*xs+b+r, 2) # Generate labels

print(xs.tolist()) # Print features
print(ys.tolist()) # Print labels

fig = plt.figure(figsize=(3, 2))
plt.xticks([0, 2, 4, 6, 8])
plt.xlim([xs[0]-0.5, xs[-1]+0.5])
plt.yticks([-1, 0, 1])
plt.ylim([ys[0]-0.5, ys[-1]+0.5])
plt.plot(xs, ys, 'x', markersize=8, markeredgewidth=4)
fig.savefig('dataset.svg')

