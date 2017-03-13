import numpy as np
import matplotlib.pyplot as plt

xs = [ 0.00,  1.00,  2.00, 3.00, 4.00, 5.00, 6.00, 7.00] # Features
ys = [-0.82, -0.94, -0.12, 0.26, 0.39, 0.64, 1.02, 1.00] # Labels

m = 0.297022 # Slope
b = -0.860827 # Intercept

x_line = np.arange(np.min(xs)-0.5, np.max(xs)+0.5, 0.001)
y_line = m*x_line+b

fig = plt.figure(figsize=(3, 2))
plt.xticks([0, 2, 4, 6, 8])
plt.xlim([xs[0]-0.5, xs[-1]+0.5])
plt.yticks([-1, 0, 1])
plt.ylim([ys[0]-0.5, ys[-1]+0.5])
plt.plot(xs, ys, 'x', markersize=8, markeredgewidth=4)
plt.plot(x_line, y_line, linewidth=4)
fig.savefig('line_of_best_fit.svg')

