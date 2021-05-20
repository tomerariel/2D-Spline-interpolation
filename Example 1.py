import spline as sp
import numpy as np
import matplotlib.pyplot as plt

coordinates = [(0, 0), (0.42, 0.46), (1.04, 1.33), (2, 3), (3.2, 4.8),
               (5.12, 6.2), (7.25, 6.14),(8.8, 5.18), (10, 4), (10.1, 2.6),
               (9.16, 2), (7.85, 2.5), (6.4, 2.7), (5.8, 1.4), (5.2, 0.2),
               (4.5, -0.8), (3.4, -1.5), (2.4, -0.5), (1, -0.9), (0.14, -0.6), (0, 0)]
c = [list(c) for c in zip(*coordinates)]

t = np.linspace(0, 10, len(coordinates))
t_range = np.arange(0, t[-1], 0.1)
curve = sp.Spline2D(t, c[0], c[1])
points = [curve.interpolate(t) for t in t_range]
px = [p[0] for p in points]
py = [p[1] for p in points]
plt.figure()
plt.scatter(px, py, marker='.', label='interpolation')
plt.scatter(c[0], c[1], c='red', marker='X', label='real')
plt.legend()
plt.show()
