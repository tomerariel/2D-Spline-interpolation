import spline as sp
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 2 * np.pi, 20)
kx = 3
ky = 2
x = np.cos(kx * t)
y = np.sin(ky * t)

curve = sp.Spline2D(t, x, y)
t_range = np.linspace(0.01, 2 * np.pi - 0.01, 100)
points = [curve.interpolate(t) for t in t_range]
px = [p[0] for p in points]
py = [p[1] for p in points]
plt.figure()
plt.plot(px, py, marker='.', label='interpolation')
plt.scatter(x, y, c='red', marker='X', label='real')
plt.legend()
plt.show()


