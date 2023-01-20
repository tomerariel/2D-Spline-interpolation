import bisect

import numpy as np


class Spline:  # natural cubic spline, one dimensional
    def __init__(self, t, y):
        self.t = t
        self.y = y
        self.n = len(t)
        self.h = self._intervals()
        self.a, self.b, self.c, self.d = self._find_coefficients()
        self.m = self.linear_spline()

    def _intervals(self):
        return [(self.t[i + 1] - self.t[i]) for i in range(self.n - 1)]

    def linear_spline(self):
        return [
            (self.y[i + 1] - self.y[i]) / (self.t[i + 1] - self.t[i])
            for i in range(self.n - 1)
        ]

    def _find_coefficients(self):
        a, n, h = self.y, self.n, self.h
        m = np.zeros((n, n))
        u = np.zeros((n, 1))
        m[0, 0] = m[n - 1, n - 1] = 1
        for i in range(1, n - 1):
            m[i, i - 1] = h[i - 1]
            m[i, i] = 2 * (h[i] + h[i - 1])
            m[i, i + 1] = h[i]
            u[i, 0] = 3 * ((a[i + 1] - a[i]) / h[i] - (a[i] - a[i - 1]) / h[i - 1])
        c = np.linalg.solve(m, u).flatten()
        b = [
            float((a[i + 1] - a[i]) / h[i] - h[i] * (2 * c[i] + c[i + 1]) / 3)
            for i in range(n - 1)
        ]
        d = [float((c[i + 1] - c[i]) / 3 * h[i]) for i in range(n - 1)]
        return a, b, c, d

    def _find_index(self, z):
        if z > self.t[self.n - 1] or z < self.t[0]:
            return None
        return bisect.bisect(self.t, z) - 1

    def derivatives(self, z, i=None):
        if i is None:
            i = self._find_index(z)
        r = z - i
        dz = 3 * self.d[i] * r**2 + 2 * self.c[i] * r + self.b[i]
        ddz = 6 * self.d[i] * r + 2 * self.c[i]
        return dz, ddz

    def curvature(self, z):
        i = np.floor(z)
        if i < 0:
            i = 0
        elif i >= self.n:
            i = self.n - 1
        dz, ddz = self.derivatives(z, i)
        return ddz / ((1 + dz**2) ** 1.5)

    def interpolate(self, z):
        i = self._find_index(z)
        if i is None:
            return None
        r = z - self.t[i]
        return (
            (self.d[i] * r**3) + (self.c[i] * r**2) + (self.b[i] * r) + (self.a[i])
        )

    def linear(self, z):
        i = self._find_index(z)
        if i is None:
            return None
        r = z - self.t[i]
        if i < self.n - 1:
            return self.m[i] * r + self.y[i]
        elif i == self.n - 1:
            return self.m[i - 1] * r + self.y[i]


class Spline2D:  # natural cubic spline, two dimensional
    def __init__(self, t, x, y):
        self.t = t
        self.x = x
        self.y = y
        self.n = len(t)
        self.sx = Spline(t, x)
        self.sy = Spline(t, y)
        self.curv = [self.curvature(t) for t in self.t[: self.n - 1]]

    def curvature(self, z):
        dx, ddx = self.sx.derivatives(z)
        dy, ddy = self.sy.derivatives(z)
        return (dx * ddy - dy * ddx) / (dx**2 + dy**2) ** (3 / 2)

    def _find_index(self, z):
        if z > self.t[self.n - 1] or z < self.t[0]:
            print("Out of range")
            return None
        return bisect.bisect(self.t, z) - 1

    def mode(self, z):
        i = self._find_index(z)
        if i is None:
            return None
        k1, k2 = abs(np.mean(self.curv[: i + 1])), abs(np.mean(self.curv[i::]))
        sig1 = np.std(self.curv[: i + 1])
        return "curve" if k1 - 2 * sig1 < k2 < k1 + 2 * sig1 else "linear"

    def interpolate(self, z):
        if z > self.t[self.n - 1] or z < self.t[0]:
            return None
        mode = self.mode(z)
        if mode == "curve":
            return self.sx.interpolate(z), self.sy.interpolate(z), "curve"
        elif mode == "linear":
            return self.sx.linear(z), self.sy.linear(z), "linear"
