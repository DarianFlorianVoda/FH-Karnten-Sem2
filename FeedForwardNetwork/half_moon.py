import numpy as np
from matplotlib import pyplot as plt


def create_half_moon(n=2000, w=0.2, r=0.6, d=0.1):
    c1 = np.array([(r + w) / 2, d / 2])
    c2 = np.array([-(r + w) / 2, -d / 2])

    # We use random radius in the interval [rad, rad+thk]
    #  and random angles from 0 to pi radians.
    r1 = np.random.rand(n) * w + r
    a1 = np.random.rand(n) * np.pi

    r2 = np.random.rand(n) * w + r
    a2 = np.random.rand(n) * np.pi + np.pi

    # In order to plot it we convert it to cartesian:
    p1 = np.array((r1 * np.cos(a1), r1 * np.sin(a1)))
    p2 = np.array((r2 * np.cos(a2), r2 * np.sin(a2)))

    x1, y1 = (p1[0] - c1[0], p1[1] - c1[1])
    x2, y2 = (p2[0] - c2[0], p2[1] - c2[1])

    plt.scatter(x1, y1, marker='.', linewidths=0.1)
    plt.scatter(x2, y2, marker='.', linewidths=0.1)
    plt.show()

    return x1, y1, x2, y2


def label_half_moon(n=2000, w=0.2, r=0.6, d=0.1):
    x1, y1, x2, y2 = create_half_moon(n=n, w=w, r=r, d=d)
    X = list()
    y = list()

    for i in range(0, len(x1)):
        x1_list = [x1[i], y1[i]]
        x2_list = [x2[i], y2[i]]

        X.append(x1_list)
        y.append(0)
        X.append(x2_list)
        y.append(1)

    return X, y
