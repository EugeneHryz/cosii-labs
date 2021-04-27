from sympy import fwht
from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt

# y = sin(x) + cos(4x)
N = 16

discrete_setX = np.arange(0, 2 * np.pi, 2 * np.pi / N)
discrete_setY = [np.sin(x) + np.cos(4 * x) for x in discrete_setX]


def walsh_transformation(point_set, n):

    if n == 1:
        return point_set

    upper_half = []
    lower_half = []
    half_size = int(n / 2)
    for i in range(half_size):
        upper_half.append(point_set[i] + point_set[i + half_size])
        lower_half.append(point_set[i] - point_set[i + half_size])

    a = walsh_transformation(upper_half, half_size)
    b = walsh_transformation(lower_half, half_size)

    return a + b


matrix = linalg.hadamard(N)
walsh_discrete_transform = np.dot(matrix, discrete_setY)

walsh_transformation_result = [r / 8 for r in walsh_transformation(discrete_setY, N)]
print(walsh_transformation_result)

figure = plt.figure("lab3")

ax1 = figure.add_subplot(221)
ax1.title.set_text("y = sin(x) + cos(4x)")
ax1.grid()
ax1.plot(discrete_setX, discrete_setY, color="black")

ax2 = figure.add_subplot(222)
ax2.title.set_text("БПУ")
ax2.grid()
ax2.plot([x / N for x in range(N)], walsh_transformation_result, color="blue")

ax3 = figure.add_subplot(223)
ax3.title.set_text("ДПУ")
ax3.grid()
ax3.plot([x / N for x in range(N)], walsh_discrete_transform, color="red")

plt.tight_layout()
plt.show()


