import matplotlib.pyplot as plt
import numpy as np

# y = sin(x); z = cos(4x)

N = 16
discrete_setX = np.arange(0, 2 * np.pi, 2 * np.pi / N)
discrete_setY = [np.sin(x) for x in discrete_setX]
discrete_setZ = [np.cos(4 * x) for x in discrete_setX]


def fast_fourier_transform(vector, size):
    if size == 1:
        return vector

    upper_part = []
    lower_part = []
    w = 1
    w0 = complex(np.cos(2 * np.pi / size), -np.sin(2 * np.pi / size))
    for i in range(int(size / 2)):
        lower_part.append((vector[i] - vector[i + int(size / 2)]) * w)
        upper_part.append(vector[i + int(size / 2)] + vector[i])
        w = w * w0

    a = fast_fourier_transform(lower_part, size / 2)
    b = fast_fourier_transform(upper_part, size / 2)

    return b + a


def inverse_fast_fourier_transform(vector, size):
    if size == 1:
        return vector

    half_size = int(size / 2)
    a = inverse_fast_fourier_transform(vector[:half_size], half_size)
    b = inverse_fast_fourier_transform(vector[half_size:], half_size)

    w = 1
    w0 = complex(np.cos(2 * np.pi / size), -np.sin(2 * np.pi / size))
    for i in range(half_size):
        b[i] = b[i] / w
        c = (a[i] + b[i]) / 2
        d = a[i] - c
        b[i] = d
        a[i] = c
        w = w * w0

    return a + b


fft_y = [c / N for c in fast_fourier_transform(discrete_setY, N)]
fft_z = [c / N for c in fast_fourier_transform(discrete_setZ, N)]

yz_convolution = inverse_fast_fourier_transform([a * b for a,b in zip(fft_y, fft_z)], N)
fft_y = [np.conj(a) for a in fft_y]

yz_cross_correlation = inverse_fast_fourier_transform([a * b for a,b in zip(fft_y, fft_z)], N)


# plots
figure = plt.figure("lab2")

ax1 = figure.add_subplot(221)
ax1.title.set_text("y = sin(x)")
ax1.grid()
ax1.plot(discrete_setX, discrete_setY, color="black")

ax2 = figure.add_subplot(222)
ax2.title.set_text("y = cos(4x)")
ax2.grid()
ax2.plot(discrete_setX, discrete_setZ, color="blue")

ax3 = figure.add_subplot(223)
ax3.title.set_text("Свертка y(x) и z(x)")
ax3.grid()
ax3.plot(discrete_setX, yz_convolution, color="green")

ax4 = figure.add_subplot(224)
ax4.title.set_text("Корреляция y(x) и z(x)")
ax4.grid()
ax4.plot(discrete_setX, yz_cross_correlation, color="red")

plt.tight_layout()
plt.show()





