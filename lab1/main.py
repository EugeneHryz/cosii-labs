import matplotlib.pyplot as plt
import numpy as np
import cmath as cm
import time

# y = sin(x) + cos(4x); N = 16
# Пусть p - период функции f(x), а q - период g(x).
# Если существуют такие целые положительные числа a и b, что
# ap = bq = r, то r является периодом суммы функций f(x) + g(x)
# В данном случае период функции y(x) равен 2pi

N = 16

step = 2 * np.pi / N
x = 0.0
discrete_setX = []
for f in range(N):
    discrete_setX.append(x)
    x = x + step

discrete_setY = [np.sin(discrete_setX[n]) + np.cos(4 * discrete_setX[n]) for n in range(N)]


def discrete_fourier_transform(point_values, inverse=False):
    fourier_t = []

    for k in range(N):
        complex_part = []
        for n in range(N):
            c = complex(np.cos(2 * np.pi * n * k / N), -np.sin(2 * np.pi * n * k / N))
            if inverse:
                c = complex(np.cos(2 * np.pi * n * k / N), np.sin(2 * np.pi * n * k / N))
            complex_part.append(c)

        y = [a * b for a, b in zip(point_values, complex_part)]
        fourier_t.append(sum(y) / N)

    return fourier_t


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


start_time = time.perf_counter()
ft = discrete_fourier_transform(discrete_setY)
end_time = time.perf_counter()
dft_cpu_time = end_time - start_time
print("dft CPU time:", dft_cpu_time, "s")


dft_amplitude_spectrum = [np.sqrt(c.real ** 2 + c.imag ** 2) for c in ft]
dft_phase_spectrum = [cm.phase(c) for c in ft]

start_time = time.perf_counter()
fft_result = [c / N for c in fast_fourier_transform(discrete_setY, N)]
end_time = time.perf_counter()
fft_cpu_time = end_time - start_time
print("fft CPU time:", fft_cpu_time, "s")
print("dft takes", dft_cpu_time / fft_cpu_time, "times longer than fft")

number_of_binary_digits = int(np.log2(N))
format_string = '0' + number_of_binary_digits.__str__() + 'b'
binary_numbers = [list(s) for s in list(map(lambda x: format(x, format_string), [i for i in range(N)]))]
for num in binary_numbers:
    num.reverse()

binary_inversion_decimal = [int(''.join(j), 2) for j in binary_numbers]

rearranged_fft = [fft_result[binary_inversion_decimal[j]] for j in range(N)]
fft_amplitude_spectrum = [np.sqrt(c.real ** 2 + c.imag ** 2) for c in rearranged_fft]
fft_phase_spectrum = [cm.phase(d) for d in rearranged_fft]

inverse_fft = inverse_fast_fourier_transform([a * N for a in fft_result], N)


# plots
figure = plt.figure("lab1")

ax0 = figure.add_subplot(331)
ax0.grid()
ax0.title.set_text("y = sin(x) + cos(4x)")
ax0.plot(discrete_setX, discrete_setY, color="black")

ax1 = figure.add_subplot(332)
ax1.grid()
ax1.title.set_text("ачх дпф")
ax1.plot(range(N), dft_amplitude_spectrum, color="red")

ax2 = figure.add_subplot(333)
ax2.grid()
ax2.title.set_text("фчх дпф")
ax2.plot(range(N), dft_phase_spectrum, color="blue")

inverse_ft = [a * N for a in discrete_fourier_transform(ft, inverse=True)]
ax3 = figure.add_subplot(334)
ax3.grid()
ax3.title.set_text("одпф")
ax3.plot(discrete_setX, inverse_ft, color='brown')

ax4 = figure.add_subplot(335)
ax4.grid()
ax4.title.set_text("ачх бпф")
ax4.plot(range(N), fft_amplitude_spectrum, color="green")

ax5 = figure.add_subplot(336)
ax5.grid()
ax5.title.set_text("фчх бпф")
ax5.plot(range(N), fft_phase_spectrum, color="purple")

ax6 = figure.add_subplot(337)
ax6.grid()
ax6.title.set_text("обпф")
ax6.plot(discrete_setX, inverse_fft, color="cyan")

plt.tight_layout()
plt.show()
