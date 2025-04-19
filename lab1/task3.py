# Wnioski:
# Dopełnienie zerami wydłuża sygnał, co skutkuje większą liczbą próbek w transformatcie Fouriera.
# Pozwala to na bardziej precyzyjne odwzorowanie kształtu widma amplitudowego i fazowego.
# Dodatkowo zwiększa się zakres analizowanych częstotliwości.
# Dodanie zer nie zmienia fizycznych właściwości sygnału, a jedynie zwiększa rozdzielczość jego widma.

import numpy as np
import matplotlib.pyplot as plt
from task1 import getSpectrums


def addZeros(signal_wave, N0):
    signal_with_zeros = np.concatenate((signal_wave, np.zeros(N0)))

    freq_padded = np.fft.fftfreq(len(signal_with_zeros), 1/len(signal_with_zeros))
    _, magnitude_spectrum, phase_spectrum = getSpectrums(signal_with_zeros)

    return freq_padded, magnitude_spectrum, phase_spectrum


def plotSpectrumsWithZeros(signal_wave, N0_values):
    plt.figure(figsize=(12, 8))

    for i, N0 in enumerate(N0_values):
        freq_padded, magnitude_spectrum, phase_spectrum = addZeros(signal_wave, N0)

        # Amplitude spectrum
        plt.subplot(len(N0_values), 2, 2 * i + 1)
        plt.stem(freq_padded, magnitude_spectrum)
        plt.title(f'Widmo amplitudowe dla N0 = {N0}')
        plt.xlabel('Częstotliwość [Hz]')
        plt.ylabel('Amplituda')

        # Phase spectrum
        plt.subplot(len(N0_values), 2, 2 * i + 2)
        plt.stem(freq_padded, phase_spectrum)
        plt.title(f'Widmo fazowe dla N0 = {N0}')
        plt.xlabel('Częstotliwość [Hz]')
        plt.ylabel('Faza [rad]')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    A = 3
    N = 10
    n = np.arange(N)
    signal_wave = A * (1 - (n % N) / N)
    N0_values = [0, 1*N, 4*N, 9*N]

    plotSpectrumsWithZeros(signal_wave, N0_values)
