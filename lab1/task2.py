# Wnioski:
# Przesunięcie sygnału w czasie powoduje obrót fazy o wartość proporcjonalną do częstotliwości.
# Widmo amplitudowe pozostaje niezmienne dla różnych przesunięć n0, co potwierdza, że przesunięcie w czasie nie wpływa na rozkład energii w widmie.

import numpy as np
import matplotlib.pyplot as plt
from task1 import getSpectrums


def plotSpectrumsSchifted(signal_wave, freq_linespace, n0_values):
    plt.figure(figsize=(12, 8))

    for i, n0 in enumerate(n0_values):
        signal_wave = A * np.cos(2 * np.pi * (n - n0) / N)
        _, magnitude_spectrum, phase_spectrum = getSpectrums(signal_wave)

        # Amplitude spectrum
        plt.subplot(len(n0_values), 2, 2 * i + 1)
        plt.stem(freq_linespace, magnitude_spectrum)
        plt.xlim(-3, 3)
        plt.title(f'Widmo amplitudowe dla n0 = {n0}')
        plt.xlabel('Częstotliwość [Hz]')
        plt.ylabel('Amplituda')

        # Phase spectrum
        plt.subplot(len(n0_values), 2, 2 * i + 2)
        plt.stem(freq_linespace, phase_spectrum)
        plt.xlim(-3, 3)
        plt.title(f'Widmo fazowe dla n0 = {n0}')
        plt.xlabel('Częstotliwość [Hz]')
        plt.ylabel('Faza [rad]')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    A = 2
    N = 48
    n = np.arange(N)
    n0_values = [0, N//4, N//2, 3*N//4]

    freq_linespace = np.fft.fftfreq(N, 1/N)

    plotSpectrumsSchifted(n, freq_linespace, n0_values)
