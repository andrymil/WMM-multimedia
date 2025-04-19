import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig


def getSignal(
    time_linespace,
    a1,
    f1,
    a2,
    f2,
    a3,
    f3,
):
    signal = (
        a1 * np.sin(2 * np.pi * f1 * time_linespace)
        + a2 * np.sin(2 * np.pi * f2 * time_linespace)
        + a3 * np.sin(2 * np.pi * f3 * time_linespace)
    )
    return signal


def getFreqSpectrum(signal):
    return np.fft.fft(signal)


def plotPeridogramContinous(signal, sampling_freq):
    fper, Pxx = sig.periodogram(
        signal, sampling_freq, "hamming", 2048, scaling="density"
    )
    plt.semilogy(fper, Pxx)
    plt.xlim(0, 10000)
    plt.xlabel("częstotliwość [Hz]")
    plt.ylabel("widmowa gęstość mocy")
    plt.title("Periodogram sumy trzech sinusów")
    plt.show()


def plotPeridogram(signal, N_samples, sampling_freq):

    freq_linespace = np.fft.fftfreq(N_samples, N_samples / sampling_freq)
    psd = np.abs(np.fft.fft(signal)) ** 2 / N_samples  # power spectral density

    plt.figure(figsize=(8, 12))

    plt.stem(psd)
    plt.plot(freq_linespace[: N_samples // 2], psd[: N_samples // 2])
    plt.xlabel("Frequencies linespace")
    plt.ylabel("PSD values")

    plt.show()


if __name__ == "__main__":

    A1 = 0.1
    f1 = 3000
    A2 = 0.4
    f2 = 4000
    A3 = 0.8
    f3 = 10000

    sampling_freq = 48000
    N_samples = 2048
    # time = N_samples / sampling_freq -> time needed to get 2048 samples

    time_linespace = np.arange(N_samples) / sampling_freq
    signal = getSignal(
        time_linespace,
        A1,
        f1,
        A2,
        f2,
        A3,
        f3,
    )
    plotPeridogram(signal, N_samples, sampling_freq)
