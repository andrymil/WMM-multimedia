import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter, clock_gettime


def getSampledSignal(period, sample_numb):
    # time_linespace = np.linspace(0, t0, N)
    time_linespace = np.linspace(0, period, sample_numb, endpoint=False)
    signal_wave = np.cos(np.pi * time_linespace)
    return signal_wave, time_linespace


def getFreqSpectrum(signal_wave):
    return np.fft.fft(signal_wave)


def getSpectrums(signal_wave):
    freq_spectrum = getFreqSpectrum(signal_wave)  # frequency
    magnitude_spectrum = np.abs(freq_spectrum)  # amplitude
    freq_spectrum[np.abs(freq_spectrum) < 1e-10] = 0  # remove small values
    phase_spectrum = np.angle(freq_spectrum)  # phase
    print(phase_spectrum)
    return freq_spectrum, magnitude_spectrum, phase_spectrum


def plotDiagrams(period, N_samples):
    plotSignal(period, N_samples)
    plotSpectrums(period, N_samples)
    plt.tight_layout()
    plt.show()


def plotSignal(period, N_samples):
    signal, time_linespace = getSampledSignal(period, N_samples)
    plt.figure(figsize=(8, 12))
    plt.subplot(3, 1, 1)
    plt.stem(time_linespace, signal)
    plt.plot(time_linespace, signal)


def plotSpectrums(period, N_samples):
    signal, time_linespace = getSampledSignal(period, N_samples)
    freq_spectrum, magnitude_spectrum, phase_spectrum = getSpectrums(signal)
    freq_linespace = np.fft.fftfreq(
        N_samples,
        2 / N_samples,
    )  # period / N_samples is sampling freq (for cos(pi*t) period is T=2)

    plt.subplot(3, 1, 2)
    plt.stem(freq_linespace, magnitude_spectrum)

    plt.subplot(3, 1, 3)
    plt.stem(freq_linespace, phase_spectrum)


def calc_fft_computation_period(period, max_sampling_exponent):
    samples = [2**n for n in range(1, max_sampling_exponent + 1)]
    data = {}
    # data = []
    for sample in samples:
        temp_times = []
        signal, _ = getSampledSignal(period, sample)
        for _ in range(100):
            start = perf_counter()
            getFreqSpectrum(signal)
            stop = perf_counter()
            time_stamp = stop - start
            temp_times.append(time_stamp)
        data[sample] = np.mean(temp_times)
        # data.append(np.mean(temp_times))

    # print(data.values())
    plt.plot(data.keys(), data.values())
    plt.xlim(0, 2**max_sampling_exponent)
    # plt.plot(samples, data)
    plt.show()


if __name__ == "__main__":

    # a)
    N_samples = 8
    period = 2  # period of one full cycle of cos(pi*t) is t = 2

    plotDiagrams(period, N_samples)

    # b)
    period = 2
    exponent = 10
    calc_fft_computation_period(period, exponent)
