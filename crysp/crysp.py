import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erfc
from tqdm import tqdm


def read_single_crystal(filename, ticks_column=0):
    df = pd.read_csv(filename)
    ticks = int(df.columns[ticks_column])
    time = df[df.columns[0]][:ticks].values
    events = int(len(df) / ticks)
    ch1 = df.values[:, 1].reshape(events, ticks)
    ch1[ch1 == -99999999] = np.min(ch1[ch1 != -99999999])

    return time, ch1


def time_resolution(
    time,
    left,
    right,
    left_energy,
    right_energy,
    left_threshold,
    right_threshold,
    sigma_left,
    sigma_right,
    signal_window,
):
    left_times = []
    right_times = []
    peak_events = (
        (right_energy > 511 - 2.355 * sigma_right / 2)
        & (right_energy < 511 + 2.355 * sigma_right / 2)
        & (left_energy > 511 - 2.355 * sigma_left / 2)
        & (left_energy < 511 + 2.355 * sigma_left / 2)
    )
    left_peak = left[peak_events]
    right_peak = right[peak_events]
    time_window = (time > signal_window[0]) & (time < signal_window[1])
    timeline = time[time_window]
    for ievt in tqdm(range(left_peak.shape[0])):
        left_time_window = left_peak[ievt, time_window]
        right_time_window = right_peak[ievt, time_window]
        right_times.append(timeline[right_time_window > right_threshold][0])
        left_times.append(timeline[left_time_window > left_threshold][0])

    return np.array(left_times), np.array(right_times)


def energy_spectra(
    left_energy, right_energy, x_range=(100, 800), fit_position=450, text_left=False
):
    fig, ax = plt.subplots(
        1, 2, figsize=(9, 4), constrained_layout=True, sharey=True, sharex=True
    )

    n_left, bins_left, _ = ax[0].hist(
        left_energy,
        bins=100,
        range=x_range,
        histtype="step",
        color="k",
    )
    bin_centers_left = (bins_left[:-1] + bins_left[1:]) / 2
    popt_left_energy, pcov_left_energy = curve_fit(
        gauss,
        bin_centers_left[bin_centers_left > fit_position],
        n_left[bin_centers_left > fit_position],
        p0=(max(n_left), 511, 100),
    )
    ax[0].plot(bin_centers_left, gauss(bin_centers_left, *popt_left_energy), c="k")
    if text_left:
        ax[0].text(
            popt_left_energy[1] * 0.95,
            popt_left_energy[0] * 0.8,
            f"511 keV\n$\Delta E/E={2.355*popt_left_energy[2]/popt_left_energy[1]*100:.2f}$\%%",
            horizontalalignment="right",
        )
    else:
        ax[0].text(
            popt_left_energy[1] * 1.16,
            popt_left_energy[0] * 0.8,
            f"511 keV\n$\Delta E/E={2.355*popt_left_energy[2]/popt_left_energy[1]*100:.2f}$\%%",
        )
    n_right, bins_right, _ = ax[1].hist(
        right_energy,
        bins=100,
        range=x_range,
        histtype="step",
        color="k",
    )
    bin_centers_right = (bins_right[:-1] + bins_right[1:]) / 2
    popt_right_energy, pcov_right_energy = curve_fit(
        gauss,
        bin_centers_right[bin_centers_right > fit_position],
        n_right[bin_centers_right > fit_position],
        p0=(max(n_right), 511, 100),
    )
    ax[1].plot(bin_centers_right, gauss(bin_centers_right, *popt_right_energy), c="k")
    if text_left:
        ax[1].text(
            popt_right_energy[1] * 0.95,
            popt_right_energy[0] * 0.8,
            f"511 keV\n$\Delta E/E={2.355*popt_right_energy[2]/popt_right_energy[1]*100:.2f}$\%%",
            horizontalalignment="right",
        )
    else:
        ax[1].text(
            popt_right_energy[1] * 1.16,
            popt_right_energy[0] * 0.8,
            f"511 keV\n$\Delta E/E={2.355*popt_right_energy[2]/popt_right_energy[1]*100:.2f}$\%%",
        )
    ax[0].set_title("Left crystal")
    ax[1].set_title("Right crystal")
    ax[0].set_xlabel("Energy [keV]")
    ax[1].set_xlabel("Energy [keV]")
    ax[0].set_ylabel(f"N. events / {bins_left[1]-bins_left[0]:.2g} keV")
    ax[0].set_xlim(*x_range)
    fig.suptitle(r"$^{22}$Na coincidence spectrum")

    return (
        fig,
        ax,
        popt_left_energy,
        pcov_left_energy,
        popt_right_energy,
        pcov_right_energy,
    )


def charge_spectra(
    integral_left, integral_right, left_range, right_range, fit_left, fit_right, n_bins=50
):
    fig, ax = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True, sharey=True)
    n_left, bins_left, _ = ax[0].hist(
        integral_left,
        bins=n_bins,
        range=left_range,
        histtype="step",
        color="k",
    )
    bin_centers_left = (bins_left[:-1] + bins_left[1:]) / 2
    popt_left, pcov_left = curve_fit(
        gauss,
        bin_centers_left[bin_centers_left > fit_left],
        n_left[bin_centers_left > fit_left],
        p0=(max(n_left), fit_left * 1.05, fit_left / 10),
    )
    xx = np.linspace(bin_centers_left[0], bin_centers_left[-1], 1000)
    ax[0].plot(xx, gauss(xx, *popt_left), c="k")

    n_right, bins_right, _ = ax[1].hist(
        integral_right, bins=n_bins, range=right_range, histtype="step", color="k"
    )
    bin_centers_right = (bins_right[:-1] + bins_right[1:]) / 2
    popt_right, pcov_right = curve_fit(
        gauss,
        bin_centers_right[bin_centers_right > fit_right],
        n_right[bin_centers_right > fit_right],
        p0=(max(n_right), fit_right * 1.05, fit_right / 10),
    )
    xx = np.linspace(bin_centers_right[0], bin_centers_right[-1], 1000)
    ax[1].plot(xx, gauss(xx, *popt_right), c="k")

    ax[0].set_title("Left crystal")
    ax[1].set_title("Right crystal")
    ax[0].set_xlabel("Charge [V$\cdot$s]")
    ax[1].set_xlabel("Charge [V$\cdot$s]")
    ax[0].set_ylabel(f"N. events / {bins_left[1]-bins_left[0]:.2e} [V$\cdot$s]")
    # ax[1].set_ylabel(f"N. events / {bins_right[1]-bins_right[0]:.2e} [V$\cdot$s]")
    fig.suptitle(r"$^{22}$Na coincidence spectrum - charge")

    return fig, ax, popt_left, pcov_left, popt_right, pcov_right


def read_file(filename):
    df = pd.read_csv(filename)
    ticks = int(df.columns[1])
    time = df.index[:ticks]
    events = int(len(df) / ticks)
    ch1 = df.values[:, 0].reshape(events, ticks)
    ch2 = df.values[:, 1].reshape(events, ticks)
    ch1[ch1 == -99999999] = np.min(ch1[ch1 != -99999999])
    ch2[ch2 == -99999999] = np.min(ch2[ch2 != -99999999])
    ch1[ch1 == 99999999] = np.max(ch1[ch1 != 99999999])
    ch2[ch2 == 99999999] = np.max(ch2[ch2 != 99999999])

    return time, ch1, ch2


def baseline_subtract(time, channel, top=0.5):
    baselines = np.average(channel[:, :150], axis=1)[:, None]
    channel_sub = channel - baselines
    result = channel_sub[
        (np.max(channel_sub, axis=1) < top)
        & (np.min(channel_sub, axis=1) > -0.02)
        & (
            np.max((channel_sub)[:, np.argwhere(time > 0.4e-7)[0][0] :], axis=1)
            < top / 2
        )
    ]
    average_result = np.average(result, axis=0)

    return result, average_result


def gauss(x, A, mu, sigma):
    return A * np.exp(-((x - mu) ** 2) / (2.0 * sigma**2))


def lin_fit(x, a, b):
    return a * x + b


def exp_gauss_pdf(x, mu, l, sigma):
    return (
        l
        / 2
        * np.exp(l / 2 * (2 * mu + l * sigma * sigma - 2 * x))
        * erfc((mu + l * sigma * sigma - x) / (np.sqrt(2) * sigma))
    )


def exp_gauss(x, A, mu, l, sigma):
    return A * exp_gauss_pdf(x, mu, l, sigma)


def two_exp_gauss(x, A, N, mu, l1, l2, sigma):
    return A * (
        N * exp_gauss_pdf(x, mu, l1, sigma) + (1 - N) * exp_gauss_pdf(x, mu, l2, sigma)
    )


def two_gauss(x, N1, N2, mu1, mu2, sigma1, sigma2):
    return N1 * gauss(x, N1, mu1, sigma1) + N2 * gauss(x, N2, mu2, sigma2)



def three_exp_gauss(x, A, N1, N2, mu, l1, l2, l3, sigma):
    return A * (
        N1 * exp_gauss_pdf(x, mu, l1, sigma)
        + (1 - N1) * exp_gauss_pdf(x, mu, l2, sigma)
        + (1 - N2) * exp_gauss_pdf(x, mu, l3, sigma)
    )


def plot_spectra(time, left, right, gain_left, gain_right, start_left, start_right):
    integral_left = np.trapz(left / gain_left, x=time)
    integral_right = np.trapz(right / gain_right, x=time)

    fig, ax = plt.subplots(1, 1)

    n_left, bins, patches = ax.hist(
        integral_left,
        bins=100,
        range=(0, max(start_left, start_right) * 1.5),
        histtype="step",
        label="Left",
        color="k",
        lw=2,
    )
    n_right, bins, patches = ax.hist(
        integral_right,
        bins=100,
        range=(0, max(start_left, start_right) * 1.5),
        histtype="step",
        label="Right",
        color="r",
        lw=2,
    )
    bin_centers = (bins[:-1] + bins[1:]) / 2
    p_511_right, pcov511_right = curve_fit(
        gauss,
        bin_centers[bin_centers > start_right],
        n_right[bin_centers > start_right],
        p0=(200, start_right * 1.1, start_right / 5),
    )
    p_511_left, pcov511_left = curve_fit(
        gauss,
        bin_centers[bin_centers > start_left],
        n_left[bin_centers > start_left],
        p0=(200, start_left * 1.1, start_left / 5),
    )

    xx = np.linspace(bin_centers[0], bin_centers[-1], 1000)
    ax.plot(xx, gauss(xx, *p_511_right), c="r")
    ax.plot(xx, gauss(xx, *p_511_left), c="k")
    ax.text(
        p_511_right[1] * 0.8,
        p_511_right[0] * 1,
        f"{p_511_right[2]/p_511_right[1]*100:.3g}\%\n{p_511_right[1]:.3g}",
        c="r",
    )
    ax.text(
        p_511_left[1] * 0.8,
        p_511_left[0] * 1.1,
        f"{p_511_left[2]/p_511_left[1]*100:.3g}\%\n{p_511_left[1]:.3g}",
        c="k",
    )

    ax.legend()
    ax.set_xlabel("Charge [p.e.]")
    ax.set_ylabel("N. events")
    fig.savefig("na22.png")

    average_left = np.average(
        left[
            (integral_left > p_511_left[1] - abs(p_511_left[2]))
            & (integral_left < p_511_left[1] + abs(p_511_left[2]))
        ],
        axis=0,
    )
    average_right = np.average(
        right[
            (integral_right > p_511_right[1] - abs(p_511_right[2]))
            & (integral_right < p_511_right[1] + abs(p_511_right[2]))
        ],
        axis=0,
    )

    return (
        fig,
        ax,
        average_left,
        average_right,
        p_511_left[1],
        p_511_left[2],
        np.sqrt(np.diag(pcov511_left)),
        p_511_right[1],
        p_511_right[2],
        np.sqrt(np.diag(pcov511_right)),
    )
