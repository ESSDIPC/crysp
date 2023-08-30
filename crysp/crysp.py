import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erfc


def read_single_crystal(filename):
    df = pd.read_csv(filename)
    ticks = int(df.columns[0])
    time = df.index[:ticks]
    events = int(len(df) / ticks)
    ch1 = df.values[:, 0].reshape(events, ticks)
    ch1[ch1 == -99999999] = np.min(ch1[ch1 != -99999999])

    return time, ch1


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
        c="k",
    )
    ax.text(
        p_511_left[1] * 0.8,
        p_511_left[0] * 1.1,
        f"{p_511_left[2]/p_511_left[1]*100:.3g}\%\n{p_511_left[1]:.3g}",
        c="r",
    )

    ax.legend(frameon=False, fontsize="large")
    ax.set_xlabel("Charge [V$\cdot$s]", fontsize="xx-large")
    ax.set_ylabel("N. events", fontsize="xx-large")
    ax.tick_params(axis="both", which="major", labelsize="x-large")
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
