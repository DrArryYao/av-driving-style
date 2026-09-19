"""P0-2: null-simulation calibration of the fluctuation-gain estimator.

Synthesizes leader/follower pairs with KNOWN ground-truth gain, passes them
through the exact estimation pipeline (5-s moving-average detrend, std
ratio), and measures bias as a function of disturbance amplitude, window
length and leader measurement noise. Ground truth: follower fluctuation =
G_true x first-order-lag(leader fluctuation) + small process noise.
"""
import numpy as np
import pandas as pd

DT = 0.1


def band_fluct(n, sigma, rng):
    """Band-limited (0.05-0.5 Hz) fluctuation with std sigma."""
    x = rng.normal(0, 1, n)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, DT)
    X[(f < 0.05) | (f > 0.5)] = 0
    y = np.fft.irfft(X, n)
    return y / max(y.std(), 1e-9) * sigma


def lag(v, tau):
    """First-order lag (discrete exponential smoothing)."""
    a = DT / (tau + DT)
    out = np.empty_like(v)
    out[0] = v[0]
    for i in range(1, v.size):
        out[i] = out[i - 1] + a * (v[i] - out[i - 1])
    return out


def detrend(v):
    half = 25  # 5 s at 10 Hz
    ker = np.ones(2 * half + 1) / (2 * half + 1)
    trend = np.convolve(np.pad(v, half, mode="edge"), ker, mode="valid")[: v.size]
    return v - trend


def estimate_gain(lead, foll):
    return detrend(foll).std() / max(detrend(lead).std(), 1e-9)


def main():
    rng = np.random.default_rng(0)
    rows = []
    for sigma_d in [0.05, 0.1, 0.2, 0.3, 0.6, 1.0]:
        for L in [81, 150, 300]:
            for noise in [0.0, 0.05, 0.15]:
                for g_true in [0.5, 0.8, 1.0, 1.2]:
                    ests = []
                    for rep in range(300):
                        n = L
                        lead = band_fluct(n, sigma_d, rng)
                        foll = g_true * lag(lead, 0.3) + rng.normal(0, 0.05, n)
                        lead_m = lead + rng.normal(0, noise, n)
                        foll_m = foll + rng.normal(0, noise, n)
                        ests.append(estimate_gain(lead_m, foll_m))
                    ests = np.array(ests)
                    rows.append(dict(
                        sigma_d=sigma_d, L=L, noise=noise, g_true=g_true,
                        bias_ratio=float(np.median(ests / g_true)),
                        iqr_lo=float(np.percentile(ests / g_true, 25)),
                        iqr_hi=float(np.percentile(ests / g_true, 75))))
    df = pd.DataFrame(rows)
    df.to_csv("p02_gain_bias.csv", index=False)

    print("=== analysis condition: sigma_d >= 0.3, all windows ===")
    sub = df[(df.sigma_d >= 0.3) & (df.noise <= 0.15)]
    print(sub.groupby(["noise", "g_true"]).bias_ratio.agg(["median", "min", "max"]).round(3).to_string())
    print("\n=== low disturbance (why we threshold) ===")
    sub2 = df[(df.sigma_d <= 0.1) & (df.noise == 0.15) & (df.g_true == 1.0)]
    print(sub2.groupby(["sigma_d", "L"]).bias_ratio.mean().round(2).to_string())


if __name__ == "__main__":
    main()
