#!/usr/bin/env python3
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from obspy import read
from scipy.signal import correlate, correlation_lags, find_peaks


SEGY_DIR = Path("")
OUT_DIR = Path("")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SHOTS = [1, 2, 3, 4, 7, 11, 12, 13, 17, 18, 19, 20, 21]
GEOPHONES = {"G1": 0, "G2": 1, "G3": 2}

PLOT_MAX_LAG = 0.25
TROUGH_MIN = 0.010
TROUGH_MAX = 0.060


def autocorrelation(trace):
    tr = trace.copy()
    tr.data = np.asarray(tr.data, dtype=float)
    tr.detrend("demean")

    ac = correlate(tr.data, tr.data, mode="full", method="fft")
    lags = correlation_lags(len(tr.data), len(tr.data), mode="full") * tr.stats.delta
    ac /= ac[len(tr.data) - 1]

    return lags, ac


def first_negative_trough(lags, ac):
    mask = (lags >= TROUGH_MIN) & (lags <= TROUGH_MAX)
    x, y = lags[mask], ac[mask]

    peaks, _ = find_peaks(-y, prominence=0.03)
    peaks = [p for p in peaks if y[p] < 0]

    return x[peaks[0]] if peaks else np.nan


results = {g: [] for g in GEOPHONES}

for shot in SHOTS:
    st = read(str(SEGY_DIR / f"shot_{shot}_apollo_14.segy"), format="SEGY")

    for geophone, index in GEOPHONES.items():
        if geophone == "G2" and shot in SHOTS[:5]:
            continue

        lags, ac = autocorrelation(st[index])
        trough = first_negative_trough(lags, ac)
        results[geophone].append((lags, ac, trough))


fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)

for ax, geophone in zip(axes, GEOPHONES):
    troughs = []

    for lags, ac, trough in results[geophone]:
        mask = np.abs(lags) <= PLOT_MAX_LAG
        ax.plot(lags[mask], ac[mask], lw=1.0, alpha=0.65)

        if np.isfinite(trough):
            troughs.append(trough)

    mean_ms = np.mean(troughs) * 1000

    ax.axvline(0, color="black", lw=0.8)
    ax.set_title(f"{geophone} — mean trough = {mean_ms:.1f} ms")
    ax.set_xlabel("Lag time (s)")
    ax.set_xlim(-PLOT_MAX_LAG, PLOT_MAX_LAG)
    ax.grid(alpha=0.2)

axes[0].set_ylabel("Normalized autocorrelation")

fig.tight_layout()

output = OUT_DIR / "apollo14_autocorrelation_G1_G2_G3.png"
fig.savefig(output, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {output}")
