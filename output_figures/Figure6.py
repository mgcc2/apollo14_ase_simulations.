"""Figure 6: observed and synthetic shot 17 waveforms (30-67 Hz) with smoothed envelopes, for all runs."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset, chartostring
from obspy import read
from obspy.signal.filter import bandpass, envelope
from obspy.signal.util import smooth
from scipy.signal import detrend

SEGY_DIR = "path/to/segy_data"   # shot_<N>_apollo_14.segy files
RUNS_DIR = "path/to/runs"   # AxiSEM3D run folders A14_Run<NN>_*

F1, F2 = 30.0, 67.0   # band-pass corners [Hz]
TMAX = 5.0   # end of the time window [s]
GEOPHONES = ["G1", "G2", "G3"]


def read_synthetics(run):
    """Time and vertical displacement (U3) at G1-G3 from the AxiSEM3D station output (all rank files)."""
    traces = {}
    for path in sorted(Path(RUNS_DIR).glob(f"A14_Run{run}_*/output/stations/shot_17/axisem3d_synthetics.nc*")):
        with Dataset(path) as ds:
            time = ds["data_time"][:].data
            stations = [s.split(".")[-1] for s in chartostring(ds["list_station"][:])]
            u3 = list(chartostring(ds["list_channel"][:])).index("U3")
            for i, name in enumerate(stations):
                if name in GEOPHONES and name not in traces:
                    traces[name] = ds["data_wave"][i, u3, :].data.astype(float)
    return time, [traces[g] for g in GEOPHONES]


def process(time, x, to_velocity=False):
    """Detrend, (differentiate), band-pass, envelope, cut to 0-5 s, normalise, smooth envelope."""
    dt = np.median(np.diff(time))
    x = detrend(np.asarray(x, dtype=float))
    if to_velocity:   # synthetics are displacement; the geophones record velocity
        x = np.gradient(x, dt, edge_order=2)
    pad = int(np.ceil(1.0 / dt))   # 1 s of mirrored signal at each end while filtering
    x = bandpass(np.pad(x, pad, mode="reflect"), F1, F2, 1 / dt, corners=4, zerophase=True)[pad:-pad]
    env = envelope(x)
    keep = (time >= 0) & (time <= TMAX)
    time, x, env = time[keep], x[keep], env[keep]
    env = smooth(env / env.max(), int(3 / (0.5 * (F1 + F2)) / dt))   # 3 periods at the mid frequency
    return time, x / np.abs(x).max(), env


# run, colour, run label, model | topography, scattering (as in Table 1; † = applied to each layer)
rows = [
    (None, "#000000", "Observation", "", ""),
    ("15", "#426C9B", "Run 15", "LGVM  |  Topography", "PLS ±50%"),
    ("10", "#1E4538", "Run 10", "2LVM  |  Topography", "LS ±50%†"),
    ("09", "#73A848", "Run 09", "2LVM  |  Topography", "PLS ±50%"),
    ("13", "#6EB8C6", "Run 13", "LGVM  |  Topography", "LS ±50%"),
    ("07", "#2E6C4B", "Run 07", "2LVM  |  Topography", "LS ±50%"),
    ("11", "#3C867F", "Run 11", "2LVM  |  Topography", "PLS ±50%†"),
    ("14", "#55937F", "Run 14", "LGVM  |  Topography", "PLS ±25%"),
    ("06", "#BCBD45", "Run 06", "2LVM  |  Topography", "LS ±25%"),
    ("12", "#2D2C4B", "Run 12", "LGVM  |  Topography", "LS ±25%"),
    ("08", "#327E17", "Run 08", "2LVM  |  Topography", "PLS ±25%"),
    ("05", "#5DADC5", "Run 05", "2LVM  |  Topography", "No scattering"),
    ("04", "#70B7EF", "Run 04", "2LVM  |  No topography", "No scattering"),
    ("03", "#4C78FF", "Run 03", "HVM  |  Topography", "No scattering"),
    ("02", "#49657E", "Run 02", "HVM  |  No topography", "No scattering"),
    ("01", "#000080", "Run 01", "HVM  |  No topography", "No scattering"),
]
best = ("10", "15")
titles = ["Geophone 1 | 18.29 m", "Geophone 2 | 27.43 m", "Geophone 3 | 73.15 m"]
spacing = 1.5

FS = 19
plt.rcParams.update({"font.size": FS, "xtick.labelsize": FS - 1})
fig, axes = plt.subplots(1, 3, figsize=(15.5, 0.95 * len(rows) + 1.4), sharey=True)
fig.subplots_adjust(left=0.01, right=0.995, bottom=0.045, top=0.955, wspace=0.05)

observed = read(str(Path(SEGY_DIR) / "shot_17_apollo_14.segy"), format="SEGY")

for i, (run, colour, *labels) in enumerate(rows):
    if run is None:   # observed: traces 1-3 of the SEGY file are geophones 1-3
        traces = [process(tr.times(), tr.data) for tr in observed[:3]]
    else:
        time, waves = read_synthetics(run)
        traces = [process(time, w, to_velocity=True) for w in waves]

    y = -spacing * i
    bold = run in best
    for k, (ax, (time, wave, env)) in enumerate(zip(axes, traces)):
        ax.plot(time, wave + y, color=colour, alpha=0.75, lw=0.9)
        ax.plot(time, env + y, color="#705086", lw=1.6)
        if labels[k]:   # labels on the right in (a) and (b), on the left in (c)
            ax.text(0.97 if k < 2 else 0.03, y + 0.6, labels[k], transform=ax.get_yaxis_transform(),
                    ha="right" if k < 2 else "left", va="bottom", fontsize=FS - 1,
                    fontweight="bold" if bold else "normal", zorder=20,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=colour, lw=2.0 if bold else 1.4))

for letter, ax, title in zip("abc", axes, titles):
    ax.set_title(title, fontsize=FS)
    ax.text(0.02, 0.98, f"({letter})", transform=ax.transAxes, ha="left", va="top",
            fontsize=FS + 1, fontweight="bold")
    ax.set_xlim(0, TMAX)
    ax.set_ylim(-spacing * (len(rows) - 1) - 1.1, 2.0)
    ax.set_xticks(np.arange(0, 6))
    ax.set_yticks([])
    ax.set_xlabel("Time after shot [s]")

fig.savefig("Figure6.png", dpi=109, bbox_inches="tight", facecolor="white")
