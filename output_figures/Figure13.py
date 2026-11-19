"""Figure 13: 10-67 Hz waveforms, spectrograms and envelopes at the three geophones for Shot 17."""

import glob

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from netCDF4 import Dataset, chartostring
from obspy import read
from scipy import signal
from scipy.ndimage import uniform_filter1d

SEGY_DIR = "path/to/segy_data"   # shot_17_apollo_14.segy
RUNS_DIR = "path/to/runs"   # AxiSEM3D run folders A14_Run10_* and A14_Run15_*

BAND = (10.0, 67.0)   # Hz
WINDOW = 0.25   # s, spectrogram window
SMOOTH = 0.05   # s, envelope smoothing
DB_FLOOR = -40


def velocity(u, dt):
    """Displacement to velocity: 4th-order central differences (2nd order at the ends)."""
    v = np.gradient(u, dt, edge_order=2)
    v[2:-2] = (u[:-4] - 8 * u[1:-3] + 8 * u[3:-1] - u[4:]) / (12 * dt)
    return v


def bandpass(x, fs):
    """Zero-phase 4th-order Butterworth band-pass, normalised to a peak of 1."""
    pad = min(len(x) - 1, round(fs))
    sos = signal.butter(4, BAND, btype="bandpass", fs=fs, output="sos")
    y = signal.sosfiltfilt(sos, np.pad(x, pad, mode="reflect"))[pad:-pad]
    return y / np.max(np.abs(y))


def envelope(x, fs):
    """Hilbert envelope smoothed over SMOOTH seconds, normalised to a peak of 1."""
    pad = min(len(x) - 1, round(fs))
    env = np.abs(signal.hilbert(np.pad(x, pad, mode="reflect")))[pad:-pad]
    env = uniform_filter1d(env, size=round(SMOOTH * fs), mode="reflect")
    return env / env.max()


def spectrogram_db(x, time, fs):
    """Velocity PSD in BAND, in dB relative to its maximum."""
    n = round(WINDOW * fs)
    f, t, power = signal.spectrogram(x, fs=fs, window="hann", nperseg=n, noverlap=round(0.8 * n),
                                     nfft=4 * n, detrend="constant", scaling="density", mode="psd")
    keep = (f >= BAND[0]) & (f <= BAND[1])
    power = power[keep] / power[keep].max()
    return t + time[0], f[keep], 10 * np.log10(np.maximum(power, 1e-12))


def read_synthetics(run, names):
    """Vertical displacement (U3) at the named stations from the AxiSEM3D station output."""
    waves = {}
    for path in glob.glob(f"{run}/output/stations/shot_17/axisem3d_synthetics.nc*"):
        with Dataset(path) as ds:
            stations = [s.split(".")[-1] for s in chartostring(ds["list_station"][:])]
            u3 = list(chartostring(ds["list_channel"][:])).index("U3")
            time = np.asarray(ds["data_time"][:])
            for i, name in enumerate(stations):
                if name in names:
                    waves[name] = np.asarray(ds["data_wave"][i, u3, :], dtype=float)
    return time, np.array([waves[name] for name in names])


def read_distances(run, names):
    """Source-receiver distances [m] from the AxiSEM3D station list."""
    rows = [line.split() for line in open(f"{run}/input/shot_17.txt") if not line.startswith("#")]
    distance = {row[0]: float(row[2]) for row in rows if row}
    return [distance[name] for name in names]


GEOPHONES = ["G1", "G2", "G3"]
runs = {n: glob.glob(f"{RUNS_DIR}/A14_Run{n}_*")[0] for n in ("10", "15")}
distances = read_distances(runs["10"], GEOPHONES)

# observed velocity (SEGY traces 1-3 = G1-G3) and synthetic displacement converted to velocity
segy = read(f"{SEGY_DIR}/shot_17_apollo_14.segy", format="SEGY")
traces = {"obs": (segy[0].times(), np.array([tr.data for tr in segy], dtype=float))}
for n, run in runs.items():
    time, u = read_synthetics(run, GEOPHONES)
    fs = 1 / np.median(np.diff(time))
    traces["run" + n] = (time, np.array([velocity(x, 1 / fs) for x in signal.detrend(u)]))

# common time window, then process each trace
start = max(time[0] for time, waves in traces.values())
end = min(time[-1] for time, waves in traces.values())
results = {}
for name, (time, waves) in traces.items():
    keep = (time >= start) & (time <= end)
    time = time[keep]
    fs = 1 / np.median(np.diff(time))
    waves = signal.detrend(waves[:, keep])
    results[name] = dict(time=time,
                         wave=[bandpass(x, fs) for x in waves],
                         envelope=[envelope(bandpass(x, fs), fs) for x in waves],
                         spectrogram=[spectrogram_db(x, time, fs) for x in waves])

# plot
FS = 18.5
FS_SMALL = 17.5
plt.rcParams.update({"font.size": FS, "xtick.labelsize": FS - 1, "ytick.labelsize": FS - 1,
                     "axes.spines.top": False, "axes.spines.right": False})

sets = [("obs", "Observation", "#222222"),
        ("run10", r"2LVM | $\mathbf{Run\ 10}$", "#0072B2"),
        ("run15", r"LGVM | $\mathbf{Run\ 15}$", "#D55E00")]


def style(ax):
    ax.set(xlim=(0, 5), xticks=range(6))
    ax.set_axisbelow(True)
    ax.grid(color="0.88", lw=0.6)


fig = plt.figure(figsize=(16.5, 17.5))
outer = fig.add_gridspec(4, 3, left=0.09, right=0.985, bottom=0.16, top=0.865,
                         height_ratios=[1, 1, 1, 0.65], hspace=0.3, wspace=0.12)

for i, (name, title, colour) in enumerate(sets):
    r = results[name]
    for j in range(3):
        pair = outer[i, j].subgridspec(2, 1, height_ratios=[0.6, 1.0], hspace=0.2)
        wave_ax = fig.add_subplot(pair[0])
        spec_ax = fig.add_subplot(pair[1], sharex=wave_ax)

        wave_ax.plot(r["time"], r["wave"][j], color=colour, lw=0.6)
        wave_ax.set_title(title, fontsize=FS)
        wave_ax.set(ylim=(-1.05, 1.05), yticks=[-1, 0, 1])
        wave_ax.tick_params(axis="x", labelbottom=False, length=0)

        mesh = spec_ax.pcolormesh(*r["spectrogram"][j], cmap="turbo", vmin=DB_FLOOR, vmax=0,
                                  shading="auto", rasterized=True)
        spec_ax.set(ylim=BAND, yticks=[10, 30, 50, 67])
        spec_ax.text(0.02, 0.98, f"({'abcdefghi'[3 * i + j]})", transform=spec_ax.transAxes,
                     ha="left", va="top", fontsize=FS + 1, fontweight="bold",
                     bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))

        for ax in (wave_ax, spec_ax):
            style(ax)
            ax.tick_params(axis="y", labelleft=j == 0)
        if j == 0:
            wave_ax.set_ylabel("Norm.\nvelocity", fontsize=FS_SMALL)
            spec_ax.set_ylabel("Frequency [Hz]", fontsize=FS_SMALL)
        if i == 0:
            fig.text(np.mean(wave_ax.get_position().intervalx), 0.905,
                     f"Geophone {j + 1} | {distances[j]:.2f} m", ha="center", va="center")

for j in range(3):
    ax = fig.add_subplot(outer[3, j])
    for name, title, colour in sets:
        time, env = results[name]["time"], results[name]["envelope"][j]
        if name == "obs":
            ax.fill_between(time, 0, env, color=".3", alpha=0.08, lw=0)
            ax.plot(time, env, color=colour, lw=1.5, zorder=4)
        else:
            ax.plot(time, env, color=colour, lw=1.2, zorder=3)
    ax.set_title("Envelope comparison", fontsize=FS)
    ax.text(0.98, 0.98, f"({'jkl'[j]})", transform=ax.transAxes, ha="right", va="top",
            fontsize=FS + 1, fontweight="bold")
    ax.set(ylim=(-0.02, 1.06), yticks=[0, 0.5, 1])
    style(ax)
    ax.set_xlabel("Time after shot [s]")
    ax.tick_params(axis="y", labelleft=j == 0)
    if j == 0:
        ax.set_ylabel("Normalised envelope", fontsize=FS_SMALL)

handles = [Line2D([], [], color=colour, lw=1.6, label=title) for name, title, colour in sets]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5375, 0.958), ncol=3, fontsize=FS_SMALL)
bar = fig.colorbar(mesh, cax=fig.add_axes([0.18, 0.092, 0.715, 0.015]), orientation="horizontal")
bar.set_label("Velocity PSD relative to each panel maximum [dB]", fontsize=FS_SMALL, labelpad=7)
bar.set_ticks(range(DB_FLOOR, 1, 10))
fig.suptitle("Observed and synthetic frequency content | 10–67 Hz", x=0.5375, y=0.988, fontsize=FS + 1)

fig.savefig("Figure13.png", dpi=106, bbox_inches="tight", facecolor="white")
