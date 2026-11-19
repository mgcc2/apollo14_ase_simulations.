"""Figure 9: geophone 1 record section, observed and simulated (Runs 10 and 15), 30-60 Hz."""

import csv
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from netCDF4 import Dataset, chartostring
from obspy import read
from obspy.signal.filter import bandpass, envelope
from scipy.signal import detrend

SEGY_DIR = "path/to/segy_data"   # shot_<N>_apollo_14.segy files and apollo14_ase_data_catalog.txt
RUNS_DIR = "path/to/runs"   # AxiSEM3D run folders A14_Run<NN>_* (input/ and output/stations/shot_17/)
PICKS_FILE = "path/to/apollo14_previous_p_wave_picks_97.txt"   # published P-wave picks table

SHORT_SHOTS = (12, 19)   # records that stop early
TMAX = 5.0   # s, processed record length
TEND = 4.95   # s, end of the drawn traces


def read_catalogue():
    """Source-receiver distances [m] to G1, G2, G3 for each shot."""
    with open(Path(SEGY_DIR) / "apollo14_ase_data_catalog.txt", encoding="utf-8-sig") as f:
        rows = csv.DictReader(f, delimiter="\t")
        return {int(r["ASE_data"].split()[-1]): [float(r[f"Distance_G{g}_m"]) for g in (1, 2, 3)]
                for r in rows}


def read_stations(run):
    """Source-receiver distance [m] of every station in the run's shot-17 station file."""
    distances = {}
    for line in (run / "input/shot_17.txt").read_text().splitlines():
        fields = line.split("#")[0].split()
        if len(fields) >= 6:
            distances[fields[0]] = float(fields[2])
    return distances


def read_synthetics(run, names):
    """Time axis and vertical displacement (U3) of the named stations."""
    u3 = {}
    for path in (run / "output/stations/shot_17").glob("axisem3d_synthetics.nc.rank*"):
        with Dataset(path) as ds:
            ds.set_auto_chartostring(False)
            stations = [s.split(".")[-1] for s in chartostring(ds["list_station"][:].filled(b""))]
            channel = list(chartostring(ds["list_channel"][:])).index("U3")
            time = np.asarray(ds["data_time"][:])
            for i, name in enumerate(stations):
                if name in names:
                    u3[name] = np.asarray(ds["data_wave"][i, channel, :], dtype=float)
    return time, u3


def read_vp(run):
    """Depth [m] and P-wave velocity [m/s] from the run's .bm model file."""
    rows = [line.split() for line in next((run / "input").glob("*.bm")).read_text().splitlines()]
    table = 1000 * np.array([r[:2] for r in rows if r and r[0][0].isdigit()], dtype=float)
    return table[:, 0], table[:, 1]


def read_picks():
    """Published picks as (source, phase, offset [m], time [s])."""
    picks = []
    for line in Path(PICKS_FILE).read_text(encoding="utf-8-sig").splitlines():
        fields = line.split()
        if len(fields) == 7 and fields[0] in authors:
            picks.append((fields[0], fields[1], float(fields[5]), float(fields[6])))
    return picks


def filter_trace(x, fs):
    """Remove the linear trend, then 30-60 Hz zero-phase band-pass with 0.25 s reflected padding."""
    i = np.arange(len(x))
    x = x - np.polyval(np.polyfit(i, x, 1), i)
    x = x - x.mean()
    n = min(round(0.25 * fs), len(x) - 1)
    return bandpass(np.pad(x, n, mode="reflect"), 30, 60, fs, corners=4, zerophase=True)[n:n + len(x)]


def smooth_envelope(x, fs):
    """Envelope smoothed with a 0.05 s moving average."""
    n = round(0.05 * fs)
    return np.convolve(np.pad(envelope(x), (n // 2, n - 1 - n // 2), mode="edge"), np.ones(n) / n, mode="valid")


def process_observed(raw, fs, short):
    """Cut trailing zeros and anything after 5 s, filter; short records also lose their last 0.1 s."""
    stop = min(np.flatnonzero(raw)[-1] + 1, int(TMAX * fs) + 1)
    wave = filter_trace(raw[:stop], fs)
    env = smooth_envelope(wave, fs)
    time = np.arange(len(wave)) / fs
    if short:
        keep = time <= time[-1] - 0.1
        time, wave, env = time[keep], wave[keep], env[keep]
    return time, wave, env


def process_synthetic(time, u):
    """Displacement to velocity (fourth-order differences), then filter the first 5 s."""
    dt = np.median(np.diff(time))
    u = detrend(u)
    v = np.gradient(u, dt, edge_order=2)
    v[2:-2] = (u[:-4] - 8 * u[1:-3] + 8 * u[3:-1] - u[4:]) / (12 * dt)
    keep = (time >= 0) & (time <= TMAX)
    wave = filter_trace(v[keep], 1 / dt)
    return time[keep], wave, smooth_envelope(wave, 1 / dt)


authors = {"Kovach_1971": ("Kovach (1971)", "#1D3557", 0.1),   # label, colour, drawing order
           "Villanova_2025": ("Villanova (2025)", "#E76F51", 0.2),
           "Martella_2022": ("Martella (2022)", "#2A9D8F", 0.3),
           "Wang_2025": ("Wang (2025)", "#E9C46A", 0.4)}
phases = {"DIRECT": ("o", "Direct"), "HEAD": ("s", "Head"), "UNKNOWN": ("D", "Not given")}

catalogue = read_catalogue()
shot = np.array(list(catalogue))
distance = np.array([d[0] for d in catalogue.values()])   # G1

# Observed G1; gain relative to the largest peak over all shots
processed = []
for s in shot:
    trace = read(str(Path(SEGY_DIR) / f"shot_{s}_apollo_14.segy"), format="SEGY")[0]
    processed.append(process_observed(trace.data.astype(float), trace.stats.sampling_rate, s in SHORT_SHOTS))
peaks = np.array([np.max(np.abs(wave)) for _, wave, _ in processed])
traces = {"obs": [(time, wave / p, env / p, peaks.max() / p) for (time, wave, env), p in zip(processed, peaks)]}

# Synthetics at the receivers matching each shot distance; gain relative to the largest peak
# away from the source. The receiver at the source is drawn at that reference gain and clipped.
models = {}
for number in (10, 15):
    run = next(Path(RUNS_DIR).glob(f"A14_Run{number:02d}_*"))
    stations = read_stations(run)
    names = [next(n for n, d in stations.items() if n.startswith("R17") and abs(d - x) <= 0.08)
             for x in distance]
    time, u3 = read_synthetics(run, names)
    fs = 1 / np.median(np.diff(time))
    processed = [process_synthetic(time, u3[name]) for name in names]
    peaks = np.array([np.max(np.abs(wave)) for _, wave, _ in processed])
    reference = peaks[distance > 0].max()
    key = f"run{number}"
    traces[key] = []
    for (t, wave, env), p, d in zip(processed, peaks, distance):
        if d == 0:   # envelope with 0.5 s reflected padding so the source pulse does not wrap around
            n = round(0.5 * fs)
            env = smooth_envelope(np.pad(wave, n, mode="reflect"), fs)[n:-n]
            traces[key].append((t, np.clip(wave / reference, -1, 1), np.clip(env / reference, 0, 1), 1.0))
        else:
            traces[key].append((t, wave / p, env / p, reference / p))
    models[key] = read_vp(run)
    x_max = max(stations[name] for name in names)

# P-wave travel times [s] in the best two-layer (Run 10) and linear-gradient (Run 15) models
x = np.linspace(0, x_max, 1000)
depth, vp = models["run10"]
v0, v1, h = vp[0], vp[-1], depth[np.flatnonzero(np.diff(vp))[0]]
run10_direct = x / v0
run10_head = x / v1 + 2 * h * np.cos(np.arcsin(v0 / v1)) / v0
run10_head[x < 2 * h * np.tan(np.arcsin(v0 / v1))] = np.nan   # no head wave before the critical distance

depth, vp = models["run15"]
v0, gradient = vp[0], (vp[-1] - vp[0]) / depth[-1]
run15_direct = x / v0
run15_diving = 2 / gradient * np.arcsinh(gradient * x / (2 * v0))
turning_depth = (np.sqrt(v0 ** 2 + (gradient * x / 2) ** 2) - v0) / gradient
run15_diving[turning_depth > depth[-1]] = np.nan

# Plot
FS = 18.5
plt.rcParams.update({"font.size": FS, "xtick.labelsize": FS - 1, "ytick.labelsize": FS - 1})
WAVE = "#546DFA"
ENVELOPE = "#8A2BE2"
DIRECT = "#009E73"
HEAD = "#4D4D4D"
DIVING = "#D55E00"
WIGGLE = 2.0

panels = [("obs", "Observation", []),
          ("run10", r"2LVM | $\mathbf{Run\ 10}$", [(run10_direct, DIRECT), (run10_head, HEAD)]),
          ("run15", r"LGVM | $\mathbf{Run\ 15}$", [(run15_direct, DIRECT), (run15_diving, DIVING)])]

fig, axes = plt.subplots(1, 3, figsize=(16.5, 13.0), sharex=True, sharey=True)
fig.subplots_adjust(left=0.155, right=0.985, bottom=0.07, top=0.875, wspace=0.08)

for letter, ax, (key, title, arrivals) in zip("abc", axes, panels):
    for (time, wave, env, factor), d in zip(traces[key], distance):
        keep = time <= TEND
        label = f"×{factor:.1f}" if factor < 10 else f"×{factor:.0f}"
        if key != "obs" and d == 0:
            label += " (clipped)"
        ax.axhline(d, color="0.88", lw=0.6, zorder=1)
        ax.plot(time[keep], d + WIGGLE * wave[keep], color=WAVE, lw=0.5, zorder=3)
        ax.plot(time[keep], d + WIGGLE * env[keep], color=ENVELOPE, lw=1.2, zorder=4)
        ax.text(4.925, d + 1.15 * WIGGLE, label, ha="right", va="center", fontsize=FS - 1,
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="black", lw=1.0), zorder=25)

    for times, colour in arrivals:
        line, = ax.plot(times, x, color=colour, ls="--", lw=1.5, zorder=12)
        line.set_path_effects([pe.Stroke(linewidth=2.8, foreground="white", alpha=0.75), pe.Normal()])

    ax.set_title(title, fontsize=FS, pad=12)
    ax.text(0.018, 0.99, f"({letter})", transform=ax.transAxes, ha="left", va="top",
            fontsize=FS, fontweight="bold",
            bbox=dict(boxstyle="square,pad=0.10", fc="white", ec="none", alpha=0.95), zorder=40)
    ax.set(xlim=(0, 5), ylim=(-3, 96), xlabel="Time after shot [s]")
    ax.set_axisbelow(True)
    ax.grid(color="0.88", lw=0.6)

axes[0].set_yticks(distance)
axes[0].set_yticklabels([f"S{s} | {d:.2f}" for s, d in zip(shot, distance)])
axes[0].set_ylabel("Source–receiver distance [m]")

# all published first-arrival picks, at their published offsets
for source, phase, offset, t in read_picks():
    axes[0].scatter(t, offset, marker=phases[phase][0], s=34, facecolors=authors[source][1],
                    edgecolors="black", linewidths=0.6, zorder=30 + authors[source][2])

handles = [Line2D([], [], marker="o", ls="none", markersize=6, markerfacecolor=colour,
                  markeredgecolor="black", markeredgewidth=0.6, label=label)
           for label, colour, _ in authors.values()]
handles += [Line2D([], [], marker=marker, ls="none", markersize=6, markerfacecolor="0.7",
                   markeredgecolor="black", markeredgewidth=0.6, label=label)
            for marker, label in phases.values()]
axes[0].legend(handles=handles, ncol=2, loc="center right", bbox_to_anchor=(0.995, 59 / 99),
               fontsize=15.5, columnspacing=0.65, handlelength=0.8, handletextpad=0.35,
               borderpad=0.35, framealpha=1.0)

handles = [Line2D([], [], color=WAVE, lw=0.5, label="Waveform"),
           Line2D([], [], color=ENVELOPE, lw=1.2, label="Envelope"),
           Line2D([], [], color=DIRECT, ls="--", lw=1.5, label="P direct"),
           Line2D([], [], color=HEAD, ls="--", lw=1.5, label="P head"),
           Line2D([], [], color=DIVING, ls="--", lw=1.5, label="P diving")]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.57, 0.962), ncol=5,
           fontsize=FS - 1, framealpha=1.0, columnspacing=1.2, handletextpad=0.6)
fig.suptitle("Geophone 1 | 30–60 Hz", x=0.57, y=0.98, fontsize=FS + 1)

fig.savefig("Figure9.png", dpi=107.7, bbox_inches="tight", facecolor="white")
