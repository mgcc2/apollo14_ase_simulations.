"""Figure 7: shot 17 at geophones G1-G3, observed and simulated (Runs 10 and 15), 30-60 Hz."""

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


catalogue = read_catalogue()
geophones = ["G1", "G2", "G3"]

# Observed shot 17; gain relative to the largest peak recorded by that geophone over all shots
observed = {}
for shot in catalogue:
    stream = read(str(Path(SEGY_DIR) / f"shot_{shot}_apollo_14.segy"), format="SEGY")
    for g, trace in zip(geophones, stream):
        observed[g, shot] = process_observed(trace.data.astype(float), trace.stats.sampling_rate,
                                             shot in SHORT_SHOTS)
traces = {"obs": []}
for g in geophones:
    largest = max(np.max(np.abs(observed[g, shot][1])) for shot in catalogue)
    time, wave, env = observed[g, 17]
    peak = np.max(np.abs(wave))
    traces["obs"].append((time, wave / peak, env / peak, largest / peak))

# Synthetics at G1-G3; gain relative to the largest peak along the receiver profile R17xx
# at the G1 shot distances
g1_distances = [d[0] for d in catalogue.values() if d[0] > 0]
distance = {}
models = {}
for number in (10, 15):
    run = next(Path(RUNS_DIR).glob(f"A14_Run{number:02d}_*"))
    stations = read_stations(run)
    profile = [name for name, d in stations.items()
               if name.startswith("R17") and any(abs(d - x) <= 0.08 for x in g1_distances)]
    time, u3 = read_synthetics(run, geophones + profile)
    reference = max(np.max(np.abs(process_synthetic(time, u3[name])[1])) for name in profile)
    key = f"run{number}"
    traces[key] = []
    for g in geophones:
        t, wave, env = process_synthetic(time, u3[g])
        peak = np.max(np.abs(wave))
        traces[key].append((t, wave / peak, env / peak, reference / peak))
    distance[key] = np.array([stations[g] for g in geophones])
    models[key] = read_vp(run)

# P-wave travel times [s] in the best two-layer (Run 10) and linear-gradient (Run 15) models
depth, vp = models["run10"]
v0, v1, h = vp[0], vp[-1], depth[np.flatnonzero(np.diff(vp))[0]]
x = distance["run10"]
run10_direct = x / v0
run10_head = x / v1 + 2 * h * np.cos(np.arcsin(v0 / v1)) / v0

depth, vp = models["run15"]
v0, gradient = vp[0], (vp[-1] - vp[0]) / depth[-1]
x = distance["run15"]
run15_direct = x / v0
run15_diving = 2 / gradient * np.arcsinh(gradient * x / (2 * v0))

# Plot
FS = 19
plt.rcParams.update({"font.size": FS, "xtick.labelsize": FS - 1, "ytick.labelsize": FS - 1})
WAVE = "#1B3CF8"
ENVELOPE = "#8A2BE2"
DIRECT = "#009E73"
HEAD = "#4D4D4D"
DIVING = "#D55E00"
WIGGLE = 0.34
rows = [0, 1, 2]   # G1 at the bottom, G3 at the top
labels = [f"{g} | {d:.2f} m" for g, d in zip(geophones, catalogue[17])]

panels = [("obs", "Observation", []),
          ("run10", r"Best two-layer model (2LVM) | $\mathbf{Run\ 10}$",
           [(run10_direct, DIRECT), (run10_head, HEAD)]),
          ("run15", r"Best linear-gradient model (LGVM) | $\mathbf{Run\ 15}$",
           [(run15_direct, DIRECT), (run15_diving, DIVING)])]

fig, axes = plt.subplots(3, 1, figsize=(16.5, 15.5), sharex=True, sharey=True)
fig.subplots_adjust(left=0.18, right=0.97, bottom=0.07, top=0.885, hspace=0.20)

for letter, ax, (key, title, arrivals) in zip("abc", axes, panels):
    for row, (time, wave, env, factor) in zip(rows, traces[key]):
        keep = time <= TEND
        ax.axhline(row, color="0.84", lw=0.5, zorder=1)
        ax.plot(time[keep], row + WIGGLE * wave[keep], color=WAVE, lw=1.0, zorder=3)
        ax.plot(time[keep], row + WIGGLE * env[keep], color=ENVELOPE, lw=1.2, zorder=4)
        ax.text(4.875, row + 1.15 * WIGGLE, f"×{factor:.1f}" if factor < 10 else f"×{factor:.0f}",
                ha="right", va="center", fontsize=FS - 1,
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="black", lw=1.0))

    for times, colour in arrivals:
        lines = ax.vlines(times, np.array(rows) - 1.02 * WIGGLE, np.array(rows) + 1.1 * WIGGLE,
                          color=colour, ls="--", lw=1.5, zorder=12)
        lines.set_path_effects([pe.Stroke(linewidth=2.8, foreground="white", alpha=0.75), pe.Normal()])

    ax.set_title(title, fontsize=FS)
    ax.text(0.006, 0.985, f"({letter})", transform=ax.transAxes, ha="left", va="top",
            fontsize=FS, fontweight="bold")
    ax.set_xlim(0, 5)
    ax.set_ylim(-0.55, 2.65)
    ax.set_yticks(rows)
    ax.set_yticklabels(labels)
    ax.set_axisbelow(True)
    ax.grid(color="0.88", lw=0.6)

axes[2].set_xlabel("Time after shot [s]")

handles = [Line2D([], [], color=WAVE, lw=1.0, label="Waveform"),
           Line2D([], [], color=ENVELOPE, lw=1.2, label="Envelope"),
           Line2D([], [], color=DIRECT, lw=1.7, ls="--", label="P direct"),
           Line2D([], [], color=HEAD, lw=1.7, ls="--", label="P head"),
           Line2D([], [], color=DIVING, lw=1.7, ls="--", label="P diving")]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.575, 0.957), ncol=5,
           fontsize=FS - 1, framealpha=0.95, columnspacing=1.1, handletextpad=0.6)
fig.suptitle("Shot 17 | 30–60 Hz", fontsize=FS + 1, x=0.575, y=0.985)

fig.savefig("Figure7.png", dpi=113, bbox_inches="tight", facecolor="white")
