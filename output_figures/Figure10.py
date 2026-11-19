"""Figure 10: geophone 2 record section, observed and simulated (Runs 10 and 15), 30-60 Hz."""

import csv
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator
from netCDF4 import Dataset, chartostring
from obspy import read
from obspy.signal.filter import bandpass, envelope
from scipy.signal import detrend

SEGY_DIR = "path/to/segy_data"   # shot_<N>_apollo_14.segy files and apollo14_ase_data_catalog.txt
RUNS_DIR = "path/to/runs"   # AxiSEM3D run folders A14_Run<NN>_* (input/ and output/stations/shot_17/)
PICKS_FILE = "path/to/apollo14_previous_p_wave_picks_97.txt"   # published P-wave picks table

SHORT_SHOTS = (12, 19)   # records that stop early
TIPPED_SHOTS = (1, 2, 3, 4, 7)   # shots recorded while G2 was tipped over
TMAX = 5.0   # s, processed record length (gains use all of it)
WINDOW = (0.0, 1.5)   # s, drawn time window


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


def read_kovach_g2_picks():
    """Kovach (1971) picks on G2 as (phase, shot, offset [m], time [s])."""
    picks = []
    for line in Path(PICKS_FILE).read_text(encoding="utf-8-sig").splitlines():
        fields = line.split()
        if len(fields) == 7 and fields[0] == "Kovach_1971" and fields[3] == "G2":
            picks.append((fields[1], int(fields[2]), float(fields[5]), float(fields[6])))
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


def receiver_at(stations, x):
    """Name of the R17xx profile receiver at distance x [m]."""
    return next(n for n, d in stations.items() if n.startswith("R17") and abs(d - x) <= 0.08)


catalogue = read_catalogue()
shot = np.array(list(catalogue))
distance = np.array([d[1] for d in catalogue.values()])   # G2
g1_distances = [d[0] for d in catalogue.values() if d[0] > 0]
tipped = np.isin(shot, TIPPED_SHOTS)

# Observed G2; gain relative to the largest peak over all shots
processed = []
for s in shot:
    trace = read(str(Path(SEGY_DIR) / f"shot_{s}_apollo_14.segy"), format="SEGY")[1]
    processed.append(process_observed(trace.data.astype(float), trace.stats.sampling_rate, s in SHORT_SHOTS))
peaks = np.array([np.max(np.abs(wave)) for _, wave, _ in processed])
traces = {"obs": [(time, wave / p, env / p, peaks.max() / p) for (time, wave, env), p in zip(processed, peaks)]}

# Synthetics at the receivers matching each G2 distance. The gain reference is the largest peak
# along the receiver profile at the G1 distances (as in Figure 9); the receiver at the source is
# drawn at that reference gain and clipped.
models = {}
for number in (10, 15):
    run = next(Path(RUNS_DIR).glob(f"A14_Run{number:02d}_*"))
    stations = read_stations(run)
    names = [receiver_at(stations, x) for x in distance]
    profile = [receiver_at(stations, x) for x in g1_distances]
    time, u3 = read_synthetics(run, names + profile)
    fs = 1 / np.median(np.diff(time))
    reference = max(np.max(np.abs(process_synthetic(time, u3[name])[1])) for name in profile)
    key = f"run{number}"
    traces[key] = []
    for name, d in zip(names, distance):
        t, wave, env = process_synthetic(time, u3[name])
        if d == 0:   # envelope with 0.5 s reflected padding so the source pulse does not wrap around
            n = round(0.5 * fs)
            env = smooth_envelope(np.pad(wave, n, mode="reflect"), fs)[n:-n]
            traces[key].append((t, np.clip(wave / reference, -1, 1), np.clip(env / reference, 0, 1), 1.0))
        else:
            p = np.max(np.abs(wave))
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

# Kovach (1971) picks on G2
pick_phase, pick_shot, pick_offset, pick_time = (np.array(column) for column in zip(*read_kovach_g2_picks()))
head_end = pick_time[pick_phase == "HEAD"].max() + 0.12

# Plot
FS = 18.5
FS_INSET = 15
plt.rcParams.update({"font.size": FS, "xtick.labelsize": FS - 1, "ytick.labelsize": FS - 1})
WAVE = "#546DFA"
ENVELOPE = "#8A2BE2"
DIRECT = "#009E73"
HEAD = "#4D4D4D"
DIVING = "#D55E00"
PICK = "#1D3557"
TIPPED = (0, (6, 3, 1.5, 3))   # envelope style for the tipped-over G2 records
WIGGLE = 2.0
markers = {"DIRECT": "o", "HEAD": "s"}
order = np.argsort(distance, kind="stable")


def add_inset(ax, key, phase, window, box, linestyle):
    """Zoom on the records with a Kovach (1971) pick of this phase, each normalised in the window."""
    t0, t1 = window
    picked = (pick_phase == phase) & (pick_time >= t0) & (pick_time <= t1)
    rows = sorted(np.flatnonzero(np.isin(shot, pick_shot[picked])), key=lambda i: (distance[i], shot[i]))
    y0, y1 = distance[rows].min() - 1.6, distance[rows].max() + 2.6
    ax.add_patch(Rectangle((t0, y0), t1 - t0, y1 - y0, fill=False, lw=1.4, ls=linestyle, zorder=15))

    ins = ax.inset_axes([box[0], box[1] - box[3], box[2], box[3]])
    for row, i in enumerate(rows):
        time, wave, env, _ = traces[key][i]
        keep = (time >= t0) & (time <= t1)
        peak = np.max(np.abs(wave[keep]))
        ins.plot(time[keep], row + 0.42 * wave[keep] / peak, color=WAVE, lw=0.7, zorder=5)
        ins.plot(time[keep], row + 0.42 * env[keep] / peak, color=ENVELOPE, lw=1.2,
                 ls=TIPPED if key == "obs" and tipped[i] else "-", zorder=6)
        ins.text(t1 + 0.025 * (t1 - t0), row, f"S{shot[i]}", va="center", fontsize=FS_INSET)
        for t in pick_time[picked & (pick_shot == shot[i])]:
            ins.scatter(t, row, marker=markers[phase], s=27, facecolors=PICK,
                        edgecolors="black", linewidths=0.6, zorder=10)
    ins.set(xlim=window, ylim=(-0.6, len(rows) - 0.4))
    ins.set_yticks(range(len(rows)))
    ins.set_yticklabels([f"{distance[i]:.0f}" for i in rows])
    ins.set_ylabel("Distance [m]", fontsize=FS_INSET, labelpad=2)
    ins.xaxis.set_major_locator(MultipleLocator(0.1))
    ins.tick_params(labelsize=FS_INSET, length=2.5, width=0.7, pad=1.5)
    ins.set_axisbelow(True)
    ins.grid(color="0.88", lw=0.6)
    for spine in ins.spines.values():
        spine.set(linewidth=1.4, linestyle=linestyle)


panels = [("obs", "Observation", []),
          ("run10", r"2LVM | $\mathbf{Run\ 10}$", [(run10_direct, DIRECT), (run10_head, HEAD)]),
          ("run15", r"LGVM | $\mathbf{Run\ 15}$", [(run15_direct, DIRECT), (run15_diving, DIVING)])]

fig, axes = plt.subplots(1, 3, figsize=(16.5, 13.0), sharey=True)
fig.subplots_adjust(left=0.075, right=0.985, bottom=0.07, top=0.86, wspace=0.08)

for letter, ax, (key, title, arrivals) in zip("abc", axes, panels):
    for i in order:
        time, wave, env, _ = traces[key][i]
        keep = (time >= WINDOW[0]) & (time <= WINDOW[1])
        ax.axhline(distance[i], color="0.88", lw=0.6, zorder=1)
        ax.plot(time[keep], distance[i] + WIGGLE * wave[keep], color=WAVE, lw=0.5, zorder=8)
        ax.plot(time[keep], distance[i] + WIGGLE * env[keep], color=ENVELOPE, lw=1.2,
                ls=TIPPED if key == "obs" and tipped[i] else "-", zorder=9)

    factor = np.array([f for *_, f in traces[key]])
    for d in np.unique(distance):   # one gain label per distance (the larger one)
        f = factor[distance == d].max()
        label = f"×{f:.1f}" if f < 10 else f"×{f:.0f}"
        if key != "obs" and d == 0:
            label += " (clipped)"
        ax.text(1.48, d + 1.15 * WIGGLE, label, ha="right", va="center", fontsize=FS - 1,
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="black", lw=1.0), zorder=25)

    for phase, marker in markers.items():
        ax.scatter(pick_time[pick_phase == phase], pick_offset[pick_phase == phase], marker=marker,
                   s=34, facecolors=PICK, edgecolors="black", linewidths=0.6, zorder=30)

    for times, colour in arrivals:
        line, = ax.plot(times, x, color=colour, ls="--", lw=1.5, zorder=12)
        line.set_path_effects([pe.Stroke(linewidth=2.8, foreground="white", alpha=0.75), pe.Normal()])

    add_inset(ax, key, "HEAD", (0.05, head_end), (0.18, 0.970, 0.67, 0.19), "-")
    add_inset(ax, key, "DIRECT", (0.22, 0.58), (0.15, 0.750, 0.70, 0.20), (0, (5, 3)))

    ax.set_title(title, fontsize=FS, pad=12)
    ax.text(0.018, 0.98, f"({letter})", transform=ax.transAxes, ha="left", va="top",
            fontsize=FS, fontweight="bold",
            bbox=dict(boxstyle="square,pad=0.10", fc="white", ec="none", alpha=0.95), zorder=40)
    ax.set(xlim=WINDOW, ylim=(-3, 95), xlabel="Time after shot [s]")
    ax.xaxis.set_major_locator(MultipleLocator(0.3))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.set_axisbelow(True)
    ax.grid(color="0.88", lw=0.6)

axes[0].set_ylabel("Source–receiver distance [m]")

handles = [Line2D([], [], color=WAVE, lw=0.5, label="Waveform"),
           Line2D([], [], color=ENVELOPE, lw=1.2, label="Envelope"),
           Line2D([], [], color=ENVELOPE, lw=1.2, ls=TIPPED, label="Tipped-over G2"),
           Line2D([], [], color=DIRECT, lw=1.5, ls="--", label="P direct"),
           Line2D([], [], color=HEAD, lw=1.5, ls="--", label="P head"),
           Line2D([], [], color=DIVING, lw=1.5, ls="--", label="P diving")]
handles += [Line2D([], [], ls="none", marker=marker, markersize=6, markerfacecolor=PICK,
                   markeredgecolor="black", markeredgewidth=0.6, label=f"Kovach (1971) {phase.lower()}")
            for phase, marker in markers.items()]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, 0.974), ncol=4,
           fontsize=FS - 1, framealpha=1.0, columnspacing=1.0, handletextpad=0.6)
fig.suptitle("Geophone 2 | 30–60 Hz", x=0.53, y=0.99, fontsize=FS + 1)

fig.savefig("Figure10.png", dpi=105.4, bbox_inches="tight", facecolor="white")
