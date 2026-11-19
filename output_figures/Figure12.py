"""Figure 12: observed and synthetic 30-60 Hz envelopes and peak amplitudes against distance."""

import csv
import glob

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from netCDF4 import Dataset, chartostring
from obspy import read
from obspy.signal.filter import bandpass, envelope
from scipy import signal
from scipy.ndimage import uniform_filter1d

SEGY_DIR = "path/to/segy_data"   # shot_<N>_apollo_14.segy files and apollo14_ase_data_catalog.txt
RUNS_DIR = "path/to/runs"   # AxiSEM3D run folders A14_Run10_* and A14_Run15_*

BAND = (30.0, 60.0)   # Hz
TMAX = 5.0   # s, processed record length
PAD = 0.25   # s, reflected padding before filtering
SMOOTH = 0.05   # s, envelope smoothing
CUT_END = 0.10   # s, cut from the end of short records (filter edge)
TRACE_END = 4.95   # s, last time drawn
GAIN = 19.8   # envelope height on the distance axis [m]


def filter_trace(x, fs):
    """Remove the linear trend and mean, pad by reflection, band-pass and cut the padding off."""
    i = np.arange(len(x))
    x = x - np.polyval(np.polyfit(i, x, 1), i)
    x = x - x.mean()
    n = min(round(PAD * fs), len(x) - 1)
    return bandpass(np.pad(x, n, mode="reflect"), *BAND, df=fs, corners=4, zerophase=True)[n:n + len(x)]


def smooth_envelope(x, fs):
    """Hilbert envelope, smoothed with a SMOOTH-second moving average."""
    return uniform_filter1d(envelope(x), size=round(SMOOTH * fs), mode="nearest")


def velocity(u, dt):
    """Displacement to velocity: 4th-order central differences (2nd order at the ends)."""
    v = np.gradient(u, dt, edge_order=2)
    v[2:-2] = (u[:-4] - 8 * u[1:-3] + 8 * u[3:-1] - u[4:]) / (12 * dt)
    return v


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
    return time, [waves[name] for name in names]


# observed: geophone G1 (first SEGY trace) for every shot, excluding the shot fired at G1
with open(f"{SEGY_DIR}/apollo14_ase_data_catalog.txt", encoding="utf-8-sig") as f:
    catalogue = list(csv.DictReader(f, delimiter="\t"))
observed = []
for row in catalogue:
    shot = int(row["ASE_data"].split()[-1])
    distance = float(row["Distance_G1_m"])
    if distance == 0:
        continue
    trace = read(f"{SEGY_DIR}/shot_{shot}_apollo_14.segy", format="SEGY")[0]
    fs = trace.stats.sampling_rate
    raw = trace.data.astype(float)
    end = min(len(raw), np.flatnonzero(raw)[-1] + 1, int(TMAX * fs) + 1)
    wave = filter_trace(raw[:end], fs)
    env = smooth_envelope(wave, fs)
    time = np.arange(len(wave)) / fs
    if end < len(raw):   # short, zero-filled record: hide the filter edge at its end
        keep = time <= time[-1] - CUT_END
        time, wave, env = time[keep], wave[keep], env[keep]
    observed.append(dict(shot=shot, distance=distance, time=time, envelope=env,
                         peak=np.max(np.abs(wave))))
observed.sort(key=lambda r: r["distance"])
results = {"obs": observed}

# synthetics: profile receiver R17<NN> at the distance of each observed record
for n in ("10", "15"):
    run = glob.glob(f"{RUNS_DIR}/A14_Run{n}_*")[0]
    rows = [line.split() for line in open(f"{run}/input/shot_17.txt") if not line.startswith("#")]
    receivers = {row[0]: float(row[2]) for row in rows if row and row[0].startswith("R17")}
    names = [min(receivers, key=lambda r: abs(receivers[r] - obs["distance"])) for obs in observed]
    time, waves = read_synthetics(run, names)
    fs = 1 / np.median(np.diff(time))
    keep = (time >= 0) & (time <= TMAX)
    results["run" + n] = []
    for name, u in zip(names, waves):
        wave = filter_trace(velocity(signal.detrend(u), 1 / fs)[keep], fs)
        results["run" + n].append(dict(distance=receivers[name], time=time[keep],
                                       envelope=smooth_envelope(wave, fs), peak=np.max(np.abs(wave))))

# amplitudes relative to the largest trace of each set
for records in results.values():
    largest = max(r["peak"] for r in records)
    for r in records:
        r["relative_peak"] = r["peak"] / largest
        r["relative_envelope"] = r["envelope"] / largest


def peak_time_fit(records):
    """Straight line through the envelope-peak times against distance."""
    distance = np.array([r["distance"] for r in records])
    peak_time = [r["time"][np.argmax(r["envelope"])] for r in records]
    y = np.linspace(distance.min(), distance.max(), 200)
    return np.polyval(np.polyfit(distance, peak_time, 1), y), y


# plot
FS = 18.5
plt.rcParams.update({"font.size": FS, "xtick.labelsize": FS - 1, "ytick.labelsize": FS - 1,
                     "axes.spines.top": False, "axes.spines.right": False})

sets = [("obs", "Observed G1", "#222222", "Observed G1"),
        ("run10", r"2LVM | $\mathbf{Run\ 10}$", "#0072B2", "Two-layer (Run 10)"),
        ("run15", r"LGVM | $\mathbf{Run\ 15}$", "#D55E00", "Linear gradient (Run 15)")]


def panel_letter(ax, letter):
    ax.text(0.02, 0.98, f"({letter})", transform=ax.transAxes, ha="left", va="top",
            fontsize=FS, fontweight="bold")


norm = LogNorm(vmin=0.01, vmax=1)
cmap = plt.get_cmap("viridis")

fig = plt.figure(figsize=(16.5, 12.0))
grid = fig.add_gridspec(1, 4, left=0.085, right=0.985, bottom=0.43, top=0.935, wspace=0.14)
axes = [fig.add_subplot(grid[0, j]) for j in range(4)]

for ax, letter, (name, title, colour, label) in zip(axes, "abc", sets):
    for r in results[name]:
        keep = r["time"] <= TRACE_END
        time = r["time"][keep]
        y = r["distance"]
        height = GAIN * r["relative_envelope"][keep]
        line_colour = cmap(norm(r["relative_peak"]))
        ax.axhline(y, color=".90", lw=0.45, zorder=0)
        ax.fill_between(time, y, y + height, color=line_colour, alpha=0.22, lw=0)
        ax.plot(time, y + height, color=line_colour, lw=1.05)
        top = np.argmax(height)
        ax.plot(time[top], y + height[top], "o", ms=3.5, color=colour, zorder=6)
        if name == "obs":
            ax.text(0.04, y + 0.7, f"S{r['shot']:02d}", fontsize=16, color=".2")
        factor = 1 / r["relative_peak"]
        decimals = max(0, 1 - int(np.floor(np.log10(factor))))
        ax.text(4.88, y + 1.65, f"×{factor:.{decimals}f}", ha="right", va="center", fontsize=16,
                zorder=20, bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="black", lw=0.7))
    ax.plot(*peak_time_fit(results[name]), color=colour, lw=1.35, zorder=5)
    axes[3].plot(*peak_time_fit(results[name]), color=colour, lw=1.2)
    ax.set_title(title, loc="left", fontsize=FS, color=colour)
    panel_letter(ax, letter)

axes[3].set_title("Peak-time trends", loc="left", fontsize=FS)
panel_letter(axes[3], "d")
for ax in axes:
    ax.set(xlim=(0, 5), ylim=(-1, 102), xticks=range(6), yticks=range(0, 100, 20))
    ax.set_xlabel("Time after shot [s]")
    ax.tick_params(axis="y", labelleft=ax is axes[0])
axes[0].set_ylabel("Source–receiver distance [m]")

x0, x1 = axes[0].get_position().x0, axes[2].get_position().x1
bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), orientation="horizontal",
                   cax=fig.add_axes([x0, 0.355, x1 - x0, 0.015]))
bar.set_label("Relative waveform peak amplitude (1 / amplification factor)", fontsize=FS - 1, labelpad=2)
bar.ax.tick_params(labelsize=16, pad=3, length=2)

ax = fig.add_axes([0.085, 0.075, 0.9, 0.205])
for name, title, colour, label in sets:
    ax.plot([r["distance"] for r in results[name]], [r["relative_peak"] for r in results[name]],
            "-o", color=colour, lw=1.5, ms=4, label=label)
ax.set_yscale("log")
ax.set(xlim=(-1, 94), xlabel="Source–receiver distance [m]", ylabel="Relative waveform\npeak amplitude")
ax.set_title("Relative waveform peak amplitudes", loc="left", fontsize=FS)
panel_letter(ax, "e")
ax.grid(axis="y", color=".85", lw=0.65)
ax.grid(axis="x", color=".92", lw=0.5)
ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y:g}"))
ax.yaxis.set_minor_formatter(NullFormatter())
ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.155), borderpad=0.45, borderaxespad=0.3, fontsize=FS - 1)

fig.suptitle("Observed and synthetic amplitudes | 30–60 Hz", fontsize=FS, y=0.995)
fig.savefig("Figure12.png", dpi=104, bbox_inches="tight", facecolor="white")
