"""Figure 1: Apollo 14 ASE records (30-60 Hz) on geophones 1-3 with the Kovach (1971) picks."""

import csv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator
from obspy import read
from obspy.signal.filter import bandpass, envelope

SEGY_DIR = "path/to/segy_data"   # shot_<N>_apollo_14.segy files and apollo14_ase_data_catalog.txt
PICKS_FILE = "path/to/apollo14_previous_p_wave_picks_97.txt"   # published P-wave picks

FS = 18.5
FS_INSET = 14.5
plt.rcParams.update({"font.size": FS, "xtick.labelsize": FS - 1, "ytick.labelsize": FS - 1,
                     "legend.fontsize": 16})

WAVE = "#546DFA"
ENVELOPE = "#8A2BE2"
KOVACH = "#1D3557"
MARKERS = {"DIRECT": "o", "HEAD": "s"}
DASH = (0, (6, 3, 1.5, 3))

GEOPHONES = ["G1", "G2", "G3"]   # SEGY trace order
TIPPED_G2 = [1, 2, 3, 4, 7]      # shots recorded while G2 was tipped over
SHORT_SHOTS = [12, 19]           # records that stop early
FREQ = (30, 60)                  # band-pass [Hz]
TMAX = 5.0                       # s
CUT_END = 0.10                   # s removed from the end of short records (filter edge)
CLIP = 1.6                       # clip amplified traces at this many trace heights


def filter_trace(x, fs):
    """Linear detrend, reflection padding, then a zero-phase 4-pole Butterworth band-pass."""
    i = np.arange(len(x))
    x = x - np.polyval(np.polyfit(i, x, 1), i)
    x = x - x.mean()
    n = min(max(1, round(0.25 * fs)), len(x) - 1)
    y = bandpass(np.pad(x, n, mode="reflect"), *FREQ, df=fs, corners=4, zerophase=True)
    return y[n:-n]


def smooth_envelope(x, fs):
    """Envelope smoothed with a 0.05 s running mean."""
    n = round(0.05 * fs)
    env = np.pad(envelope(x), (n // 2, n - 1 - n // 2), mode="edge")
    return np.convolve(env, np.ones(n) / n, mode="valid")


# ---- read and process the records
with open(f"{SEGY_DIR}/apollo14_ase_data_catalog.txt", encoding="utf-8-sig") as f:
    catalog = list(csv.DictReader(f, delimiter="\t"))

traces = {g: [] for g in GEOPHONES}
for row in catalog:
    shot = int(row["ASE_data"].split()[-1])
    for g, tr in zip(GEOPHONES, read(f"{SEGY_DIR}/shot_{shot}_apollo_14.segy", format="SEGY")):
        fs = tr.stats.sampling_rate
        raw = tr.data.astype(float)
        stop = min(np.flatnonzero(raw)[-1] + 1, int(TMAX * fs) + 1)
        wave = filter_trace(raw[:stop], fs)
        env = smooth_envelope(wave, fs)
        t = np.arange(stop) / fs
        if shot in SHORT_SHOTS:
            keep = t <= t[-1] - CUT_END
            t, wave, env = t[keep], wave[keep], env[keep]
        peak = np.abs(wave).max()
        traces[g].append(dict(shot=shot, dist=float(row[f"Distance_{g}_m"]), t=t,
                              wave=wave / peak, env=env / peak, peak=peak))

# sort by distance; gain = amplification relative to the largest trace on that geophone
for g in GEOPHONES:
    traces[g].sort(key=lambda tr: tr["dist"])
    largest = max(tr["peak"] for tr in traces[g])
    for tr in traces[g]:
        tr["gain"] = largest / tr["peak"]

# G3 shot 19 is a short record: draw it with the mean gain of its two neighbours instead
g3 = traces["G3"]
k = [tr["shot"] for tr in g3].index(19)
gain = (g3[k - 1]["gain"] + g3[k + 1]["gain"]) / 2
g3[k]["wave"] *= gain / g3[k]["gain"]
g3[k]["env"] *= gain / g3[k]["gain"]
g3[k]["gain"] = gain

# Kovach (1971) picks (source, phase, shot, geophone, shot_x, offset, time), placed on their trace
picks = []
for line in open(PICKS_FILE, encoding="utf-8-sig"):
    source, phase, shot, g, _, _, t = (line.split() + [""] * 7)[:7]
    if source == "Kovach_1971":
        dist = next(tr["dist"] for tr in traces[g] if tr["shot"] == int(shot))
        picks.append((g, int(shot), dist, float(t), phase))

all_distances = np.array([tr["dist"] for g in GEOPHONES for tr in traces[g]])
height = 0.42 * np.diff(np.unique(all_distances)).min()
pad = max(2.5, 1.35 * height)
y0, y1 = -pad, max(100, all_distances.max()) + pad


def add_inset(ax, distance_range, time_range, box, ls):
    """Zoom on the G2 traces in distance_range: one row per trace, renormalised in the window."""
    rows = sorted((tr for tr in traces["G2"] if distance_range[0] <= tr["dist"] <= distance_range[1]),
                  key=lambda tr: (tr["dist"], tr["shot"]))
    t0, t1 = time_range
    ax.add_patch(Rectangle((t0, rows[0]["dist"] - 1.6), t1 - t0, rows[-1]["dist"] - rows[0]["dist"] + 4.2,
                           fill=False, lw=1.4, ls=ls, zorder=15))

    inset = ax.inset_axes([box[0], box[1] - box[3], box[2], box[3]], zorder=30)
    for spine in inset.spines.values():
        spine.set(linewidth=1.4, linestyle=ls)
    inset.tick_params(labelsize=FS_INSET, length=2.5, width=0.7, pad=1.5)

    for i, tr in enumerate(rows):
        window = (tr["t"] >= t0) & (tr["t"] <= t1)
        scale = np.abs(tr["wave"][window]).max()
        inset.plot(tr["t"][window], i + 0.42 * tr["wave"][window] / scale, color=WAVE, lw=0.7, zorder=5)
        inset.plot(tr["t"][window], i + 0.42 * tr["env"][window] / scale, color=ENVELOPE, lw=1.2,
                   ls=DASH if tr["shot"] in TIPPED_G2 else "-", zorder=6)
        inset.text(t1 + 0.02 * (t1 - t0), i, f"S{tr['shot']}", va="center", fontsize=FS_INSET)
        for g, shot, _, t, phase in picks:
            if g == "G2" and shot == tr["shot"] and t0 <= t <= t1:
                inset.scatter(t, i, s=27, marker=MARKERS[phase], fc=KOVACH, ec="black", lw=0.6, zorder=10)

    inset.set(xlim=time_range, ylim=(-0.6, len(rows) - 0.4))
    inset.set_ylabel("Distance [m]", fontsize=FS_INSET, labelpad=2)
    inset.set_yticks(range(len(rows)))
    inset.set_yticklabels([f"{tr['dist']:.0f}" for tr in rows])
    inset.xaxis.set_major_locator(MultipleLocator(0.1))


# ---- plot
fig, axes = plt.subplots(1, 3, figsize=(16.5, 10.5), sharey=True)
fig.subplots_adjust(left=0.06, right=0.995, bottom=0.075, top=0.955, wspace=0.06)

for letter, ax, g in zip("abc", axes, GEOPHONES):
    for tr in traces[g]:
        ls = DASH if g == "G2" and tr["shot"] in TIPPED_G2 else "-"
        ax.plot(tr["t"], tr["dist"] + height * np.clip(tr["wave"], -CLIP, CLIP), color=WAVE, lw=0.5, zorder=8)
        ax.plot(tr["t"], tr["dist"] + height * np.clip(tr["env"], 0, CLIP), color=ENVELOPE, lw=1.2,
                ls=ls, zorder=9)

    # one gain label per distance (the larger one where two G2 shots share a distance)
    for d in np.unique([tr["dist"] for tr in traces[g]]):
        f = max(tr["gain"] for tr in traces[g] if tr["dist"] == d)
        ax.text(0.985 * TMAX, d + 1.15 * height, f"×{f:.1f}" if f < 10 else f"×{f:.0f}",
                ha="right", va="center", fontsize=FS - 1, zorder=25,
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="black"))

    for pg, _, d, t, phase in picks:
        if pg == g:
            ax.scatter(t, d, s=34, marker=MARKERS[phase], fc=KOVACH, ec="black", lw=0.6, zorder=20)

    ax.set_title(f"Geophone {g[1]}", fontsize=FS, pad=12)
    ax.text(0.018, 0.98, f"({letter})", transform=ax.transAxes, va="top", fontsize=FS,
            fontweight="bold", bbox=dict(fc="white", ec="none", pad=1), zorder=40)
    ax.set_xlim(0, TMAX)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.set_xlabel("Time after shot [s]")

add_inset(axes[1], (31, 47), (0.05, 0.55), (0.15, 0.955, 0.70, 0.235), "-")
add_inset(axes[1], (4, 28), (0.0, 0.35), (0.15, 0.665, 0.70, 0.125), (0, (5, 3)))

handles = [
    Line2D([], [], color=WAVE, lw=1.2, label="Waveform"),
    Line2D([], [], color=ENVELOPE, lw=1.4, label="Envelope"),
    Line2D([], [], color=ENVELOPE, lw=1.4, ls=DASH, label="Tipped-over G2"),
    Line2D([], [], marker="o", ls="", ms=6, mfc=KOVACH, mec="black", mew=0.6, label="Kovach direct"),
    Line2D([], [], marker="s", ls="", ms=6, mfc=KOVACH, mec="black", mew=0.6, label="Kovach head"),
]
axes[0].legend(handles=handles, loc="center", bbox_to_anchor=(0.5, (55 - y0) / (y1 - y0)), ncol=2,
               framealpha=0.95, handlelength=1.6, handletextpad=0.4, borderpad=0.4,
               labelspacing=0.3, columnspacing=1.0)
axes[0].set_ylim(y0, y1)
axes[0].yaxis.set_major_locator(MultipleLocator(20))
axes[0].set_ylabel("Source–receiver distance [m]")

fig.savefig("Figure1.png", dpi=103, bbox_inches="tight", facecolor="white")
