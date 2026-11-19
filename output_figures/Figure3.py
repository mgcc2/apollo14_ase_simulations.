"""Figure 3: all 97 published P-wave picks with the first arrivals of our best linear-gradient model (LGVM)."""

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
MODEL_FILE = "path/to/APOLLO14_best_linear_gradient_unlabelled.bm"   # best linear-gradient model (.bm, with "# Global t0")

FS = 18.5
FS_INSET = 14.5
plt.rcParams.update({"font.size": FS, "xtick.labelsize": FS - 1, "ytick.labelsize": FS - 1,
                     "legend.fontsize": 16})

WAVE = "#546DFA"
ENVELOPE = "#8A2BE2"
MODEL = "#A57C32"
AUTHORS = {"Kovach_1971": ("Kovach (1971)", "#1D3557"),
           "Villanova_2025": ("Villanova (2025)", "#E76F51"),
           "Martella_2022": ("Martella (2022)", "#2A9D8F"),
           "Wang_2025": ("Wang (2025)", "#E9C46A")}
ON_TOP = {"Kovach_1971": 0.1, "Villanova_2025": 0.2, "Martella_2022": 0.3, "Wang_2025": 0.4}
MARKERS = {"DIRECT": "o", "HEAD": "s", "UNKNOWN": "D"}
DASH = (0, (6, 3, 1.5, 3))

GEOPHONES = ["G1", "G2", "G3"]   # SEGY trace order
TIPPED_G2 = [1, 2, 3, 4, 7]      # shots recorded while G2 was tipped over
SHORT_SHOTS = [12, 19]           # records that stop early
FREQ = (30, 60)                  # band-pass [Hz]
PROCESS_END = 1.45               # s of record that is filtered
TMAX = 1.1                       # s shown
MATCH = 2.5                      # m; offset-only picks are drawn on every trace this close in distance
FAR = 60                         # m; picks beyond this are faded


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
        stop = int(PROCESS_END * fs) + 1
        if shot in SHORT_SHOTS:
            stop = min(stop, np.flatnonzero(raw)[-1] + 1)
        wave = filter_trace(raw[:stop], fs)
        env = smooth_envelope(wave, fs)
        if shot in SHORT_SHOTS:   # show 0.05 s of flat line after the end of a short record
            flat = np.zeros(min(round(0.05 * fs), int(PROCESS_END * fs) + 1 - stop))
            wave, env = np.r_[wave, flat], np.r_[env, flat]
        t = np.arange(len(wave)) / fs
        keep = t <= TMAX
        peak = np.abs(wave[keep]).max()
        traces[g].append(dict(shot=shot, dist=float(row[f"Distance_{g}_m"]), t=t[keep],
                              wave=wave[keep] / peak, env=env[keep] / peak))
for g in GEOPHONES:
    traces[g].sort(key=lambda tr: tr["dist"])

# ---- picks: on their own trace, or (offset only) on every trace at about that distance
picks = []
for line in open(PICKS_FILE, encoding="utf-8-sig"):
    fields = line.split()
    if len(fields) == 7 and fields[0] in AUTHORS:   # source, phase, shot, geophone, shot_x, offset, time
        source, phase, shot, g, _, offset, t = fields
        picks.append((source, phase, shot, g, float(offset), float(t)))

markers = []
for source, phase, shot, g, offset, t in picks:
    if g == "NA":
        matches = [(mg, tr) for mg in GEOPHONES for tr in traces[mg] if abs(tr["dist"] - offset) <= MATCH]
    else:
        matches = [(g, tr) for tr in traces[g] if tr["shot"] == int(shot)]
    for mg, tr in matches:
        markers.append((mg, tr["shot"], tr["dist"], t, source, phase, offset))

# ---- linear-gradient model: depth [km] and V_P [km/s] columns, and the fitted time shift t0
lines = open(MODEL_FILE).read().splitlines()
T0 = float(next(l for l in lines if l.startswith("# Global t0")).split("=")[1].split()[0])
table = np.array([l.split()[:2] for l in lines if l.split() and l.split()[0][0].isdigit()], float) * 1000
depth, vp = table.T
V0, BASE = vp[0], depth[-1]
G = (vp[-1] - V0) / BASE   # V_P = V0 + G z


def model_times(x):
    """Direct and diving-wave travel times [s] at distance x [m] (diving ray must turn above the model base)."""
    diving = 2 / G * np.arcsinh(G * x / (2 * V0)) + T0
    turning_depth = (np.sqrt(V0 ** 2 + (G * x / 2) ** 2) - V0) / G
    return x / V0 + T0, np.where(turning_depth <= BASE, diving, np.nan)


def first_arrival(x):
    return np.fmin(*model_times(x))


def draw_pick(ax, t, y, source, phase, offset, size=34, z=20):
    ax.scatter(t, y, s=size, marker=MARKERS[phase], fc=AUTHORS[source][1], ec="black", lw=0.6,
               alpha=0.35 if offset > FAR else 1, zorder=z + ON_TOP[source])


all_distances = np.array([tr["dist"] for g in GEOPHONES for tr in traces[g]])
height = 0.42 * np.diff(np.unique(all_distances)).min()
pad = max(2.5, 1.35 * height)


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
        t = first_arrival(tr["dist"])
        inset.plot([t, t], [i - 0.4, i + 0.4], color=MODEL, lw=2, alpha=0.8, solid_capstyle="butt", zorder=8)
        for g, shot, _, t, source, phase, offset in markers:
            if g == "G2" and shot == tr["shot"] and t0 <= t <= t1:
                draw_pick(inset, t, i, source, phase, offset, size=27, z=10)

    inset.set(xlim=time_range, ylim=(-0.6, len(rows) - 0.4))
    inset.set_yticks(range(len(rows)))
    inset.set_yticklabels([f"{tr['dist']:.0f}" for tr in rows])
    inset.xaxis.set_major_locator(MultipleLocator(0.1))


# ---- plot
fig, axes = plt.subplots(1, 4, figsize=(16.5, 10.84), sharey=True)
fig.subplots_adjust(left=0.06, right=0.995, bottom=0.075, top=0.955, wspace=0.07)

# (a)-(c) waveforms with picks and model first arrivals
for ax, g in zip(axes, GEOPHONES):
    for tr in traces[g]:
        ls = DASH if g == "G2" and tr["shot"] in TIPPED_G2 else "-"
        ax.plot(tr["t"], tr["dist"] + height * tr["wave"], color=WAVE, lw=0.5, zorder=8)
        ax.plot(tr["t"], tr["dist"] + height * tr["env"], color=ENVELOPE, lw=1.2, ls=ls, zorder=9)
        t = first_arrival(tr["dist"])
        ax.plot([t, t], [tr["dist"] - 1.25 * height, tr["dist"] + 1.25 * height], color=MODEL, lw=2.5,
                alpha=0.8, solid_capstyle="butt", zorder=12)
    for mg, _, d, t, source, phase, offset in markers:
        if mg == g:
            draw_pick(ax, t, d, source, phase, offset)
    ax.set_title(f"Geophone {g[1]}", fontsize=FS, pad=12)

add_inset(axes[1], (31, 47), (0.05, 0.50), (0.15, 0.955, 0.70, 0.235), "-")
add_inset(axes[1], (4, 28), (0.0, 0.35), (0.15, 0.665, 0.70, 0.125), (0, (5, 3)))

axes[2].legend(handles=[
    Line2D([], [], color=WAVE, lw=1.2, label="Waveform"),
    Line2D([], [], color=ENVELOPE, lw=1.4, label="Envelope"),
    Line2D([], [], color=ENVELOPE, lw=1.4, ls=DASH, label="Tipped-over G2"),
    Line2D([], [], color=MODEL, lw=2.5, alpha=0.8, label="LGVM prediction")],
    loc="center right", bbox_to_anchor=(1.0, 0.635), framealpha=0.95,
    handlelength=1.6, handletextpad=0.4, borderpad=0.4, labelspacing=0.3)

# (d) travel times of all picks
ax = axes[3]
for source, phase, _, _, offset, t in picks:
    draw_pick(ax, t, offset, source, phase, offset)
x = np.linspace(0, 95, 400)
direct, diving = model_times(x)
ax.plot(direct, x, color=MODEL, lw=2, zorder=17)
ax.plot(diving, x, color=MODEL, lw=2, ls=(0, (4, 2)), zorder=17)
ax.set_title("Travel time picks", fontsize=FS, pad=12)
ax.text(0.98, 0.98, f"$V_P = {V0:.0f} + {G:.1f}\\,z$ m/s", transform=ax.transAxes,
        ha="right", va="top", fontsize=16)

curves = [Line2D([], [], color="0.3", lw=2, label="Direct wave"),
          Line2D([], [], color="0.3", lw=2, ls=(0, (4, 2)), label="Diving wave"),
          Line2D([], [], color=MODEL, lw=2, label="Linear-gradient\nmodel")]
shapes = [Line2D([], [], marker=MARKERS[k], ls="", ms=6, mfc="0.65", mec="black", mew=0.35, label=label)
          for k, label in (("DIRECT", "Direct"), ("HEAD", "Head"), ("UNKNOWN", "Not given"))]
authors = [Line2D([], [], marker="o", ls="", ms=6, mfc=colour, mec="black", mew=0.35, label=label)
           for label, colour in AUTHORS.values()]
ax.legend(handles=curves + shapes + authors, loc="lower right", framealpha=0.95,
          handlelength=1.3, handletextpad=0.4, borderpad=0.4, labelspacing=0.25)

axes[0].set_ylim(-pad, max(100, all_distances.max()) + pad)
axes[0].yaxis.set_major_locator(MultipleLocator(20))
axes[0].set_ylabel("Source–receiver distance [m]")
for letter, ax in zip("abcd", axes):
    ax.text(0.018, 0.98, f"({letter})", transform=ax.transAxes, va="top", fontsize=FS,
            fontweight="bold", bbox=dict(fc="white", ec="none", pad=1), zorder=40)
    ax.set_xlim(0, TMAX)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.set_xlabel("Time after shot [s]")
    ax.set_axisbelow(True)
    ax.grid(color="0.88", lw=0.6)

fig.savefig("Figure3.png", dpi=103, bbox_inches="tight", facecolor="white")
