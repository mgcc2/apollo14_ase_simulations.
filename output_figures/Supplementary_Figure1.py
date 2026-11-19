"""Supplementary Figure 1: every observed trace with the published picks (one per author), 30-60 Hz."""

import csv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from obspy import read
from obspy.signal.filter import bandpass, envelope

SEGY_DIR = "path/to/segy_data"   # shot_<N>_apollo_14.segy files and apollo14_ase_data_catalog.txt
PICKS_FILE = "path/to/apollo14_previous_p_wave_picks_97.txt"   # published P-wave picks
TWO_LAYER_FILE = "path/to/APOLLO14_best_two_layer_unlabelled.bm"   # best two-layer model (.bm, with "# Global t0")
GRADIENT_FILE = "path/to/APOLLO14_best_linear_gradient_unlabelled.bm"   # best linear-gradient model

BAND = (30.0, 60.0)   # Hz, plotted waveforms
TITLE = "Master zoom: published picks (one per author per trace)   |   30–60 Hz · all 97 picks"

SHOTS = (1, 2, 3, 4, 7, 11, 12, 13, 17, 18, 19, 20, 21)
GEOPHONES = ("G1", "G2", "G3")
SHORT_SHOTS = (12, 19)   # records that stop early
TIPPED_G2 = (1, 2, 3, 4, 7)   # shots where G2 had tipped over
TMAX = 1.1   # s, end of the processed traces
MATCH_TOL = 2.5   # m, offset tolerance for picks published without shot/geophone

WAVE, ENV = "#1B3CF8", "#8A2BE2"
AUTHORS = {"Kovach_1971": ("Kovach (1971)", "#0072B2"),
           "Villanova_2025": ("Villanova (2025)", "#E76F51"),
           "Martella_2022": ("Martella (2022)", "#2A9D8F"),
           "Wang_2025": ("Wang (2025)", "#CC79A7")}
PHASES = {"DIRECT": ("Direct", "-"), "HEAD": ("Head", (0, (1, 1.3))),
          "UNKNOWN": ("Phase not given", (0, (5, 1.5, 1, 1.5)))}
DASH = (0, (4, 1.5, 1, 1.5))   # envelope of a tipped-over G2

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "axes.linewidth": 0.8})


def process(raw, fs, band):
    """Linear detrend, 0.25 s reflected padding, 4-pole zero-phase band-pass; then 0.05 s smoothed envelope."""
    i = np.arange(len(raw))
    x = raw - np.polyval(np.polyfit(i, raw, 1), i)
    x = x - x.mean()
    n = min(round(0.25 * fs), len(x) - 1)
    wave = bandpass(np.pad(x, n, mode="reflect"), *band, df=fs, corners=4, zerophase=True)[n:n + len(x)]
    m = round(0.05 * fs)
    env = np.convolve(np.pad(envelope(wave), (m // 2, m - 1 - m // 2), mode="edge"), np.ones(m) / m, mode="valid")
    return wave, env


def load_traces(band):
    """All 39 shot-geophone traces, processed in the given band, cut to TMAX and normalised."""
    with open(f"{SEGY_DIR}/apollo14_ase_data_catalog.txt", encoding="utf-8-sig") as f:
        distance = {int(r["ASE_data"].split()[-1]): r for r in csv.DictReader(f, delimiter="\t")}
    traces = {}
    for shot in SHOTS:
        stream = read(f"{SEGY_DIR}/shot_{shot}_apollo_14.segy", format="SEGY")
        for geo, tr in zip(GEOPHONES, stream):
            raw = np.asarray(tr.data, float)
            fs = tr.stats.sampling_rate
            stop = int(1.45 * fs) + 1
            if shot in SHORT_SHOTS:
                stop = min(stop, np.flatnonzero(raw)[-1] + 1)
            wave, env = process(raw[:stop], fs, band)
            if shot in SHORT_SHOTS:   # 0.05 s of zeros where the record stops
                extra = min(round(0.05 * fs), int(1.45 * fs) + 1 - stop)
                wave, env = np.r_[wave, np.zeros(extra)], np.r_[env, np.zeros(extra)]
            t = np.arange(len(wave)) / fs
            keep = t <= TMAX
            scale = np.abs(wave[keep]).max()
            traces[shot, geo] = dict(shot=shot, geo=geo, fs=fs, t=t[keep], y=wave[keep] / scale,
                                     e=env[keep] / scale, dist=float(distance[shot][f"Distance_{geo}_m"]))
    return traces


def read_picks():
    picks = []
    for line in open(PICKS_FILE, encoding="utf-8-sig"):
        f = line.split()
        if len(f) == 7 and f[0] in AUTHORS:
            picks.append(dict(source=f[0], phase=f[1], shot=None if f[2] == "NA" else int(f[2]),
                              geo=None if f[3] == "NA" else f[3], offset=float(f[5]), time=float(f[6])))
    return picks


def read_model(path):
    """Depth [m], Vp [m/s] and the fitted time shift t0 [s] from a .bm file."""
    rows, t0 = [], 0.0
    for line in open(path):
        if line.startswith("# Global t0"):
            t0 = float(line.split("=")[1].split()[0])
        f = line.split("#")[0].split()
        if f and f[0][0].isdigit():
            rows.append([float(f[0]), float(f[1])])
    depth, vp = 1000 * np.array(rows).T
    return depth, vp, t0


def model_times(x):
    """Direct and head/diving travel times [s] at distance x [m] for both models."""
    times = []
    depth, vp, t0 = read_model(TWO_LAYER_FILE)
    h = depth[np.flatnonzero((np.diff(depth) == 0) & (np.diff(vp) != 0))[0]]
    v1, v2 = vp[0], vp[-1]
    angle = np.arcsin(v1 / v2)
    times += [x / v1 + t0]
    if x >= 2 * h * np.tan(angle):
        times += [x / v2 + 2 * h * np.cos(angle) / v1 + t0]
    depth, vp, t0 = read_model(GRADIENT_FILE)
    v0, g = vp[0], (vp[-1] - vp[0]) / depth[-1]
    times += [x / v0 + t0]
    if (np.sqrt(v0 ** 2 + (g * x / 2) ** 2) - v0) / g <= depth[-1]:   # turning depth inside the model
        times += [2 / g * np.arcsinh(g * x / (2 * v0)) + t0]
    return times


def candidates(pick, traces):
    """Traces a pick can belong to: its own trace, or every trace within MATCH_TOL of its offset."""
    if pick["shot"] is not None:
        return [traces[pick["shot"], pick["geo"]]]
    return sorted((tr for tr in traces.values() if abs(tr["dist"] - pick["offset"]) <= MATCH_TOL),
                  key=lambda tr: (tr["geo"], tr["dist"]))


def onset_strength(pick, tr):
    """20 log10 of the RMS 60 ms after the pick over the RMS 60 ms before it (None if no room)."""
    i, n = round(pick["time"] * tr["fs"]), round(0.06 * tr["fs"])
    if i - n < 0 or i + n >= len(tr["y"]):
        return None
    before, after = np.sqrt(np.mean(tr["y"][i - n:i] ** 2)), np.sqrt(np.mean(tr["y"][i:i + n] ** 2))
    return 20 * np.log10(after / before) if before and after else None


def best_trace(pick, traces):
    """The candidate trace with the strongest onset at the pick time (30-60 Hz)."""
    options = candidates(pick, traces)
    scores = [(s, k) for k, tr in enumerate(options) if (s := onset_strength(pick, tr)) is not None]
    return options[max(scores)[1] if len(options) > 1 and scores else 0]


ref = load_traces((30.0, 60.0))   # used to assign picks to traces, whatever the plotted band
traces = ref if BAND == (30.0, 60.0) else load_traces(BAND)
picks = read_picks()

# one pick per author per trace: where an author has several, keep the strongest onset
on_trace = {}
for p in picks:
    tr = best_trace(p, ref)
    score = onset_strength(p, tr)
    score = -np.inf if score is None else score
    slot = on_trace.setdefault((tr["shot"], tr["geo"]), {})
    if p["source"] not in slot or score > slot[p["source"]][0]:
        slot[p["source"]] = (score, p)

rows = sorted(traces.values(), key=lambda tr: (tr["dist"], tr["geo"], tr["shot"]))
ncol = 5
nrow = int(np.ceil(len(rows) / ncol))
height = 1.42 * nrow + 1.45
fig, axes = plt.subplots(nrow, ncol, figsize=(16, height), squeeze=False)
fig.subplots_adjust(left=0.012, right=0.988, top=1 - 0.95 / height, bottom=0.5 / height,
                    hspace=0.42, wspace=0.05)

for ax, tr in zip(axes.ravel(), rows):
    # window: 0.15 s before the earliest pick or model arrival on this trace to 0.20 s after the latest
    times = [p["time"] for p in picks if any(c is ref[tr["shot"], tr["geo"]] for c in candidates(p, ref))]
    times += model_times(tr["dist"])
    t0, t1 = max(0.0, min(times) - 0.15), min(TMAX, max(times) + 0.20)
    keep = (tr["t"] >= t0) & (tr["t"] <= t1)
    scale = np.abs(tr["y"][keep]).max()
    tipped = tr["geo"] == "G2" and tr["shot"] in TIPPED_G2
    ax.plot(tr["t"][keep], tr["y"][keep] / scale, color=WAVE, alpha=0.4, lw=0.55, zorder=3)
    ax.plot(tr["t"][keep], tr["e"][keep] / scale, color=ENV, lw=1.0, ls=DASH if tipped else "-", zorder=4)
    for _, p in on_trace.get((tr["shot"], tr["geo"]), {}).values():
        ax.axvline(p["time"], color=AUTHORS[p["source"]][1], lw=1.7, ls=PHASES[p["phase"]][1], zorder=6)
    ax.set(xlim=(t0, t1), ylim=(-1.12, 1.12), yticks=[])
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.tick_params(direction="out", length=2.5, width=0.7, labelsize=7, pad=1.5)
    ax.set_title(f"S{tr['shot']}-{tr['geo']}   {tr['dist']:.1f} m", fontsize=8.5, pad=2.5)
for ax in axes.ravel()[len(rows):]:
    ax.axis("off")

handles = [Line2D([], [], color=c, lw=1.8, label=name) for name, c in AUTHORS.values()]
handles += [Line2D([], [], color="0.5", lw=1.8, ls=ls, label=name) for name, ls in PHASES.values()]
handles += [Line2D([], [], color=WAVE, lw=1.2, alpha=0.5, label="Waveform"),
            Line2D([], [], color=ENV, lw=1.2, label="Envelope"),
            Line2D([], [], color=ENV, lw=1.2, ls=DASH, label="Tipped-over G2")]
fig.suptitle(TITLE, fontsize=11.5, fontweight="bold", y=1 - 0.2 / height)
fig.legend(handles=handles, loc="upper center", ncol=10, bbox_to_anchor=(0.5, 1 - 0.38 / height),
           fontsize=8.5, frameon=False, handlelength=2.2, columnspacing=1.6)
fig.text(0.5, 0.12 / height, "Time after shot [s]", ha="center", va="bottom", fontsize=9)

fig.savefig("Supplementary_Figure1.png", dpi=300, facecolor="white")
