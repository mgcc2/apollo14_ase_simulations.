"""Supplementary Figure 3: observed traces aligned on each author's published pick, 30-60 Hz."""

import csv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from obspy import read
from obspy.signal.filter import bandpass, envelope

SEGY_DIR = "path/to/segy_data"   # shot_<N>_apollo_14.segy files and apollo14_ase_data_catalog.txt
PICKS_FILE = "path/to/apollo14_previous_p_wave_picks_97.txt"   # published P-wave picks

BAND = (30.0, 60.0)   # Hz, plotted waveforms
TITLE = "Traces aligned on each author's pick   |   30–60 Hz · all 97 picks"

SHOTS = (1, 2, 3, 4, 7, 11, 12, 13, 17, 18, 19, 20, 21)
GEOPHONES = ("G1", "G2", "G3")
SHORT_SHOTS = (12, 19)   # records that stop early
TIPPED_G2 = (1, 2, 3, 4, 7)   # shots where G2 had tipped over
TMAX = 1.1   # s, end of the processed traces
MATCH_TOL = 2.5   # m, offset tolerance for picks published without shot/geophone
T0, T1 = -0.15, 0.30   # s, window around each pick

WAVE, ENV = "#1B3CF8", "#8A2BE2"
AUTHORS = {"Kovach_1971": "Kovach (1971)", "Villanova_2025": "Villanova (2025)",
           "Martella_2022": "Martella (2022)", "Wang_2025": "Wang (2025)"}
IDENTIFIED = ("Kovach_1971", "Villanova_2025")   # authors who give shot and geophone for each pick
PHASES = {"DIRECT": ("Direct", "#0057B8"), "HEAD": ("Head", "#D62828"), "UNKNOWN": ("Phase not given", "black")}
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

# rows per author: (pick, trace), direct (and phase-not-given) picks first, then by distance
rows = {}
for source in AUTHORS:
    chosen = [(p, best_trace(p, ref)) for p in picks if p["source"] == source]
    chosen.sort(key=lambda r: (r[0]["phase"] == "HEAD", r[1]["dist"], r[0]["time"]))
    rows[source] = [(p, traces[tr["shot"], tr["geo"]]) for p, tr in chosen]

height = 0.225 * max(len(r) for r in rows.values()) + 1.95
fig = plt.figure(figsize=(16, height))
grid = fig.add_gridspec(2, 3, height_ratios=[len(rows["Martella_2022"]), len(rows["Wang_2025"])],
                        left=0.065, right=0.965, top=1 - 1.2 / height, bottom=0.55 / height,
                        wspace=0.36, hspace=0.17)
axes = [fig.add_subplot(grid[:, 0]), fig.add_subplot(grid[:, 1]),
        fig.add_subplot(grid[0, 2]), fig.add_subplot(grid[1, 2])]

for ax, source, letter in zip(axes, AUTHORS, "abcd"):
    ax.axvspan(T0, 0, color="0.95", zorder=0)
    ax.axvline(0, color="k", lw=1.0, zorder=12)
    for i, (p, tr) in enumerate(rows[source]):
        t = tr["t"] - p["time"]
        keep = (t >= T0) & (t <= T1)
        scale = np.abs(tr["y"][keep]).max()
        tipped = tr["geo"] == "G2" and tr["shot"] in TIPPED_G2
        ax.plot(t[keep], i + 0.45 * tr["y"][keep] / scale, color=WAVE, alpha=0.4, lw=0.5, zorder=5)
        ax.plot(t[keep], i + 0.45 * tr["e"][keep] / scale, color=ENV, lw=0.95, ls=DASH if tipped else "-", zorder=6)

    ax.set_yticks(range(len(rows[source])))
    ax.set_yticklabels([f"S{tr['shot']}-{tr['geo']}  {tr['dist']:.0f} m" for _, tr in rows[source]], fontsize=6.6)
    phases = [p["phase"] for p, _ in rows[source]]
    for label, phase in zip(ax.get_yticklabels(), phases):
        label.set_color(PHASES[phase][1])
    for phase in dict.fromkeys(phases):   # phase name beside each group, line between groups
        idx = [i for i, x in enumerate(phases) if x == phase]
        ax.text(1.015, (idx[0] + idx[-1]) / 2, f"{PHASES[phase][0]} picks", transform=ax.get_yaxis_transform(),
                rotation=270, ha="left", va="center", fontsize=8.5, color=PHASES[phase][1], fontweight="bold")
        if idx[0] > 0:
            ax.axhline(idx[0] - 0.5, color="0.4", lw=0.7, zorder=11)

    ax.set_xlim(T0, T1)
    ax.set_ylim(-0.7, len(phases) - 0.3)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_xlabel("Time relative to pick [s]")
    ax.tick_params(direction="out", length=3, width=0.8)
    ax.tick_params(axis="y", length=0)
    note = "" if source in IDENTIFIED else " (best-matching trace)"
    ax.text(0.0, 1.012, f"({letter})", transform=ax.transAxes, ha="left", va="bottom", fontsize=10.5,
            fontweight="bold")
    ax.text(0.5, 1.012, AUTHORS[source] + note, transform=ax.transAxes, ha="center", va="bottom", fontsize=9.5)
axes[0].set_ylabel("Shot – geophone, distance   (by distance within each phase)")

handles = [Line2D([], [], color=WAVE, lw=1.2, alpha=0.5, label="Waveform"),
           Line2D([], [], color=ENV, lw=1.2, label="Envelope"),
           Line2D([], [], color=ENV, lw=1.2, ls=DASH, label="Tipped-over G2"),
           Line2D([], [], color="k", lw=1.5, label="Pick time")]
fig.suptitle(TITLE, fontsize=11.5, fontweight="bold", y=1 - 0.2 / height)
fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1 - 0.38 / height),
           fontsize=8.5, frameon=False, handlelength=2.2, columnspacing=1.6)

fig.savefig("Supplementary_Figure3.png", dpi=300, facecolor="white")
