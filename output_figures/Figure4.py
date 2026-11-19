"""Figure 4: the three velocity models (HVM, 2LVM, LGVM) used in the simulations."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, MultipleLocator

# AxiSEM3D .bm model files
MODEL_FILES = {
    "HVM": "path/to/homogeneous_model.bm",
    "2LVM": "path/to/two_layer_model.bm",
    "LGVM": "path/to/linear_gradient_model.bm",
}

FS = 16.5
plt.rcParams.update({"font.size": FS, "xtick.labelsize": FS - 1, "ytick.labelsize": FS - 1})

REGOLITH = 8.5   # m
MAX_DEPTH = 84.5   # m
colours = {"HVM": "#0B1F8A", "2LVM": "#1D4F15", "LGVM": "#A57C32"}

panels = [("vp", "P-wave velocity, $V_P$", "$V_P$ [m/s]", "m/s"),
          ("vs", "S-wave velocity, $V_S$", "$V_S$ [m/s]", "m/s"),
          ("rho", r"Density, $\rho$", r"$\rho$ [kg/m$^3$]", "kg/m$^3$")]


def read_bm(path):
    """Read depth, vp, vs, rho from a .bm file, converted to m, m/s and kg/m^3."""
    lines = [line.split("#")[0].split() for line in open(path)]
    lines = [words for words in lines if words]
    units = next(words[1] for words in lines if words[0] == "UNITS")
    columns = next(words[1:] for words in lines if words[0] == "COLUMNS")
    table = np.array([[float(w) for w in words] for words in lines if words[0][0] in "-.0123456789"])
    scale = 1000 if units == "km" else 1
    return {name: table[:, columns.index(name)] * scale for name in ("depth", "vp", "vs", "rho")}


def legend_label(name, m, var, unit):
    top, bottom = m[var][0], m[var][-1]
    if name == "HVM":
        return f"Homogeneous (HVM)\n{top:.0f} {unit}"
    if name == "2LVM":
        h = m["depth"][np.flatnonzero(np.diff(m["depth"]) == 0)[0]]
        return f"Two-layer (2LVM)\n{top:.0f} | {bottom:.0f} {unit}\n$h$ = {h:.1f} m"
    gradient = (bottom - top) / (m["depth"][-1] - m["depth"][0])
    return f"Linear gradient (LGVM)\n{top:.0f} + {gradient:.1f}$\\,z$ {unit}"


models = {name: read_bm(path) for name, path in MODEL_FILES.items()}

fig, axes = plt.subplots(1, 3, sharey=True, figsize=(16.5, 10.5))
fig.subplots_adjust(left=0.06, right=0.995, bottom=0.085, top=0.95, wspace=0.08)

for letter, ax, (var, title, xlabel, unit) in zip("abc", axes, panels):
    for name, m in models.items():
        ax.plot(m[var], m["depth"], color=colours[name], lw=2.5,
                label=legend_label(name, m, var, unit))
    ax.axhline(REGOLITH, color="black", lw=1.4, ls=(0, (5, 3)), zorder=1)

    ax.set_title(title, fontsize=FS)
    ax.text(0.98, 0.99, f"({letter})", transform=ax.transAxes, ha="right", va="top",
            fontsize=FS + 1, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylim(MAX_DEPTH, 0)
    ax.margins(x=0.06)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_axisbelow(True)
    ax.grid(color="0.88", lw=0.6)
    ax.legend(loc="upper right", bbox_to_anchor=(0.995 if var != "rho" else 0.948, 0.885),
              fontsize=FS - 1, framealpha=0.95, handlelength=0.9, handletextpad=0.3,
              borderpad=0.3, labelspacing=0.35)

axes[2].xaxis.set_major_locator(MultipleLocator(100))
axes[2].xaxis.set_minor_locator(MultipleLocator(50))
axes[0].set_ylabel("Depth [m]")
axes[1].text(0.97, REGOLITH - 0.6,
             f"Proposed regolith thickness {REGOLITH:g} m\n(Kovach & Watkins, 1973)",
             transform=axes[1].get_yaxis_transform(), ha="right", va="bottom", fontsize=FS - 1)

fig.savefig("Figure4.png", dpi=104, bbox_inches="tight", facecolor="white")
