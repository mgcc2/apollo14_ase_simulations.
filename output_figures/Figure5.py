"""Figure 5: example random V_P perturbation fields for the scattering models (linear and power-law tapers)."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

FS = 18.5
plt.rcParams.update({"font.size": FS, "xtick.labelsize": FS - 1, "ytick.labelsize": FS - 1})

BASE = 84.5   # model depth [m]
LAYER = 7.5   # layer boundary [m]
x = np.linspace(-75, 75, 360)
z = np.linspace(0, BASE, 220)
rng = np.random.default_rng(14)


def make_noise():
    """Uniform random noise in 2 x 2 blocks, scaled to [-1, 1]."""
    small = rng.uniform(-1, 1, (len(z) // 2, len(x) // 2))
    noise = np.repeat(np.repeat(small, 2, axis=0), 2, axis=1)
    return noise / np.abs(noise).max()


def taper(kind, top, bottom):
    u = np.clip((z - top) / (bottom - top), 0, 1)
    return 1 - u if kind == "linear" else 1 - u**0.25


def make_field(kind, amp, noise, layered=False):
    if layered:
        t = np.where(z <= LAYER, taper(kind, 0, LAYER), taper(kind, LAYER, BASE))
    else:
        t = taper(kind, 0, BASE)
    field = amp * t[:, None] * noise
    if kind == "power":   # full amplitude in the top 4 rows of each layer
        cap = amp * noise[:4] / np.abs(noise[:4]).max()
        field[:4] = cap
        if layered:
            k = np.argmax(z >= LAYER)
            field[k:k + 4] = cap
    return field


noise_linear, noise_power, noise_layer_linear, noise_layer_power = (make_noise() for _ in range(4))

# title, label of runs, field, amplitude [%], layered
panels = [
    ("Global linear (±25%)", "Runs 06, 12", make_field("linear", 25, noise_linear), 25, False),
    ("Global linear (±50%)", "Runs 07, 13", make_field("linear", 50, noise_linear), 50, False),
    ("2-layer linear (±50%)", r"Run $\mathbf{10}$",
     make_field("linear", 50, noise_layer_linear, layered=True), 50, True),
    ("Global power-law (±25%)", "Runs 08, 14", make_field("power", 25, noise_power), 25, False),
    ("Global power-law (±50%)", r"Runs 09, $\mathbf{15}$", make_field("power", 50, noise_power), 50, False),
    ("2-layer power-law (±50%)", "Run 11",
     make_field("power", 50, noise_layer_power, layered=True), 50, True),
]

fig, axs = plt.subplots(2, 3, figsize=(16.5, 10.6), sharex=True, sharey=True)
fig.subplots_adjust(left=0.085, right=0.99, top=0.915, bottom=0.2, wspace=0.16, hspace=0.2)

for letter, ax, (title, runs, field, amp, layered) in zip("abcdef", axs.flat, panels):
    ax.pcolormesh(x, z, field, shading="nearest", cmap="RdBu_r", norm=Normalize(-amp, amp),
                  rasterized=True)
    ax.set_title(title, fontsize=FS)
    ax.text(0.018, 0.025, f"({letter})", transform=ax.transAxes, ha="left", va="bottom",
            fontweight="bold", zorder=40,
            bbox=dict(boxstyle="square,pad=0.10", fc="white", ec="none", alpha=0.95))
    ax.text(0.5, 0.04, runs, transform=ax.transAxes, ha="center", va="bottom", fontsize=FS - 1,
            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="black"), zorder=6)
    if layered:
        ax.axhline(LAYER, color="black", ls=(0, (5, 3)), lw=1.4, zorder=5)
    ax.set_xlim(-75, 75)
    ax.set_ylim(BASE, 0)
    ax.set_xticks(np.arange(-75, 76, 25))
    ax.set_yticks(np.arange(0, 81, 10))

for ax in axs[:, 0]:
    ax.set_ylabel("Depth [m]")
for ax in axs[1, :]:
    ax.set_xlabel("Distance along model [m]")

# row labels, separating bars, column headers
pa, pb, pc = (axs[0, j].get_position() for j in range(3))
pd = axs[1, 0].get_position()
for p, label in ((pa, "Linear taper"), (pd, "Power-law taper")):
    fig.text(0.02, 0.5 * (p.y0 + p.y1), label, rotation=90, ha="center", va="center",
             fontweight="bold")

fig.canvas.draw()
inv = fig.transFigure.inverted()
row2_title_top = inv.transform(axs[1, 0].title.get_window_extent())[1, 1]
row1_title_top = inv.transform(axs[0, 0].title.get_window_extent())[1, 1]

xbar = 0.5 * (pa.x1 + pb.x0)
ybar = pa.y0 - 0.62 * (pa.y0 - row2_title_top)
fig.add_artist(plt.Line2D([xbar, xbar], [pd.y0, pa.y1], color="black", lw=2.0))
fig.add_artist(plt.Line2D([pa.x0, pc.x1], [ybar, ybar], color="black", lw=2.0))
fig.text(0.5 * (pa.x0 + pa.x1), row1_title_top + 0.012, "±25%", ha="center", va="bottom",
         fontsize=FS + 1, fontweight="bold")
fig.text(0.5 * (pb.x0 + pc.x1), row1_title_top + 0.012, "±50%", ha="center", va="bottom",
         fontsize=FS + 1, fontweight="bold")

# colour bars
cax25 = fig.add_axes([pa.x0, 0.075, pa.width, 0.022])
cax50 = fig.add_axes([pb.x0, 0.075, pc.x1 - pb.x0, 0.022])
cb25 = fig.colorbar(ScalarMappable(Normalize(-25, 25), "RdBu_r"), cax=cax25, orientation="horizontal")
cb50 = fig.colorbar(ScalarMappable(Normalize(-50, 50), "RdBu_r"), cax=cax50, orientation="horizontal")
cb25.set_ticks([-25, -10, 0, 10, 25])
cb50.set_ticks([-50, -40, -20, 0, 20, 40, 50])
for cb in (cb25, cb50):
    cb.set_label("$V_P$ perturbation [%]", labelpad=4)

fig.savefig("Figure5.png", dpi=103, bbox_inches="tight", facecolor="white")
