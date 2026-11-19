"""Figure 8: required number of Fourier modes N_r from wavefield scanning, for shot 17 in Runs 02, 03, 06 and 10."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from netCDF4 import Dataset
from scipy.interpolate import RegularGridInterpolator

RUNS_DIR = "path/to/runs"   # AxiSEM3D run folders A14_Run<NN>_* (input/ and output/scanning_output_Nr.nc)

RUNS = ["02", "03", "06", "10"]
SOUTH, NORTH = 73.152, 76.848   # model extent south and north of the shot [m]
SOURCE_DEPTH = 0.0   # [m]


def weight(depth, low, mid, high):
    """1 at the undulating surface, falling linearly to 0 at the ends of its depth range."""
    return np.clip(np.minimum((depth - low) / (mid - low), (high - depth) / (high - mid)), 0, 1)


def deform(depth, x, y, layers):
    """Map reference depths to depths in the model with topography and rough interfaces."""
    depth = np.broadcast_to(np.asarray(depth, dtype=float), np.shape(x))
    result = depth.copy()
    for grid, low, mid, high in layers:
        result -= weight(depth, low, mid, high) * grid(np.column_stack([x, y]))
    return result


def read_run(run):
    folder = next(Path(RUNS_DIR).glob(f"A14_Run{run}_*"))
    config = yaml.safe_load((folder / "input/inparam.model.yaml").read_text())

    with Dataset(folder / "input" / config["model1D"]["exodus_mesh"]) as ds:
        mesh_top, mesh_bottom = ds["coordy"][:].max(), ds["coordy"][:].min()
    with Dataset(folder / "output/scanning_output_Nr.nc") as ds:
        s, z = ds["pointwise_sz"][:].data.T   # cylindrical radius and height [m]
        nr = ds["pointwise_Nr"][:].data
        nr_start = ds["starting_Nr_for_scanning"][:].data

    # undulating surfaces (topography, rough interface): (interpolator, top of range, own depth, bottom)
    layers = []
    for item in config["list_of_3D_models"]:
        model = next(iter(item.values()))
        if model["activated"] and model["class_name"] == "StructuredGridG3D":
            with Dataset(folder / "input" / model["nc_data_file"]) as ds:
                grid = RegularGridInterpolator(
                    (ds["x"][:].astype(float), ds["y"][:].astype(float)),
                    ds[model["undulation_data"]["nc_var"]][:].astype(float) * model["undulation_data"]["factor"])
            low, high = model["undulation_range"]["min_max"]
            layers.append((grid, low, model["undulation_range"]["interface"], high))

    stations = {}
    for line in (folder / "input/shot_17.txt").read_text().splitlines():
        f = line.split("#", 1)[0].split()
        if f and f[0] in ("G1", "G2", "G3"):   # name, network, distance, angle, -, depth
            stations[f[0]] = (float(f[2]), float(f[3]), float(f[5]))

    return float(mesh_top), float(mesh_bottom), s, z, nr, nr_start, layers, stations


x = np.linspace(-SOUTH, NORTH, 1000)   # profile through the shot
y = np.zeros_like(x)
results = {}
for run in RUNS:
    mesh_top, mesh_bottom, s, z, nr, nr_start, layers, stations = read_run(run)
    depth = mesh_top - z
    mesh_depth = mesh_top - mesh_bottom

    # unfold the axisymmetric section into a south (-) and north (+) profile
    south = (s > 1e-10) & (s <= SOUTH + 1e-8)
    north = s <= NORTH + 1e-8
    px = np.concatenate([-s[south], s[north]])
    pz = deform(np.concatenate([depth[south], depth[north]]), px, np.zeros_like(px), layers)
    pnr = np.concatenate([nr[south], nr[north]])
    active = pnr > 1

    distance, angle, station_depth = np.array([stations[g] for g in ("G1", "G2", "G3")]).T
    station_x, station_y = distance * np.cos(angle), distance * np.sin(angle)
    surface = deform(0.0, x, y, layers)
    results[run] = dict(
        x=px[active], depth=pz[active], nr=pnr[active], surface=surface,
        boundaries=[deform(mid, x, y, layers) for _, _, mid, _ in layers if mid > 0],
        station_x=station_x, stations=deform(station_depth, station_x, station_y, layers),
        source=deform(SOURCE_DEPTH, [0.0], [0.0], layers)[0],
        top=surface.min() - 2.0,
        bottom=max(mesh_depth, deform(mesh_depth, x, y, layers).max()),
        nr_max=nr_start.max(),
    )

top = min(r["top"] for r in results.values())
bottom = max(r["bottom"] for r in results.values())
nr_max = max(r["nr_max"] for r in results.values())
box = dict(boxstyle="square,pad=0.12", fc="white", ec="black", lw=0.7)

FS = 16.5
plt.rcParams.update({"font.size": FS, "xtick.labelsize": FS - 1, "ytick.labelsize": FS - 1})
fig, axes = plt.subplots(2, 2, figsize=(16.5, 10.5), sharex=True, sharey=True)
fig.subplots_adjust(left=0.09, right=0.875, bottom=0.11, top=0.90, wspace=0.045, hspace=0.075)

for letter, ax, (run, r) in zip("abcd", axes.flat, results.items()):
    points = ax.scatter(r["x"], r["depth"], c=r["nr"], s=4, edgecolors="none", cmap="viridis",
                        vmin=1, vmax=nr_max, zorder=5, rasterized=True)

    # free surface (white above it), layer boundary, source and geophones
    ax.fill_between(x, top, r["surface"], fc="white", ec="none", zorder=20)
    ax.plot(x, r["surface"], color="black", lw=1.2, zorder=21)
    for boundary in r["boundaries"]:
        ax.plot(x, boundary, color="0.10", ls="--", lw=1.3, zorder=22)

    ax.scatter(0, r["source"], marker="*", s=130, fc="gold", ec="black", lw=0.7, zorder=30, clip_on=False)
    ax.annotate("Shot 17", (0, r["source"]), xytext=(0, 9), textcoords="offset points",
                ha="center", va="bottom", bbox=box, zorder=31, annotation_clip=False)
    for name, sx, sz in zip(["G1", "G2", "G3"], r["station_x"], r["stations"]):
        ax.scatter(sx, sz, marker="v", s=70, fc="white", ec="black", lw=0.9, zorder=30, clip_on=False)
        ax.annotate(name, (sx, sz), xytext=(0, 7), textcoords="offset points",
                    ha="center", va="bottom", bbox=box, zorder=31, annotation_clip=False)

    ax.text(0.015, 0.02, f"({letter}) Run {run}", transform=ax.transAxes, ha="left", va="bottom",
            fontweight="bold", zorder=50,
            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.95))
    ax.set_xlim(-SOUTH, NORTH)
    ax.set_ylim(bottom, top)
    ax.set_aspect("equal")
    ax.set_axisbelow(True)
    ax.grid(color="0.88", lw=0.6)
    ax.spines["top"].set_visible(False)

# shared labels, colour bar and title
left, right = axes[0, 0].get_position().x0, axes[0, 1].get_position().x1
low, high = axes[1, 0].get_position().y0, axes[0, 0].get_position().y1
fig.supxlabel("Distance from Shot 17 [m] | South (−), North (+)", x=(left + right) / 2,
              y=low - 0.075, fontsize=FS)
fig.supylabel("Depth below reference surface [m]", x=0.025, y=(low + high) / 2, fontsize=FS)
fig.suptitle(r"Shot 17 | Wavefield-scanning $\mathrm{N_r}$ comparison", fontsize=FS + 1,
             x=(left + right) / 2, y=0.975)
cax = fig.add_axes([right + 0.022, low, 0.015, high - low])
fig.colorbar(points, cax=cax).set_label(r"Required $\mathrm{N_r}$ from wavefield scanning", labelpad=10)

fig.savefig("Figure8.png", dpi=108, bbox_inches="tight", facecolor="white")
