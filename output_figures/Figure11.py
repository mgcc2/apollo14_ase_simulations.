"""Figure 11: Shot 17 wavefield snapshots (vertical displacement) for Runs 10 and 15, from AxiSEM3D element output."""

import glob

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from netCDF4 import Dataset, chartostring
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator

RUNS_DIR = "path/to/runs"   # AxiSEM3D run folders A14_Run10_* and A14_Run15_* (input/ and output/elements/)

SNAPSHOT_TIMES = [0.15, 0.40, 0.75]   # s
NX = 500   # pixels across the section
SLICES = {"scattering_slice_south": -1, "scattering_slice_north": 1}   # element groups, side of the shot


def read_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def weight(depth, low, mid, high):
    """Fraction of an undulation applied at a reference depth: 1 at its control depth, 0 at its limits."""
    depth = np.asarray(depth, dtype=float)
    return np.clip(np.where(depth <= mid, (depth - low) / (mid - low), (high - depth) / (high - mid)), 0, 1)


def read_undulations(run):
    """Geometric models (topography, rough interfaces) as (grid interpolator, low, mid, high)."""
    models = []
    for item in read_yaml(f"{run}/input/inparam.model.yaml")["list_of_3D_models"]:
        model = next(iter(item.values()))
        if model["class_name"] != "StructuredGridG3D" or not model["activated"]:
            continue
        x_name, y_name = model["coordinates"]["nc_variables"]
        with Dataset(f"{run}/input/{model['nc_data_file']}") as ds:
            var = ds[model["undulation_data"]["nc_var"]]
            values = np.asarray(var[:], dtype=float) * model["undulation_data"].get("factor", 1.0)
            if var.dimensions[0] != x_name:
                values = values.T
            grid = RegularGridInterpolator((ds[x_name][:], ds[y_name][:]), values)
        low, high = sorted(model["undulation_range"]["min_max"])
        models.append((grid, low, model["undulation_range"]["interface"], high))
    return models


def shift(models, depth, x, y=0.0):
    """Upward displacement of reference depth(s) `depth` at horizontal position(s) (x, y)."""
    return sum(weight(depth, low, mid, high) * grid((x, y)) for grid, low, mid, high in models)


def read_boundaries(run):
    """Depths [m] of the discontinuities in the 1-D model (.bm file named in input.mesh.yaml)."""
    name = read_yaml(f"{run}/input/input.mesh.yaml")["basic"]["model"]
    rows, scale = [], 1000.0
    for line in open(f"{run}/input/{name}"):
        words = line.split("#")[0].split()
        if words and words[0].upper() == "UNITS":
            scale = 1000.0 if words[1].lower() == "km" else 1.0
        elif words and words[0][0].isdigit():
            rows.append([float(w) for w in words])
    return [rows[i][0] * scale for i in range(1, len(rows))
            if rows[i][0] == rows[i - 1][0] and rows[i][1:] != rows[i - 1][1:]]


def element_coordinates(ds):
    """GLL-point coordinates as (element, [s, z], GLL point)."""
    var = ds["list_element_coords"]
    dims = [d.lower() for d in var.dimensions]
    e = next(i for i, d in enumerate(dims) if "element" in d)
    g = next(i for i, d in enumerate(dims) if "gll" in d)
    return np.moveaxis(np.asarray(var[:], dtype=float), (e, 3 - e - g, g), (0, 1, 2))


def element_rows(ds, name, n_elements, n_coords):
    """Rows of list_element_coords that belong to the elements of data variable `name`."""
    if n_elements == n_coords:
        return np.arange(n_elements)
    ids = np.asarray(ds["list_element" + name[len("data_wave"):]][:], dtype=int).ravel()
    table = np.asarray(ds["list_element_na"][:], dtype=int).reshape(n_coords, -1)
    for column in table.T:   # element IDs listed in a column of list_element_na
        lookup = {value: row for row, value in enumerate(column)}
        if len(lookup) == n_coords and all(i in lookup for i in ids):
            return np.array([lookup[i] for i in ids])
    return ids   # otherwise the IDs are row indices


def read_slice(folder, side, surface_z):
    """GLL points (x, depth) and U3 values at the snapshot times for one saved element slice."""
    points, values = [], []
    for path in sorted(glob.glob(f"{folder}/axisem3d_synthetics.nc.rank*"), key=lambda p: int(p.rsplit("rank", 1)[1])):
        with Dataset(path) as ds:
            ds.set_auto_mask(False)
            time = np.asarray(ds["data_time"][:])
            frames = [int(np.argmin(np.abs(time - t))) for t in SNAPSHOT_TIMES]
            channel = list(chartostring(ds["list_channel"][:])).index("U3")
            coords = element_coordinates(ds)
            for name in sorted(ds.variables):
                if not name.startswith("data_wave"):
                    continue
                var = ds[name]
                dims = [d.lower() for d in var.dimensions]
                index = [frames if "time" in d else channel if "channel" in d else
                         slice(None) if ("element" in d or "gll" in d) else 0 for d in dims]
                kept = [d for d, i in zip(dims, index) if not isinstance(i, int)]
                order = [next(k for k, d in enumerate(kept) if w in d) for w in ("time", "element", "gll")]
                data = np.moveaxis(np.asarray(var[tuple(index)], dtype=float), order, (0, 1, 2))
                xyz = coords[element_rows(ds, name, data.shape[1], len(coords))]
                points.append(np.column_stack((side * xyz[:, 0, :].ravel(), surface_z - xyz[:, 1, :].ravel())))
                values.append(data.reshape(len(frames), -1))
    points, values = np.concatenate(points), np.concatenate(values, axis=1)
    _, unique = np.unique(np.round(points, 8), axis=0, return_index=True)
    return points[unique], values[:, unique]


runs = {}
for number in (10, 15):
    run = glob.glob(f"{RUNS_DIR}/A14_Run{number}_*")[0]
    model = read_yaml(f"{run}/input/inparam.model.yaml")
    with Dataset(f"{run}/input/{model['model1D']['exodus_mesh']}") as ds:
        mesh_z = np.asarray(ds["coordy"][:])
    surface_z, mesh_depth = mesh_z.max(), mesh_z.max() - mesh_z.min()
    groups = {k: v for item in read_yaml(f"{run}/input/inparam.output.yaml")["list_of_element_groups"]
              for k, v in item.items()}
    source = next(iter(read_yaml(f"{run}/input/inparam.source.yaml")["list_of_sources"][0].values()))
    stations = {}
    for line in open(f"{run}/input/shot_17.txt"):
        words = line.split("#")[0].split()
        if words and words[0] in ("G1", "G2", "G3"):
            r, phi, depth = float(words[2]), float(words[3]), float(words[5])
            stations[words[0]] = (r * np.cos(phi), r * np.sin(phi), depth)

    # pixel grid on the reference (undeformed) section: south of the shot x < 0, north x >= 0
    x_min = -groups["scattering_slice_south"]["elements"]["horizontal_range"][1]
    x_max = groups["scattering_slice_north"]["elements"]["horizontal_range"][1]
    ny = max(2, round(NX * mesh_depth / (x_max - x_min)))
    distance = np.linspace(x_min, x_max, NX)
    reference_depth = np.linspace(0, mesh_depth, ny)
    xx, dd = np.meshgrid(distance, reference_depth)

    # interpolate the saved GLL values onto the pixels, one side at a time
    snapshots = np.full((len(SNAPSHOT_TIMES), ny, NX), np.nan)
    for name, side in SLICES.items():
        points, values = read_slice(f"{run}/output/elements/{name}", side, surface_z)
        pixels = xx < 0 if side < 0 else xx >= 0
        snapshots[:, pixels] = LinearNDInterpolator(points, values.T)(xx[pixels], dd[pixels]).T

    # colour scale: robust maximum away from the source
    source_depth = float(source["location"]["depth"])
    far = np.hypot(xx, dd - source_depth) >= 2.0
    limits = []
    for frame in snapshots:
        a = np.abs(frame[far])
        limits.append(np.percentile(a[np.isfinite(a) & (a > 0)], 99.7))

    # topography and rough interfaces deform the section, the surface and the model boundaries
    models = read_undulations(run)
    runs[number] = dict(
        snapshots=snapshots, scale=np.percentile(limits, 90),
        distance=distance,
        display_depth=reference_depth[:, None] - shift(models, reference_depth[:, None], distance[None, :]),
        surface=0.0 - shift(models, 0.0, distance),
        boundaries=[(b, b - shift(models, b, distance)) for b in read_boundaries(run) if 0 < b < mesh_depth],
        source_depth=source_depth - shift(models, source_depth, 0.0, 0.0),
        stations=[(name, x, z - shift(models, z, x, y)) for name, (x, y, z) in stations.items()],
        x_range=(x_min, x_max), bottom=mesh_depth)

# common display extent and colour scale
x_min, x_max = runs[10]["x_range"]
bottom = max(r["bottom"] for r in runs.values())
surface_min = min(r["surface"].min() for r in runs.values())
top = surface_min - max(2.0, 0.075 * (bottom - surface_min))
scale = max(r["scale"] for r in runs.values())
cmap = plt.get_cmap("RdBu_r").copy()
cmap.set_bad("white")
norm = Normalize(-1, 1)

FS = 20.5
plt.rcParams.update({"font.size": FS, "xtick.labelsize": FS, "ytick.labelsize": FS})

titles = {10: r"2LVM | $\mathbf{Run\ 10}$", 15: r"LGVM | $\mathbf{Run\ 15}$"}
box = dict(boxstyle="square,pad=0.12", fc="white", ec="black", lw=0.7)

height = 17.1
left, right = 0.105, 0.985
fig, axes = plt.subplots(3, 2, figsize=(16.5, height), sharex=True, sharey=True)
fig.subplots_adjust(left=left, right=right, bottom=1.8 / height, top=1 - 1.2 / height,
                    wspace=0.07, hspace=0.12)

for col, (number, r) in enumerate(runs.items()):
    for row in range(3):
        ax = axes[row, col]
        x = np.broadcast_to(r["distance"], r["display_depth"].shape)
        ax.pcolormesh(x, r["display_depth"], r["snapshots"][row] / scale, cmap=cmap, norm=norm,
                      shading="nearest", rasterized=True)
        ax.fill_between(r["distance"], top, r["surface"], color="white", lw=0, zorder=4)
        ax.plot(r["distance"], r["surface"], color="black", lw=1.0, zorder=5)
        for depth, curve in r["boundaries"]:
            ax.plot(r["distance"], curve, color="0.25", ls="--", lw=0.85, zorder=6)
            if row == 0:
                ax.annotate(f"{depth:g} m interface", xy=(r["distance"][-1], curve[-1]),
                            xytext=(-4, -4), textcoords="offset points", ha="right", va="top",
                            bbox=dict(fc="white", ec="none", pad=1), zorder=7)

        ax.scatter(0, r["source_depth"], marker="*", s=130, fc="gold", ec="black", lw=0.65, zorder=9)
        ax.annotate("Shot 17", xy=(0, r["source_depth"]), xytext=(0, 9), textcoords="offset points",
                    ha="center", va="bottom", bbox=box, zorder=10)
        for name, sx, sz in r["stations"]:
            ax.scatter(sx, sz, marker="v", s=70, fc="white", ec="black", lw=0.75, zorder=9, clip_on=False)
            ax.annotate(name, xy=(sx, sz), xytext=(0, 7), textcoords="offset points",
                        ha="center" if name == "G2" else "left", va="bottom", bbox=box,
                        annotation_clip=False, zorder=10)

        ax.text(0.018, 0.025, f"({'abcdef'[2 * row + col]})", transform=ax.transAxes,
                fontweight="bold", bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"), zorder=20)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(bottom, top)
        ax.set_aspect("equal")
        ax.spines["top"].set_visible(False)
        ax.tick_params(direction="out", length=4, width=0.9, pad=3)
        ax.label_outer()

    position = axes[0, col].get_position()
    fig.text(np.mean(position.intervalx), 1 - 0.7 / height, titles[number], ha="center", va="center")

for row, t in enumerate(SNAPSHOT_TIMES):
    position = axes[row, 0].get_position()
    fig.text(left - 0.045, np.mean(position.intervaly), f"t = {t:.2f} s", ha="center", va="center",
             rotation=90, fontweight="bold")

# black lines separating the six panels
fig.canvas.draw()
xs = [ax.get_position() for ax in axes[0]]
ys = [ax.get_position() for ax in axes[:, 0]]
x_mid = (xs[0].x1 + xs[1].x0) / 2
fig.add_artist(plt.Line2D([x_mid, x_mid], [ys[2].y0, ys[0].y1], color="black", lw=2.0))
for upper, lower in zip(ys[:-1], ys[1:]):
    y_mid = (upper.y0 + lower.y1) / 2
    fig.add_artist(plt.Line2D([xs[0].x0, xs[1].x1], [y_mid, y_mid], color="black", lw=2.0))

bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), orientation="horizontal",
                   cax=fig.add_axes([left, 0.65 / height, right - left, 0.2 / height]))
bar.set_ticks([-1, -0.5, 0, 0.5, 1])
bar.set_label("Normalised vertical displacement", labelpad=10)

fig.supxlabel("Distance from Shot 17 [m] | South (−), North (+)", x=(left + right) / 2, y=1.12 / height, fontsize=FS)
fig.supylabel("Depth below reference surface [m]", x=0.025, fontsize=FS)
fig.suptitle("Shot 17 | Wavefield snapshots", x=(left + right) / 2, y=1 - 0.12 / height, fontsize=FS, va="top")

fig.savefig("Figure11.png", dpi=105, bbox_inches="tight", facecolor="white")
